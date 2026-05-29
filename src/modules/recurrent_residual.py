import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused RMSNorm for torch.compile
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x

class RecurrentResidualCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        read_gate_bias: float = -3.0,
        forget_gate_bias: float = 3.0,
        update_gate_bias: float = -2.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
        memory_gain_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps

        # Read Gate
        self.read_weight = nn.Parameter(torch.empty(d_model))
        self.read_bias = nn.Parameter(torch.full((d_model,), read_gate_bias))
        
        # Forget Gate (High bias = retain history)
        self.forget_weight = nn.Parameter(torch.empty(d_model))
        self.forget_bias = nn.Parameter(torch.full((d_model,), forget_gate_bias))
        
        # Update Gate (Low bias = write slowly)
        self.update_weight = nn.Parameter(torch.empty(d_model))
        self.update_bias = nn.Parameter(torch.full((d_model,), update_gate_bias))
        
        # Memory Gain (Zero-start protocol)
        self.memory_gain = nn.Parameter(torch.full((d_model,), memory_gain_init))
        
        # Depth Bias
        self.depth_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        
        # Initial Memory State
        self.m_init = nn.Parameter(torch.zeros(d_model))
        
        # Normalization for the READ path only (Fixes Magnitude Blindness)
        self.memory_norm = RMSNorm(d_model, eps=eps)
        self.h_norm = RMSNorm(d_model, eps=eps)

        # Initializations
        nn.init.normal_(self.read_weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.forget_weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.update_weight, mean=0.0, std=gate_init_std)

    @property
    def parameter_count_formula(self) -> int:
        """Return the methodology parameter count."""
        return (self.num_sublayers + 8) * self.d_model

    def get_initial_state(self, batch_size: int, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        # Expand learnable initial memory to batch and sequence dimensions
        target_device = device if device is not None else self.m_init.device
        return self.m_init.to(target_device).view(1, 1, -1).expand(batch_size, seq_len, -1).clone()

    def _sublayer_position(self, layer_idx: int, sublayer: int) -> int:
        """Map a layer/sublayer pair to a depth-bias row.

        Args:
            layer_idx: Zero-based transformer layer index.
            sublayer: ``0`` for attention and ``1`` for FFN.

        Returns:
            Row index in ``depth_bias``.

        Preconditions:
            ``0 <= layer_idx < num_layers`` and ``sublayer`` is ``0`` or ``1``.
        """
        if sublayer not in (0, 1):
            raise ValueError(f"sublayer must be 0 or 1, got {sublayer}.")
        position = layer_idx * 2 + sublayer
        if position >= self.num_sublayers:
            raise IndexError(f"sublayer position {position} exceeds {self.num_sublayers} entries.")
        return position

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        layer_idx: int,
        sublayer: int = 0,
        h_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._sublayer_position(layer_idx, sublayer)

        if h_norm is None:
            # Use RMSNorm instead of LayerNorm to match modern backbones
            h_norm = self.h_norm(h_prev)
        
        # 1. Read Path
        read_gate = torch.sigmoid(self.read_weight * h_norm + self.read_bias)
        # Normalize memory ONLY when reading to preserve accumulated magnitude in storage
        memory_read = self.memory_gain * self.memory_norm(m) 
        
        # Inject memory into hidden state
        h_new = h_prev + y + read_gate * memory_read

        # 2. Update Path
        # Forget gate decides what to keep from the past
        forget_gate = torch.sigmoid(self.forget_weight * y + self.forget_bias)
        # Update gate decides what to write from the present
        update_gate = torch.sigmoid(
            self.update_weight * y + self.update_bias + self.depth_bias[position]
        )
        
        # Do NOT apply RMSNorm here. Let the memory accumulate natural magnitude.
        m_new = (forget_gate * m) + (update_gate * y)

        return h_new, m_new
