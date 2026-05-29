import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        damp_gate_bias: float = 3.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
        memory_gain_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps
        rank = max(32, d_model // 8)

        def make_gate():
            return nn.Sequential(
                nn.Linear(d_model, rank, bias=False),
                nn.Linear(rank, d_model, bias=True)
            )

        self.read_proj = make_gate()
        self.forget_proj = make_gate()
        self.update_proj = make_gate()
        self.damp_proj = make_gate()
        
        self.memory_gain = nn.Parameter(torch.full((d_model,), memory_gain_init))
        
        self.depth_read_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_forget_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_update_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_damp_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        
        self.m_init = nn.Parameter(torch.zeros(d_model))
        
        self.memory_norm = RMSNorm(d_model, eps=eps)
        self.memory_out = nn.Linear(d_model, d_model, bias=False)
        self.h_norm = RMSNorm(d_model, eps=eps)

        for proj in [self.read_proj, self.forget_proj, self.update_proj, self.damp_proj]:
            nn.init.normal_(proj[0].weight, std=gate_init_std)
            nn.init.normal_(proj[1].weight, std=gate_init_std)
            nn.init.zeros_(proj[1].bias)
            
        with torch.no_grad():
            self.read_proj[1].bias.fill_(read_gate_bias)
            self.forget_proj[1].bias.fill_(forget_gate_bias)
            self.update_proj[1].bias.fill_(update_gate_bias)
            self.damp_proj[1].bias.fill_(damp_gate_bias)

    @property
    def parameter_count_formula(self) -> int:
        return (self.num_sublayers * 4 + 10) * self.d_model

    def get_initial_state(self, batch_size: int, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        target_device = device if device is not None else self.m_init.device
        return self.m_init.to(target_device).view(1, 1, -1).expand(batch_size, seq_len, -1).clone()

    def _sublayer_position(self, layer_idx: int, sublayer: int) -> int:
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
        m_norm = self.memory_norm(m)

        # Gate computations
        read_gate   = torch.sigmoid(self.read_proj(y)   + self.depth_read_bias[position])
        damp_gate   = torch.sigmoid(self.damp_proj(y)   + self.depth_damp_bias[position])
        forget_gate = torch.sigmoid(self.forget_proj(m_norm) + self.depth_forget_bias[position])
        update_gate = torch.sigmoid(self.update_proj(y) + self.depth_update_bias[position])

        memory_read = self.memory_gain * self.memory_out(m_norm) 
        h_new = damp_gate * h_prev + y + read_gate * memory_read
        m_new = forget_gate * m + update_gate * y

        return h_new, m_new
