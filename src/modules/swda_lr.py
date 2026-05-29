"""Sliding-Window Depth Attention with Low-Rank History (SWDA-LR).

This cell implements the SWDA-LR residual transition. It maintains a sliding window
FIFO buffer for exact local routing and a low-rank running covariance state for
deep historical retrieval. At initialization, memory read weights are zero, matching
a standard Pre-LN residual addition.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped

class RMSNorm(nn.Module):
    """Parameter-free root mean square normalization.

    Args:
        d_model: Hidden dimension used to validate the input shape.
        eps: Numerical stability constant.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` by its RMS over the final dimension.

        Args:
            x: Tensor with final dimension ``d_model``.

        Returns:
            RMS-normalized tensor with the same shape as ``x``.
        """
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.scale * x

class SWDALRCell(nn.Module):
    """Sliding-Window Depth Attention with Low-Rank History (SWDA-LR) cell.

    Args:
        d_model: Hidden dimension ``d``.
        num_layers: Total transformer layers; allocates depth biases per layer.
        window_size: Number of previous sublayer outputs kept in FIFO.
        rank: Rank of low-rank history projection.
        n_heads: Number of heads for deep memory.
        v_dim: Optional compression dimension for Values. If None, defaults to d_model.
        decay_bias_init: Initial value for decay bias logits.
        read_gate_bias: Initial read-gate bias.
        write_gate_bias: Initial write-gate bias.
        eps: Epsilon for RMSNorm.
        gate_init_std: Standard deviation for diagonal gate weights.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        window_size: int = 8,
        rank: int = 16,
        n_heads: int = 4,
        v_dim: int | None = None,
        decay_bias_init: float = 3.0,
        read_gate_bias: float = -3.0,
        write_gate_bias: float = -2.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.window_size = window_size
        self.rank = rank
        self.n_heads = n_heads
        self.v_dim = v_dim if v_dim is not None else d_model
        self.eps = eps

        if rank % n_heads != 0:
            raise ValueError(f"rank ({rank}) must be divisible by n_heads ({n_heads}).")
        if self.v_dim % n_heads != 0:
            raise ValueError(f"v_dim ({self.v_dim}) must be divisible by n_heads ({n_heads}).")

        self.r_head = rank // n_heads
        self.d_head = self.v_dim // n_heads

        # Fused Read and Write Gate parameters (Row 0: Read, Row 1: Write)
        self.gate_weights = nn.Parameter(torch.empty(2, d_model))
        self.gate_biases = nn.Parameter(torch.empty(2, d_model))

        # State decay parameters (SSM-style)
        self.decay_bias = nn.Parameter(torch.empty(self.num_sublayers, rank))
        self.key_decay_bias = nn.Parameter(torch.empty(self.num_sublayers, rank))

        # Logarithmic distribution timescale initialization for decay gates
        with torch.no_grad():
            for pos in range(self.num_sublayers):
                min_ts = 1.0
                max_ts = float(self.num_sublayers)
                ts = torch.exp(torch.linspace(math.log(min_ts), math.log(max_ts), rank))
                alpha_init = 1.0 - (1.0 / ts)
                alpha_init = torch.clamp(alpha_init, 0.001, 0.999)
                decay_val = torch.log(alpha_init / (1.0 - alpha_init))
                self.decay_bias[pos].copy_(decay_val)
                self.key_decay_bias[pos].copy_(decay_val)

        # Learned Depth Pseudo-Queries (initialized to zero)
        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        # Local Projections
        self.q_local_proj = nn.Linear(d_model, d_model, bias=False)
        self.local_norm = RMSNorm(d_model, eps=eps)

        # RMSNorm for input path
        self.h_norm = RMSNorm(d_model, eps=eps)

        # Relative Depth Bias for FIFO attention
        self.fifo_depth_bias = nn.Parameter(torch.zeros(window_size))

        # Register the arange mask as a non-persistent buffer
        self.register_buffer(
            "window_indices",
            torch.arange(window_size).view(1, 1, -1),
            persistent=False,
        )

        # Fused Deep Projections (MLA-inspired)
        self.kv_deep_proj = nn.Linear(d_model, rank + self.v_dim, bias=False)
        self.q_deep_proj = nn.Linear(d_model, rank, bias=False)
        self.deep_norm = RMSNorm(self.v_dim, eps=eps)

        # Back-projection if value dimension is compressed
        if self.v_dim != d_model:
            self.out_proj = nn.Linear(self.v_dim, d_model, bias=False)
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=gate_init_std)
        else:
            # pyrefly: ignore [bad-assignment]
            self.out_proj = nn.Identity()

        # Initializations
        with torch.no_grad():
            self.gate_weights[0].normal_(mean=0.0, std=gate_init_std)
            self.gate_weights[1].normal_(mean=0.0, std=gate_init_std)
            self.gate_biases[0].fill_(read_gate_bias)
            self.gate_biases[1].fill_(write_gate_bias)

        # Orthogonal Initialization for deep projections
        nn.init.orthogonal_(self.kv_deep_proj.weight)
        nn.init.orthogonal_(self.q_deep_proj.weight)
        nn.init.normal_(self.q_local_proj.weight, mean=0.0, std=gate_init_std)

        self.last_read_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)
        self.last_decay_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)

    def get_initial_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the initial states: empty FIFO tensors, zero step index, zero S, zero z.

        Args:
            batch_size: Batch size ``B``.
            seq_len: Sequence length ``S``.
            device: Optional target device.

        Returns:
            Tuple of (FIFO tensor buffer, FIFO norm buffer, step index, S_init, z_init).
        """
        target_device = self.gate_weights.device if device is None else device
        fifo_buf = torch.zeros(
            batch_size, seq_len, self.window_size, self.d_model, device=target_device, dtype=torch.float32
        )
        fifo_norm_buf = torch.zeros(
            batch_size, seq_len, self.window_size, self.d_model, device=target_device, dtype=torch.float32
        )
        fifo_idx = torch.tensor(0, device=target_device, dtype=torch.long)
        
        # S shape: (B, S, n_heads, r_head, d_head). Kept in FP32 for stable accumulation.
        S_init = torch.zeros(
            batch_size, seq_len, self.n_heads, self.r_head, self.d_head, device=target_device, dtype=torch.float32
        )
        # z shape: (B, S, n_heads, r_head). Kept in FP32.
        z_init = torch.zeros(
            batch_size, seq_len, self.n_heads, self.r_head, device=target_device, dtype=torch.float32
        )
        return fifo_buf, fifo_norm_buf, fifo_idx, S_init, z_init

    def _sublayer_position(self, layer_idx: int, sublayer: int) -> int:
        """Map a layer/sublayer pair to a depth-bias row."""
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
        m: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        layer_idx: int,
        sublayer: int = 0,
        h_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute one SWDA-LR transition.

        Args:
            h_prev: Hidden state entering the sublayer, shape ``(B, S, d_model)``.
            y: Sublayer output, shape ``(B, S, d_model)``.
            m: Memory state tuple ``(fifo_buf, fifo_norm_buf, fifo_idx, S_prev, z_prev)``.
            layer_idx: Zero-based transformer layer index.
            sublayer: ``0`` for attention and ``1`` for FFN.
            h_norm: Optional pre-computed normalized representation of ``h_prev``.

        Returns:
            Tuple ``(h_new, m_new)``.
        """
        fifo_buf, fifo_norm_buf, fifo_idx, S_prev, z_prev = m
        position = self._sublayer_position(layer_idx, sublayer)

        if h_norm is None:
            # Use custom RMSNorm
            h_norm = self.h_norm(h_prev)

        B, S, d = h_prev.shape

        # Fused Read/Write Gates: Stack inputs and evaluate in one call
        gate_inputs = torch.stack([h_norm, y], dim=0)
        gates = torch.sigmoid(
            gate_inputs * self.gate_weights.view(2, 1, 1, -1) + self.gate_biases.view(2, 1, 1, -1)
        )
        read_gate, write_gate = gates[0], gates[1]
        self.last_read_gate = read_gate.detach().mean()

        # Local FIFO sliding window attention
        write_idx = fifo_idx % self.window_size
        
        # Clone buffers to prevent autograd in-place modification error
        fifo_buf = fifo_buf.clone()
        fifo_norm_buf = fifo_norm_buf.clone()
        
        # In-place FIFO write
        fifo_buf[:, :, write_idx, :] = y.to(fifo_buf.dtype)
        # Cache FIFO norm_buf
        fifo_norm_buf[:, :, write_idx, :] = self.local_norm(y).to(fifo_norm_buf.dtype)
        
        next_idx = fifo_idx + 1
        keys_local = fifo_norm_buf

        q_local = self.q_local_proj(y)  # shape: (B, S, d)

        # Matmul-based attention instead of einsum
        logits = torch.matmul(keys_local, q_local.unsqueeze(-1)).squeeze(-1)  # shape: (B, S, W)
        logits = logits / math.sqrt(self.d_model)
        logits = logits + self.fifo_depth_bias.view(1, 1, -1)

        # Causal mask for unfilled buffer slots in early layers using registered window_indices buffer
        is_unfilled = (fifo_idx < self.window_size - 1).view(1, 1, 1)
        # pyrefly: ignore [unsupported-operation]
        mask = torch.where((self.window_indices <= write_idx.view(1, 1, 1)) | ~is_unfilled, 0.0, float("-inf"))
        logits = logits + mask

        weights = torch.softmax(logits, dim=-1)

        # Matmul-based context aggregation instead of einsum
        c_local = torch.matmul(weights.unsqueeze(-2), fifo_buf).squeeze(-2)  # shape: (B, S, d)

        # Deep multi-head low-rank history retrieval (fused K, V projection; separate Q projection)
        kv = self.kv_deep_proj(y)
        K, V = kv.split([self.rank, self.v_dim], dim=-1)
        Q = self.q_deep_proj(h_norm)

        # Reshape to multi-head format
        K_reshaped = K.view(B, S, self.n_heads, self.r_head)
        Q_reshaped = Q.view(B, S, self.n_heads, self.r_head)
        V_reshaped = V.view(B, S, self.n_heads, self.d_head)

        # Apply Learned Depth Pseudo-Queries to the query projection
        Q_biased = Q_reshaped + self.query_bias[position].view(1, 1, self.n_heads, self.r_head)

        # Positivity constraint for unconditionally stable linear attention
        K_phi = F.elu(K_reshaped) + 1.0
        Q_phi = F.elu(Q_biased) + 1.0

        # Cast to float32 for stable recurrent calculations
        Q_phi_f = Q_phi.float()
        K_phi_f = K_phi.float()
        
        # Apply Write Gate to Value projection before deep storage
        if self.v_dim == self.d_model:
            V_gated = (V * write_gate).view(B, S, self.n_heads, self.d_head)
        else:
            write_gate_v = torch.matmul(write_gate, self.out_proj.weight)
            V_gated = (V * write_gate_v).view(B, S, self.n_heads, self.d_head)
        V_gated_f = V_gated.float()

        # Numerator: Q_phi (B, S, H, 1, r_head) @ S_prev (B, S, H, r_head, d_head) -> (B, S, H, 1, d_head) -> squeeze
        num = torch.matmul(Q_phi_f.unsqueeze(-2), S_prev).squeeze(-2)

        # Denominator: Q_phi_f * z_prev -> sum over r_head -> shape (B, S, H, 1)
        den = torch.sum(Q_phi_f * z_prev, dim=-1, keepdim=True)
        c_deep = num / (den + self.eps) # (B, S, H, d_head)

        # Reshape back to combined value dimension
        c_deep = c_deep.view(B, S, self.v_dim)

        # Apply parameter-free RMSNorm on the retrieved deep context
        c_deep = self.deep_norm(c_deep)

        # Project back to d_model if v_dim was compressed
        c_deep = self.out_proj(c_deep.to(y.dtype))

        # Memory Injection (memory_gain has been removed as redundant)
        h_new = h_prev + y + read_gate * (c_local + c_deep)

        # State updates
        decay = torch.sigmoid(self.decay_bias[position]).view(self.n_heads, self.r_head)  # shape: (H, r_head)
        key_decay = torch.sigmoid(self.key_decay_bias[position]).view(self.n_heads, self.r_head)  # shape: (H, r_head)
        self.last_decay_gate = decay.detach().mean()

        # Update running covariance state S
        # outer: K_phi_f (B, S, H, r_head, 1) @ V_gated_f (B, S, H, 1, d_head) -> (B, S, H, r_head, d_head)
        outer = torch.matmul(K_phi_f.unsqueeze(-1), V_gated_f.unsqueeze(-2))
        
        # Fused state updates using torch.addcmul
        S_new = torch.addcmul(
            outer,
            decay.view(1, 1, self.n_heads, self.r_head, 1),
            S_prev
        )

        # Update running key-sum normalizer z
        # Fused normalizer updates using torch.addcmul
        z_new = torch.addcmul(
            K_phi_f,
            key_decay.view(1, 1, self.n_heads, self.r_head),
            z_prev
        )

        return h_new, (fifo_buf, fifo_norm_buf, next_idx, S_new, z_new)
