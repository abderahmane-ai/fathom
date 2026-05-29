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
        v_dim: Optional compression dimension for Values. If None, defaults to d_model.
        decay_bias_init: Initial value for decay bias logits.
        read_gate_bias: Initial read-gate bias.
        write_gate_bias: Initial write-gate bias.
        eps: Epsilon for RMSNorm and LayerNorm.
        gate_init_std: Standard deviation for diagonal gate weights.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        window_size: int = 8,
        rank: int = 16,
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
        self.v_dim = v_dim if v_dim is not None else d_model
        self.eps = eps

        # Read gate parameters
        self.read_weight = nn.Parameter(torch.empty(d_model))
        self.read_bias = nn.Parameter(torch.full((d_model,), read_gate_bias))
        self.memory_gain = nn.Parameter(torch.ones(d_model))  # Zero-start via read gate

        # Write gate parameters
        self.write_weight = nn.Parameter(torch.empty(d_model))
        self.write_bias = nn.Parameter(torch.full((d_model,), write_gate_bias))

        # State decay parameters (SSM-style)
        self.decay_bias = nn.Parameter(torch.full((self.num_sublayers, rank), decay_bias_init))
        self.key_decay_bias = nn.Parameter(torch.full((self.num_sublayers, rank), decay_bias_init))

        # Local Projections
        self.q_local_proj = nn.Linear(d_model, d_model, bias=False)
        self.local_norm = RMSNorm(d_model, eps=eps)

        # Relative Depth Bias for FIFO attention
        self.fifo_depth_bias = nn.Parameter(torch.zeros(window_size))

        # Deep Projections (Corrected Dimensions)
        self.k_deep_proj = nn.Linear(d_model, rank, bias=False)  # Keys -> rank
        self.q_deep_proj = nn.Linear(d_model, rank, bias=False)  # Queries -> rank
        self.v_deep_proj = nn.Linear(d_model, self.v_dim, bias=False)  # Values -> v_dim
        self.deep_norm = RMSNorm(self.v_dim, eps=eps)

        # Back-projection if value dimension is compressed
        if self.v_dim != d_model:
            self.out_proj = nn.Linear(self.v_dim, d_model, bias=False)
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=gate_init_std)
        else:
            # pyrefly: ignore [bad-assignment]
            self.out_proj = nn.Identity()

        # Initializations
        nn.init.normal_(self.read_weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.write_weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.q_local_proj.weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.k_deep_proj.weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.q_deep_proj.weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.v_deep_proj.weight, mean=0.0, std=gate_init_std)

        self.last_read_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)
        self.last_decay_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)

    def get_initial_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the initial states: empty FIFO tensor, zero step index, zero S, zero z.

        Args:
            batch_size: Batch size ``B``.
            seq_len: Sequence length ``S``.
            device: Optional target device.

        Returns:
            Tuple of (FIFO tensor buffer, step index, S_init, z_init).
        """
        target_device = self.read_weight.device if device is None else device
        fifo_buf = torch.zeros(
            self.window_size, batch_size, seq_len, self.d_model, device=target_device, dtype=torch.float32
        )
        fifo_idx = torch.tensor(0, device=target_device, dtype=torch.long)
        # S shape: (B, S, rank, v_dim). Kept in FP32 for stable accumulation.
        S_init = torch.zeros(
            batch_size, seq_len, self.rank, self.v_dim, device=target_device, dtype=torch.float32
        )
        # z shape: (B, S, rank). Kept in FP32.
        z_init = torch.zeros(
            batch_size, seq_len, self.rank, device=target_device, dtype=torch.float32
        )
        return fifo_buf, fifo_idx, S_init, z_init

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
        m: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        layer_idx: int,
        sublayer: int = 0,
        h_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute one SWDA-LR transition.

        Args:
            h_prev: Hidden state entering the sublayer, shape ``(B, S, d_model)``.
            y: Sublayer output, shape ``(B, S, d_model)``.
            m: Memory state tuple ``(fifo_buf, fifo_idx, S_prev, z_prev)``.
            layer_idx: Zero-based transformer layer index.
            sublayer: ``0`` for attention and ``1`` for FFN.
            h_norm: Optional pre-computed normalized representation of ``h_prev``.

        Returns:
            Tuple ``(h_new, m_new)``.
        """
        fifo_buf, fifo_idx, S_prev, z_prev = m
        position = self._sublayer_position(layer_idx, sublayer)

        if h_norm is None:
            h_norm = F.layer_norm(h_prev, (self.d_model,), eps=self.eps)

        # 1. Read Gate
        read_gate = torch.sigmoid(self.read_weight * h_norm + self.read_bias)
        self.last_read_gate = read_gate.detach().mean()

        # Write Gate (Controls memory write selectivity)
        write_gate = torch.sigmoid(self.write_weight * y + self.write_bias)

        # 2. Local FIFO sliding window attention
        write_idx = fifo_idx % self.window_size
        fifo_buf = fifo_buf.index_copy(0, write_idx.unsqueeze(0), y.unsqueeze(0).to(fifo_buf.dtype))
        next_idx = fifo_idx + 1

        keys_local = self.local_norm(fifo_buf)
        q_local = self.q_local_proj(y)  # shape: (B, S, d)

        # Attention logits over the window dimension
        # logits shape: (W, B, S)
        logits = torch.einsum("b s d, w b s d -> w b s", q_local, keys_local) / math.sqrt(self.d_model)
        logits = logits + self.fifo_depth_bias.view(-1, 1, 1)

        # Causal mask for unfilled buffer slots in early layers
        indices = torch.arange(self.window_size, device=y.device).view(-1, 1, 1)
        is_unfilled = (fifo_idx < self.window_size - 1).view(1, 1, 1)
        mask = torch.where((indices <= write_idx.view(1, 1, 1)) | ~is_unfilled, 0.0, float("-inf"))
        logits = logits + mask

        weights = torch.softmax(logits, dim=0)

        # c_local shape: (B, S, d)
        c_local = torch.einsum("w b s, w b s d -> b s d", weights, fifo_buf)

        # 3. Deep low-rank history retrieval with linear attention normalization
        K = self.k_deep_proj(y)  # shape: (B, S, r)
        Q = self.q_deep_proj(h_norm)  # shape: (B, S, r)
        V = self.v_deep_proj(y)  # shape: (B, S, v_dim)

        # Positivity constraint for unconditionally stable linear attention
        K_phi = F.elu(K) + 1.0
        Q_phi = F.elu(Q) + 1.0

        # Cast to float32 for stable recurrent calculations
        Q_phi_f = Q_phi.float()
        K_phi_f = K_phi.float()
        
        # Apply Write Gate to Value projection before deep storage
        V_gated = V * write_gate
        V_gated_f = V_gated.float()

        # Numerator: Q_phi (B, S, 1, r) @ S_prev (B, S, r, v) -> (B, S, 1, v) -> squeeze to (B, S, v)
        num = torch.matmul(Q_phi_f.unsqueeze(-2), S_prev).squeeze(-2)

        # Denominator: Q_phi_f * z_prev -> sum over r -> shape (B, S, 1)
        den = torch.sum(Q_phi_f * z_prev, dim=-1, keepdim=True)
        c_deep = num / (den + self.eps)

        # Apply parameter-free RMSNorm on the retrieved deep context
        c_deep = self.deep_norm(c_deep)

        # Project back to d_model if v_dim was compressed
        c_deep = self.out_proj(c_deep.to(y.dtype))

        # 4. Memory Injection
        h_new = h_prev + y + read_gate * (self.memory_gain * (c_local + c_deep))

        # 5. State updates
        decay = torch.sigmoid(self.decay_bias[position])  # shape: (r,)
        key_decay = torch.sigmoid(self.key_decay_bias[position])  # shape: (r,)
        self.last_decay_gate = decay.detach().mean()

        # Update running covariance state S
        # outer: K_phi_f (B, S, r, 1) @ V_gated_f (B, S, 1, v) -> (B, S, r, v)
        outer = torch.matmul(K_phi_f.unsqueeze(-1), V_gated_f.unsqueeze(-2))
        S_new = decay.view(1, 1, -1, 1) * S_prev + outer

        # Update running key-sum normalizer z
        z_new = key_decay.view(1, 1, -1) * z_prev + K_phi_f

        return h_new, (fifo_buf, next_idx, S_new, z_new)
