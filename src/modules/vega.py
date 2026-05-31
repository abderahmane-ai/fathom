"""VEGA — Vertical EMA Gated Attention.

Depth memory cell that maintains a linear-attention EMA state across layers.
Under Lean VEGA*, the state size is conditional: it uses a Vector State if r_head <= 64,
and falls back to a Matrix State for larger ranks.

Math (per sublayer at depth position pos):
    y_norm = RMSNorm(y)
    K  = key_proj(y_norm),   V  = val_proj(y_norm),  Q  = query_proj(y_norm)
    g  = σ(write_gate(y_norm))                  -- write gate

    K_dep = RMSNorm(K) / sqrt(r_head)
    Q_dep = RMSNorm(Q + query_bias[pos]) / sqrt(r_head)
    φ(x)  = ELU(x) + 1                          -- positive feature map

    # Linear-attention retrieval from previous state
    # If r_head <= 64 (Vector State):
        c = (φ(Q_dep) * S_prev) / (sum(φ(Q_dep) * z_prev) + ε)
    # Else (Matrix State):
        c = φ(Q_dep) S_prev / (φ(Q_dep) z_prev + ε)

    c_out = out_proj(norm_c(c))

    # Single read gate and damp gate
    r    = σ(read_proj(y_norm))
    damp = σ(damp_weight * y_norm + damp_bias)

    h_new = damp * h_prev + y + r * c_out

    # EMA state update (per-head per-rank decay logits decay[pos])
    α = σ(decay[pos])
    # If r_head <= 64 (Vector State):
        S_new = α * S_prev + φ(K_dep) * (g * V)
    # Else (Matrix State):
        S_new = α[..., None] * S_prev + φ(K_dep)[..., None] * (g * V)[..., None, :]
    z_new = α * z_prev + φ(K_dep)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class _ParameterFreeRMSNorm(nn.Module):
    """Parameter-free RMSNorm used only inside VEGACell."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class VEGACell(nn.Module):
    """VEGA depth-memory cell, shared (weight-tied) across all transformer layers.

    Args:
        d_model: Hidden dimension.
        num_layers: Number of transformer layers (used to size depth biases).
        rank: Total rank of the linear-attention state (must be divisible by n_heads).
        n_heads: Number of EMA heads.
        n_fast_heads: How many heads are in the fast (short-horizon) group.
        fast_decay_range: (min, max) logit range for fast-head decay initialization.
        slow_decay_range: (min, max) logit range for slow-head decay initialization.
        read_gate_bias: Initial bias for both read gates (negative → gates start closed).
        write_gate_bias: Initial bias for the write gate.
        damp_gate_bias: Initial value for the damp bias vector.
        eps: Stability epsilon for the denominator in the linear-attention retrieval.
        gate_init_std: Std for gate weight initialization.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        rank: int = 32,
        n_heads: int = 4,
        n_fast_heads: int = 2,
        fast_decay_range: tuple[float, float] = (0.0, 1.2),
        slow_decay_range: tuple[float, float] = (2.0, 4.5),
        read_gate_bias: float = -3.0,
        write_gate_bias: float = -2.0,
        damp_gate_bias: float = 3.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()

        assert rank % n_heads == 0, "rank must be divisible by n_heads"
        assert 0 < n_fast_heads < n_heads, "n_fast_heads must be in (0, n_heads)"

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.rank = rank
        self.n_heads = n_heads
        self.n_fast_heads = n_fast_heads
        self.r_head = rank // n_heads
        self.eps = eps
        self.y_norm = _ParameterFreeRMSNorm(eps=eps)

        # Conditional state type based on head rank (Lean VEGA*)
        self.use_vector_state = (self.r_head <= 64)

        # Projections into the rank-dimensional EMA space.
        # Fused Q, K, V projection to reduce GPU kernel launch overhead.
        self.qkv_proj = nn.Linear(d_model, 3 * rank, bias=False)
        self.write_gate = nn.Linear(d_model, rank, bias=True)

        # Single read gate projection.
        self.read_proj = nn.Linear(d_model, d_model, bias=True)

        # Element-wise damp gate: σ(damp_weight ⊙ y + damp_bias).
        self.damp_weight = nn.Parameter(torch.empty(d_model))
        self.damp_bias = nn.Parameter(torch.full((d_model,), damp_gate_bias))

        # Single output projection and RMSNorm.
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.norm_c = RMSNorm(rank, eps=eps)

        # Per-sublayer depth biases — query bias only (Lean VEGA* Cut 4).
        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        # Decay logits for the EMA. Spaced log-linearly within head groups (Lean VEGA*).
        self.decay = nn.Parameter(torch.empty(self.num_sublayers, n_heads, self.r_head))
        n_slow_heads = n_heads - n_fast_heads
        with torch.no_grad():
            fast_logits = torch.linspace(
                fast_decay_range[0], fast_decay_range[1], n_fast_heads * self.r_head
            )
            slow_logits = torch.linspace(
                slow_decay_range[0], slow_decay_range[1], n_slow_heads * self.r_head
            )
            all_logits = torch.cat([fast_logits, slow_logits])
            self.decay.copy_(all_logits.view(1, self.n_heads, self.r_head).expand(
                self.num_sublayers, -1, -1
            ))

        # Orthogonal init for the EMA projections — improves conditioning of the state.
        with torch.no_grad():
            q_init = torch.empty(rank, d_model)
            k_init = torch.empty(rank, d_model)
            v_init = torch.empty(rank, d_model)
            nn.init.orthogonal_(q_init)
            nn.init.orthogonal_(k_init)
            nn.init.orthogonal_(v_init)
            self.qkv_proj.weight.copy_(torch.cat([q_init, k_init, v_init], dim=0))

        # Gate inits: tiny weights so biases dominate at the start.
        nn.init.normal_(self.write_gate.weight, 0.0, gate_init_std)
        self.write_gate.bias.data.fill_(write_gate_bias)

        nn.init.normal_(self.read_proj.weight, 0.0, gate_init_std)
        self.read_proj.bias.data.fill_(read_gate_bias)

        nn.init.normal_(self.damp_weight, 0.0, gate_init_std)

        # Zero-init output projection so the cell starts as a standard residual.
        nn.init.zeros_(self.out_proj.weight)

    @property
    def key_proj(self) -> nn.Linear:
        """Alias for qkv_proj (kept for device lookup in tests)."""
        return self.qkv_proj

    def get_initial_state(
        self, B: int, S: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (S_0, z_0) state tensors.

        Args:
            B: Batch size.
            S: Sequence length.
            device: Target device.

        Returns:
            ``(S_state, z_state)`` — EMA covariance and normalization state.
        """
        if self.use_vector_state:
            S0 = torch.zeros(
                B, S, self.n_heads, self.r_head, device=device, dtype=torch.float32
            )
        else:
            S0 = torch.zeros(
                B, S, self.n_heads, self.r_head, self.r_head, device=device, dtype=torch.float32
            )
        z0 = torch.zeros(B, S, self.n_heads, self.r_head, device=device, dtype=torch.float32)
        return S0, z0

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        sublayer: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one VEGA depth step.

        Args:
            h_prev: Hidden state entering this sublayer, shape (B, S, d_model).
            y: Sublayer output (attention or FFN), shape (B, S, d_model).
            state: ``(S_prev, z_prev)`` EMA state from the previous sublayer.
            layer_idx: 0-based index of the enclosing transformer layer.
            sublayer: 0 for the attention sublayer, 1 for the FFN sublayer.

        Returns:
            ``(h_new, (S_new, z_new))`` — updated hidden state and EMA state.
        """
        S_prev, z_prev = state
        pos = layer_idx * 2 + sublayer
        B, Seq, _ = y.shape
        y_norm = self.y_norm(y)

        # Project into the EMA space.
        qkv = self.qkv_proj(y_norm)
        Q, K, V = qkv.chunk(3, dim=-1)

        K = K.view(B, Seq, self.n_heads, self.r_head)
        V = V.view(B, Seq, self.n_heads, self.r_head)
        Q = Q.view(B, Seq, self.n_heads, self.r_head)
        g = torch.sigmoid(self.write_gate(y_norm)).view(B, Seq, self.n_heads, self.r_head)

        # Add per-sublayer query depth bias (Lean VEGA* Cut 4).
        Q_dep = Q + self.query_bias[pos].view(1, 1, self.n_heads, self.r_head)
        K_dep = K

        # Normalize and scale query and key vectors before the positive feature map.
        K_scale = torch.rsqrt(K_dep.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        K_dep = K_dep * K_scale / (self.r_head ** 0.5)

        Q_scale = torch.rsqrt(Q_dep.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        Q_dep = Q_dep * Q_scale / (self.r_head ** 0.5)

        # Positive feature map φ(x) = ELU(x) + 1 (ensures state positivity).
        K_phi = (F.elu(K_dep) + 1.0).float()
        Q_phi = (F.elu(Q_dep) + 1.0).float()

        # Linear-attention retrieval.
        if self.use_vector_state:
            # Vector state: S_prev is (B, Seq, n_heads, r_head).
            # Computes element-wise retrieval c_i = (Q_i * S_i) / sum(Q_j * z_j).
            num = Q_phi * S_prev.float()
            den = (Q_phi * z_prev.float()).sum(-1, keepdim=True).clamp(min=self.eps)
            c = (num / den).to(y.dtype)
        else:
            # Matrix state: S_prev is (B, Seq, n_heads, r_head, r_head).
            # Computes matrix retrieval c = (Q^T * S) / (Q^T * z).
            num = torch.matmul(Q_phi.unsqueeze(-2), S_prev.float()).squeeze(-2)
            den = (Q_phi * z_prev.float()).sum(-1, keepdim=True).clamp(min=self.eps)
            c = (num / den).to(y.dtype)

        # Single output projection and RMSNorm.
        c_flat = c.contiguous().view(B, Seq, self.rank)
        c_out = self.out_proj(self.norm_c(c_flat))

        # Single read gate and damp gate.
        read_gate = torch.sigmoid(self.read_proj(y_norm))
        damp = torch.sigmoid(self.damp_weight * y_norm + self.damp_bias)

        h_new = damp * h_prev + y + read_gate * c_out

        # EMA state update.
        # Note: decay has shape (1, 1, n_heads, r_head). In matrix state, we unsqueeze
        # it to decay.unsqueeze(-1) of shape (1, 1, n_heads, r_head, 1) to match S_prev.
        # z_prev is always a vector and matches decay directly.
        decay = torch.sigmoid(self.decay[pos]).view(1, 1, self.n_heads, self.r_head)
        if self.use_vector_state:
            S_new = decay * S_prev + K_phi * (g * V.float())
        else:
            outer = K_phi.unsqueeze(-1) * (g * V.float()).unsqueeze(-2)
            S_new = decay.unsqueeze(-1) * S_prev + outer
        z_new = decay * z_prev + K_phi

        return h_new, (S_new, z_new)
