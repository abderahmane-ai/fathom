"""VEGA — Vertical EMA Gated Attention.

Depth memory cell that maintains a linear-attention EMA state across layers.
Heads are split into fast (short depth-horizon) and slow (long depth-horizon)
groups so the model can simultaneously track local and global depth context.

Math (per sublayer at depth position pos):
    K  = key_proj(y),   V  = val_proj(y),  Q  = query_proj(y)
    g  = σ(write_gate(y))                       -- write gate

    K_dep = K + key_bias[pos],   Q_dep = Q + query_bias[pos]
    φ(x)  = ELU(x) + 1                          -- positive feature map

    # Linear-attention retrieval from previous state
    c         = φ(Q_dep) S_prev / (φ(Q_dep) z_prev + ε)
    c_fast    = c[:n_fast_heads, :]
    c_slow    = c[n_fast_heads:, :]
    c_out     = out_fast(norm_fast(c_fast)) + out_slow(norm_slow(c_slow))

    # Separate per-timescale read gates (the key architectural feature)
    r_fast = σ(read_fast(y)),  r_slow = σ(read_slow(y))
    damp   = σ(damp_weight * y + damp_bias)

    h_new = damp * h_prev + y + r_fast * c_out_fast + r_slow * c_out_slow

    # EMA state update
    α      = σ(decay[pos])                      -- per-head per-rank decay
    S_new  = α[..., None] * S_prev + φ(K_dep)[..., None] * (g * V)[..., None, :]
    z_new  = α * z_prev + φ(K_dep)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


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

        self.d_model      = d_model
        self.num_layers   = num_layers
        self.num_sublayers = num_layers * 2
        self.rank         = rank
        self.n_heads      = n_heads
        self.n_fast_heads = n_fast_heads
        self.r_head       = rank // n_heads
        self.eps          = eps

        # Projections into the rank-dimensional EMA space.
        # Fused Q, K, V projection to reduce GPU kernel launch overhead.
        self.qkv_proj   = nn.Linear(d_model, 3 * rank, bias=False)
        self.write_gate = nn.Linear(d_model, rank, bias=True)

        # Fused read gates to reduce GPU kernel launch overhead.
        self.read_proj = nn.Linear(d_model, 2 * d_model, bias=True)

        # Element-wise damp gate: σ(damp_weight ⊙ y + damp_bias).
        self.damp_weight = nn.Parameter(torch.empty(d_model))
        self.damp_bias   = nn.Parameter(torch.full((d_model,), damp_gate_bias))

        n_slow_heads = n_heads - n_fast_heads
        self.out_fast  = nn.Linear(n_fast_heads * self.r_head, d_model, bias=False)
        self.out_slow  = nn.Linear(n_slow_heads  * self.r_head, d_model, bias=False)
        self.norm_fast = RMSNorm(n_fast_heads * self.r_head, eps=eps)
        self.norm_slow = RMSNorm(n_slow_heads  * self.r_head, eps=eps)

        # Per-sublayer depth biases — allow each depth position to specialize its
        # query/key bias without changing the projection weights.
        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))
        self.key_bias   = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        # Decay logits for the EMA.  Initialized log-linearly so fast heads have
        # short depth horizons and slow heads have long ones.
        self.decay = nn.Parameter(torch.empty(self.num_sublayers, n_heads, self.r_head))
        with torch.no_grad():
            for pos in range(self.num_sublayers):
                fast_logits = torch.linspace(*fast_decay_range, n_fast_heads * self.r_head)
                slow_logits = torch.linspace(*slow_decay_range, n_slow_heads  * self.r_head)
                self.decay[pos, :n_fast_heads]  = fast_logits.view(n_fast_heads, self.r_head)
                self.decay[pos, n_fast_heads:]  = slow_logits.view(n_slow_heads,  self.r_head)

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

        # Zero-init output projections so the cell starts as a standard residual.
        nn.init.zeros_(self.out_fast.weight)
        nn.init.zeros_(self.out_slow.weight)

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

        # Project into the EMA space.
        qkv = self.qkv_proj(y)
        Q, K, V = qkv.chunk(3, dim=-1)

        K = K.view(B, Seq, self.n_heads, self.r_head)
        V = V.view(B, Seq, self.n_heads, self.r_head)
        Q = Q.view(B, Seq, self.n_heads, self.r_head)
        g = torch.sigmoid(self.write_gate(y)).view(B, Seq, self.n_heads, self.r_head)

        # Add per-sublayer depth biases.
        K_dep = K + self.key_bias[pos].view(1, 1, self.n_heads, self.r_head)
        Q_dep = Q + self.query_bias[pos].view(1, 1, self.n_heads, self.r_head)

        # Positive feature map φ(x) = ELU(x) + 1 (ensures state positivity).
        K_phi = (F.elu(K_dep) + 1.0).float()
        Q_phi = (F.elu(Q_dep) + 1.0).float()

        # Linear-attention retrieval: c = Q_phi S_prev / (Q_phi z_prev + ε)
        num = torch.matmul(Q_phi.unsqueeze(-2), S_prev.float()).squeeze(-2)
        den = (Q_phi * z_prev.float()).sum(-1, keepdim=True).clamp(min=self.eps)
        c = (num / den).to(y.dtype)  # (B, Seq, n_heads, r_head)

        # Split retrieval by timescale and apply separate output projections.
        c_fast = c[:, :, :self.n_fast_heads, :].contiguous().view(B, Seq, -1)
        c_slow = c[:, :, self.n_fast_heads:, :].contiguous().view(B, Seq, -1)
        c_out_fast = self.out_fast(self.norm_fast(c_fast))
        c_out_slow = self.out_slow(self.norm_slow(c_slow))

        # Separate read gates per timescale — the key distinction from a single gate.
        read_gates = self.read_proj(y)
        r_fast, r_slow = read_gates.chunk(2, dim=-1)
        r_fast = torch.sigmoid(r_fast)
        r_slow = torch.sigmoid(r_slow)
        damp   = torch.sigmoid(self.damp_weight * y + self.damp_bias)

        h_new = damp * h_prev + y + r_fast * c_out_fast + r_slow * c_out_slow

        # EMA state update: S_new = α S_prev + φ(K) ⊗ (g * V)
        decay = torch.sigmoid(self.decay[pos]).view(1, 1, self.n_heads, self.r_head)
        outer = K_phi.unsqueeze(-1) * (g * V.float()).unsqueeze(-2)
        S_new = decay.unsqueeze(-1) * S_prev + outer
        z_new = decay * z_prev + K_phi

        return h_new, (S_new, z_new)
