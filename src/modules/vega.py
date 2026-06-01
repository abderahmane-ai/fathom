"""VEGA — Vertical EMA Gated Attention.

Linear attention run **vertically across depth**.  If AttnRes (Rung 4 of the
design ladder, see METHODOLOGY.md §1.1) is "softmax-attention over depth at
O(L²) cost", VEGA is the corresponding "linear-attention over depth at O(L)
cost" — the depth-axis analog of how RWKV / GLA / Linear Transformers
approximate softmax attention in the token axis.  The state is a
multi-head linear-attention EMA `(S, z)` accumulated across sublayers; the
retrieval is the standard linear-attention form ``c = φ(Q) S / φ(Q) z``.

Depth memory cell that maintains a linear-attention EMA state across layers.
The state size is conditional on head rank: vector state when r_head ≤ _VECTOR_STATE_MAX_R_HEAD,
matrix state otherwise.

Math (per sublayer at depth position pos):
    y_norm = ParameterFreeRMSNorm(y)   -- no learnable scale
    K  = key_proj(y_norm),   V  = val_proj(y_norm),  Q  = query_proj(y_norm)
    g  = σ(write_gate(y_norm))                  -- write gate

    K_dep = RMSNorm(K) / sqrt(r_head)
    Q_dep = RMSNorm(Q + query_bias[pos]) / sqrt(r_head)
    φ(x)  = ELU(x) + 1                          -- positive feature map

    # Linear-attention retrieval from previous state
    # If r_head <= _VECTOR_STATE_MAX_R_HEAD (Vector State):
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
    # If r_head <= _VECTOR_STATE_MAX_R_HEAD (Vector State):
        S_new = α * S_prev + φ(K_dep) * (g * V)
    # Else (Matrix State):
        S_new = α[..., None] * S_prev + φ(K_dep)[..., None] * (g * V)[..., None, :]
    z_new = α * z_prev + φ(K_dep)

Design-ladder role (see METHODOLOGY.md §1.1, §4.6):
    VEGA is **Rung 2** of the design ladder — the multi-head linear-attention
    rung.  It is more expressive than RR (Rung 1) because the retrieval is
    Q-conditioned, not a fixed projection; it is cheaper than AttnRes (Rung 4)
    by replacing the softmax over the full history with a closed-form linear
    recurrence over a fixed-size state.  In token-axis language: VEGA is to
    AttnRes as RWKV is to softmax attention.

Init contract (verified by tests/test_design_ladder.py::test_vega_zero_start_at_init):
    out_proj.weight = 0 at init → c_out = 0 for any input.  Combined with
    the read-gate bias of -3 (read_gate ≈ 0.047) and the damp-bias of +3
    (damp ≈ 0.953), the cell produces h_new ≈ 0.953 · h_prev + y_l.  This
    is the *soft* zero-start — strictly not equal to the standard residual,
    but close enough that the first few hundred training steps are stable.
    The state (S, z) is actively written but never read at step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm

_VECTOR_STATE_MAX_R_HEAD: int = 64
_MATRIX_STATE_CHUNK_S: int = 256


class _ParameterFreeRMSNorm(nn.Module):
    """Parameter-free RMSNorm used only inside VEGACell."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(dtype)


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

        # Use vector state for small head ranks; matrix state for larger ranks.
        self.use_vector_state = self.r_head <= _VECTOR_STATE_MAX_R_HEAD

        # Projections into the rank-dimensional EMA space.
        # Fused Q, K, V projection to reduce GPU kernel launch overhead.
        self.qkv_proj = nn.Linear(d_model, 3 * rank, bias=False)
        # Single fused write gate and read down projection
        _read_rank = max(32, d_model // 16)
        self.read_rank = _read_rank
        self.fused_write_read_proj = nn.Linear(d_model, rank + _read_rank, bias=True)
        self.read_proj_up = nn.Linear(_read_rank, d_model, bias=True)

        # Element-wise damp gate: σ(damp_weight ⊙ y + damp_bias).
        self.damp_weight = nn.Parameter(torch.empty(d_model))
        self.damp_bias = nn.Parameter(torch.full((d_model,), damp_gate_bias))

        # Single output projection and RMSNorm.
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.norm_c = RMSNorm(rank, eps=eps)

        # Per-sublayer query depth bias only. Key depth bias is omitted; query bias alone is
        # sufficient for read selectivity.
        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        # Decay logits initialized log-linearly: fast heads cover short horizons, slow heads
        # cover long horizons.
        self.decay = nn.Parameter(torch.empty(self.num_sublayers, n_heads, self.r_head))
        n_slow_heads = n_heads - n_fast_heads
        with torch.no_grad():
            fast_logits = torch.linspace(fast_decay_range[0], fast_decay_range[1], n_fast_heads * self.r_head)
            slow_logits = torch.linspace(slow_decay_range[0], slow_decay_range[1], n_slow_heads * self.r_head)
            all_logits = torch.cat([fast_logits, slow_logits])
            self.decay.copy_(all_logits.view(1, self.n_heads, self.r_head).expand(self.num_sublayers, -1, -1))

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
        with torch.no_grad():
            nn.init.normal_(self.fused_write_read_proj.weight[:rank], 0.0, gate_init_std)
            nn.init.normal_(self.fused_write_read_proj.weight[rank:], 0.0, gate_init_std)
            self.fused_write_read_proj.bias.data[:rank].fill_(write_gate_bias)
            # read_proj_down originally had bias=False; zeroing this slice preserves equivalence
            # and prevents future maintainers from accidentally adding a non-zero bias here.
            self.fused_write_read_proj.bias.data[rank:].zero_()

            nn.init.normal_(self.read_proj_up.weight, 0.0, gate_init_std)
            self.read_proj_up.bias.data.fill_(read_gate_bias)

        nn.init.normal_(self.damp_weight, 0.0, gate_init_std)

        # Zero-init output projection so the cell starts as a standard residual.
        nn.init.zeros_(self.out_proj.weight)

    @property
    def key_proj(self) -> nn.Linear:
        """Alias for the fused QKV projection (backward-compatible name used in tests)."""
        return self.qkv_proj

    def get_initial_state(self, B: int, S: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (S_0, z_0) state tensors.

        Args:
            B: Batch size.
            S: Sequence length.
            device: Target device.

        Returns:
            ``(S_state, z_state)`` — EMA covariance and normalization state.
        """
        if self.use_vector_state:
            S0 = torch.zeros(B, S, self.n_heads, self.r_head, device=device, dtype=torch.float32)
        else:
            S0 = torch.zeros(B, S, self.n_heads, self.r_head, self.r_head, device=device, dtype=torch.float32)
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
        # Fused write gate and read down projection
        fused_gate_down = self.fused_write_read_proj(y_norm)
        write_gate_proj, read_proj_down_proj = fused_gate_down.split([self.rank, self.read_rank], dim=-1)
        g = torch.sigmoid(write_gate_proj).view(B, Seq, self.n_heads, self.r_head).float()

        # Per-sublayer query depth bias only. Key depth bias is omitted; query bias alone is
        # sufficient for read selectivity.
        Q_dep = Q + self.query_bias[pos].view(1, 1, self.n_heads, self.r_head)
        K_dep = K

        # Normalize and scale query and key vectors before the positive feature map.
        # Compute norms in float32 to prevent underflow in bfloat16.
        K_dep_f32 = K_dep.float()
        K_scale = torch.rsqrt(K_dep_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        K_dep = (K_dep_f32 * K_scale / (self.r_head**0.5)).to(K_dep.dtype)

        Q_dep_f32 = Q_dep.float()
        Q_scale = torch.rsqrt(Q_dep_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        Q_dep = (Q_dep_f32 * Q_scale / (self.r_head**0.5)).to(Q_dep.dtype)

        # Positive feature map φ(x) = ELU(x) + 1 (ensures state positivity).
        K_phi = (F.elu(K_dep) + 1.0).float()
        Q_phi = (F.elu(Q_dep) + 1.0).float()

        # Linear-attention retrieval (fp32; bf16 cast before norm can overflow).
        # Use den + eps, not clamp(min=eps): clamping lets c explode when z is tiny but S is not.
        if self.use_vector_state:
            num = Q_phi * S_prev.float()
            den = (Q_phi * z_prev.float()).sum(-1, keepdim=True) + self.eps
            c = (num / den).clamp(-1e4, 1e4)
        else:
            num = torch.matmul(Q_phi.unsqueeze(-2), S_prev.float()).squeeze(-2)
            den = (Q_phi * z_prev.float()).sum(-1, keepdim=True) + self.eps
            c = (num / den).clamp(-1e4, 1e4)

        c_flat = c.contiguous().view(B, Seq, self.rank)
        c_out = self.out_proj(self.norm_c(c_flat).to(y.dtype))

        # Single read gate and damp gate.
        read_gate = torch.sigmoid(self.read_proj_up(read_proj_down_proj))
        damp = torch.sigmoid(self.damp_weight * y_norm + self.damp_bias)

        h_new = damp * h_prev + y + read_gate * c_out

        # EMA state update.
        # decay: (1, 1, n_heads, r_head). For the matrix state, unsqueeze to
        # (1, 1, n_heads, r_head, 1) to broadcast against S_prev's extra rank dim.
        # z is always a vector regardless of state type.
        decay = torch.sigmoid(self.decay[pos]).view(1, 1, self.n_heads, self.r_head)
        if self.use_vector_state:
            S_new = decay * S_prev + K_phi * (g * V.float())
        else:
            # Dynamically decide if sequence length is safe for single-step parallel outer product
            # Max safe sequence length for unchunked outer product (~512MB budget)
            max_safe_seq = int(512 * 1024 * 1024 / (B * self.n_heads * self.r_head**2 * 4))
            if Seq <= max_safe_seq:
                outer = K_phi.unsqueeze(-1) * (g * V.float()).unsqueeze(-2)
                S_new = decay.unsqueeze(-1) * S_prev + outer
            else:
                # Chunked outer product to cap peak memory for large r_head / S.
                # Without chunking, the intermediate (B, S, H, r_head, r_head) tensor
                # consumes ~4.2 GB for B=8, S=2048, H=32, r_head=128.  Chunking over
                # the sequence dimension limits the peak to O(chunk_size * r_head^2).
                S_new = decay.unsqueeze(-1) * S_prev
                for s_start in range(0, Seq, _MATRIX_STATE_CHUNK_S):
                    s_end = min(s_start + _MATRIX_STATE_CHUNK_S, Seq)
                    K_chunk = K_phi[:, s_start:s_end]
                    V_chunk = g[:, s_start:s_end] * V[:, s_start:s_end].float()
                    outer_chunk = K_chunk.unsqueeze(-1) * V_chunk.unsqueeze(-2)
                    S_new[:, s_start:s_end] += outer_chunk
        z_new = decay * z_prev + K_phi

        return h_new, (S_new, z_new)
