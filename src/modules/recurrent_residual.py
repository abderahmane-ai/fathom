"""Recurrent Residual Cell — gated depth-wise working memory.

Each sublayer call reads from and writes to a persistent memory tensor ``m``
that flows across all layers.  The cell weights are shared (weight-tied) across every
layer to keep weight parameter overhead at O(d) regardless of depth, while
per-sublayer depth biases scale as O(L * d).

Design-ladder role (see METHODOLOGY.md §1.1, §3.4):
    RR is **Rung 1** of the design ladder — the rank-1 recurrent rung.  It
    is the simplest cell in the ladder that can in principle learn to
    summarize depth history: a single linear memory ``m`` updated by a gated
    linear recurrence, with **no QKV projection** and therefore no
    attention-style inductive bias.  It is cheaper than VEGA (state size 2d
    vs. n_heads · r_head) and more interpretable (a single memory vector
    you can read directly), at the cost of expressivity.  In the benchmark
    suite, RR is the **first-order baseline** against which VEGA's
    linear-attention retrieval is measured — the question "is QKV
    conditioning worth the extra state?" is one of the design choices
    this project is designed to answer.

Init contract (verified by tests/test_design_ladder.py::test_rr_zero_start_at_init):
    At init, memory_gain = 0 and read_gate bias = -3 (read_gate ≈ 0.047) so
    the read term read_gate * memory_read is zero; damp_gate bias = +3
    (damp_gate ≈ 0.953) so h_new ≈ 0.953 · h_prev + y.  The memory is
    actively *written* (update_gate ≈ 0.119 is not closed) but the read-out
    is gated off, so the cell has no net effect on the hidden state at
    step 0.  This is the same *soft* zero-start as VEGA — strictly not
    equal to the standard residual, but close enough that the first
    few hundred training steps are stable.

Math (per sublayer):
    y_norm = RMSNorm(y)
    read_gate  = σ(read_proj(y_norm)  + depth_read_bias[pos])
    damp_gate  = σ(damp_proj(y_norm)  + depth_damp_bias[pos])
    forget_gate= σ(forget_proj(RMSNorm(m)) + depth_forget_bias[pos])
    update_gate= σ(update_proj(y_norm) + depth_update_bias[pos])

    memory_read = memory_gain * memory_out(RMSNorm(m))

    h_new = damp_gate * h_prev + y + read_gate * memory_read
    m_new = forget_gate * m    + update_gate * tanh(y)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MemoryNorm(nn.Module):
    """Parameter-free RMSNorm used only inside RecurrentResidualCell.

    A learnable scale would add d parameters per cell usage; the cell already
    has memory_gain for directional control, so a bare normalization suffices.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(dtype)


class RecurrentResidualCell(nn.Module):
    """Gated depth-wise memory cell shared across all transformer layers.

    All four gates use a low-rank factorization (d → rank → d) so the
    overhead per depth position is O(rank * d) rather than O(d^2).

    The cell is instantiated once in TransformerDecoder and the same object
    is passed to every TransformerLayer (weight sharing across depth).

    Args:
        d_model: Hidden dimension.
        num_layers: Number of transformer layers (used to size depth biases).
        read_gate_bias: Initial bias for the read gate (negative → gate starts closed).
        forget_gate_bias: Initial bias for the forget gate (positive → retentive).
        update_gate_bias: Initial bias for the update gate (negative → conservative write).
        damp_gate_bias: Initial bias for the damp gate (positive → h_prev mostly kept).
        eps: Epsilon for the internal memory norm.
        gate_init_std: Std for the low-rank weight initialization.
        memory_gain_init: Initial value for the memory gain vector (0 → zero-start).
    """

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
        self.num_sublayers = num_layers * 2  # attention + FFN per layer
        self.eps = eps
        rank = max(32, d_model // 8)

        # Fused gates for those sharing y_norm as input (read, damp, update)
        self.y_gates_down = nn.Linear(d_model, 3 * rank, bias=False)
        self.y_gates_up = nn.Linear(3 * rank, 3 * d_model, bias=True)

        # forget_proj remains independent as it takes m_norm
        self.forget_proj = nn.Sequential(
            nn.Linear(d_model, rank, bias=False),
            nn.Linear(rank, d_model, bias=True),
        )

        # Per-dimension gain applied to the normalized memory before injection.
        # Starts at memory_gain_init (default 0) so the cell begins as a standard residual.
        self.memory_gain = nn.Parameter(torch.full((d_model,), memory_gain_init))

        # Per-sublayer depth biases let each position specialize independently.
        self.depth_read_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_forget_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_update_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.depth_damp_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))

        # Learnable initial memory state (starts at zero).
        self.m_init = nn.Parameter(torch.zeros(d_model))

        self.memory_norm = _MemoryNorm(d_model, eps=eps)
        self.memory_out = nn.Linear(d_model, d_model, bias=False)
        self.y_norm = _MemoryNorm(d_model, eps=eps)

        # Low-rank weight init: tiny random weights so biases dominate at start.
        nn.init.normal_(self.forget_proj[0].weight, std=gate_init_std)
        nn.init.normal_(self.forget_proj[1].weight, std=gate_init_std)
        nn.init.zeros_(self.forget_proj[1].bias)

        with torch.no_grad():
            # Down weights: shape (3 * rank, d_model)
            # Initialize each block of shape (rank, d_model) independently
            for i in range(3):
                nn.init.normal_(
                    self.y_gates_down.weight[i * rank : (i + 1) * rank], std=gate_init_std
                )

            # Up weights: shape (3 * d_model, 3 * rank)
            # Set to zero, then initialize diagonal blocks of shape (d_model, rank) independently
            self.y_gates_up.weight.zero_()
            for i in range(3):
                nn.init.normal_(
                    self.y_gates_up.weight[
                        i * d_model : (i + 1) * d_model, i * rank : (i + 1) * rank
                    ],
                    std=gate_init_std,
                )

            # Biases: shape (3 * d_model)
            self.y_gates_up.bias[0 * d_model : 1 * d_model].fill_(read_gate_bias)
            self.y_gates_up.bias[1 * d_model : 2 * d_model].fill_(damp_gate_bias)
            self.y_gates_up.bias[2 * d_model : 3 * d_model].fill_(update_gate_bias)

            self.forget_proj[1].bias.fill_(forget_gate_bias)

        self.register_buffer("last_read_gate", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_update_gate", torch.tensor(0.0), persistent=False)

    def get_initial_state(
        self, batch_size: int, seq_len: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return m_0 broadcast to (batch_size, seq_len, d_model)."""
        target_device = device if device is not None else self.m_init.device
        return self.m_init.to(target_device).view(1, 1, -1).expand(batch_size, seq_len, -1).clone()

    def _sublayer_position(self, layer_idx: int, sublayer: int) -> int:
        """Map (layer_idx, sublayer) to a flat depth-bias index.

        Args:
            layer_idx: 0-based layer index.
            sublayer: 0 (attention) or 1 (FFN).

        Returns:
            Flat index into the depth-bias parameters.

        Raises:
            ValueError: If sublayer is not 0 or 1.
            IndexError: If the computed position exceeds num_sublayers.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one gated depth step.

        Args:
            h_prev: Hidden state entering this sublayer, shape (B, S, d_model).
            y: Sublayer output (attention or FFN), shape (B, S, d_model).
            m: Current memory state, shape (B, S, d_model).
            layer_idx: 0-based index of the enclosing transformer layer.
            sublayer: 0 for the attention sublayer, 1 for the FFN sublayer.

        Returns:
            ``(h_new, m_new)`` — updated hidden state and memory.
        """
        position = self._sublayer_position(layer_idx, sublayer)
        m = m.float()
        m_norm = self.memory_norm(m)
        y_norm = self.y_norm(y)

        # Fused projection for read, damp, update gates
        fused_y = self.y_gates_up(self.y_gates_down(y_norm))
        read_gate_proj, damp_gate_proj, update_gate_proj = fused_y.chunk(3, dim=-1)

        read_gate = torch.sigmoid(read_gate_proj + self.depth_read_bias[position])
        damp_gate = torch.sigmoid(damp_gate_proj + self.depth_damp_bias[position])
        forget_gate = torch.sigmoid(self.forget_proj(m_norm) + self.depth_forget_bias[position])
        update_gate = torch.sigmoid(update_gate_proj + self.depth_update_bias[position])

        memory_read = self.memory_gain * self.memory_out(m_norm)
        h_new = damp_gate * h_prev + y + read_gate * memory_read
        m_new = forget_gate.float() * m + update_gate.float() * torch.tanh(y).float()

        self.last_read_gate.copy_(read_gate.mean().detach())
        self.last_update_gate.copy_(update_gate.mean().detach())

        return h_new, m_new
