"""Recurrent Residual Cell for transformer layers.

Augments standard additive residual connections with a per-layer gated memory
that persists and is selectively updated across depth.

Equations (per sublayer call)
------------------------------
    h_norm  = LayerNorm(h_prev)
    r       = sigmoid(W_r · h_norm + b_r)          # reset gate
    m_inj   = W_m · RMSNorm(m)                     # memory injection
    h_new   = h_prev + y + r ⊙ m_inj               # residual update

    e_l     = depth_embedding[layer_idx * 2 + sublayer]
    alpha   = sigmoid(W_α · y + b_α + e_l)         # write gate
    m_new   = alpha ⊙ y + (1 − alpha) ⊙ m          # EMA memory update

Design notes
------------
* Gates (``W_r``, ``W_α``) are full ``d_model → d_model`` linear layers,
  not diagonal element-wise scalings.  This allows cross-dimension gating.
* Depth bias uses a learnable ``nn.Embedding`` so each sublayer position gets
  its own trainable offset, providing genuine depth-awareness.
* ``m`` is stored as a plain instance attribute (not a registered buffer) and
  is reset by the caller before each forward pass via ``reset_memory()``.
  This avoids the DDP synchronisation pitfalls of re-assigning registered
  buffers inside ``forward``.
* The ``sublayer`` argument disambiguates Attn (0) and FFN (1) sublayers so
  depth embeddings are distinct for each within the same transformer layer.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentResidualCell(nn.Module):
    """Gated recurrent residual connection across transformer layers.

    Args:
        d_model: Hidden dimension ``d``.
        num_layers: Total transformer layers.  Depth embeddings are allocated
                    for ``num_layers * 2`` sublayer positions (Attn + FFN).
        eps: Epsilon for RMSNorm numerical stability.
    """

    def __init__(self, d_model: int, num_layers: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sublayers = num_layers * 2  # Attn + FFN per layer
        self.eps = eps

        # ── Reset gate: W_r · LN(h_prev) + b_r ──────────────────────────
        # Initialise to strongly negative bias so gate ≈ 0 at init,
        # making h_new ≈ h_prev + y (identity-like start).
        self.gate_r = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_r.weight)
        nn.init.constant_(self.gate_r.bias, -3.0)

        # ── Memory projection: W_m · RMSNorm(m) ──────────────────────────
        # Zero-init weight → zero injection at init (no noise from memory).
        self.proj_m = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj_m.weight)

        # ── Write gate: W_α · y + b_α + depth_emb ────────────────────────
        # Negative bias → low write rate at init; memory fills gradually.
        self.gate_alpha = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_alpha.weight)
        nn.init.constant_(self.gate_alpha.bias, -2.0)

        # Learnable depth embedding — one per sublayer position.
        self.depth_emb = nn.Embedding(self.num_sublayers, d_model)
        nn.init.zeros_(self.depth_emb.weight)

        # Memory: not a registered buffer; managed externally via reset_memory.
        # Shape: (B, S, d_model) — set by reset_memory before each forward pass.
        self.m: torch.Tensor = torch.zeros(1, 1, d_model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation (no learned scale — used for memory only).

        Args:
            x: Input ``(B, S, d_model)``.

        Returns:
            RMS-normalised tensor of the same shape.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_memory(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset the memory state to zeros.

        Must be called by ``TransformerDecoder`` at the start of every forward
        pass to ensure clean state for each new batch.

        Args:
            batch_size: Batch size ``B``.
            seq_len: Sequence length ``S``.
            device: Target device.  Defaults to the parameter device.
        """
        if device is None:
            device = next(self.parameters()).device
        self.m = torch.zeros(batch_size, seq_len, self.d_model, device=device)

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        layer_idx: int,
        sublayer: int = 0,
    ) -> torch.Tensor:
        """Compute recurrent residual update.

        Args:
            h_prev: Previous hidden state ``(B, S, d_model)``.
            y: Sublayer output ``(B, S, d_model)``.
            layer_idx: Zero-based transformer layer index.
            sublayer: 0 for the Attn sublayer, 1 for the FFN sublayer.
                      Used to index depth embeddings.

        Returns:
            Updated hidden state ``h_new`` of shape ``(B, S, d_model)``.

        Preconditions:
            * ``reset_memory`` must have been called for the current batch
              before the first ``forward`` call.
            * ``0 <= sublayer <= 1``.
        """
        assert h_prev.shape[-1] == self.d_model, (
            f"Expected d_model={self.d_model}, got {h_prev.shape[-1]}"
        )

        m = self.m

        # ── Reset gate ────────────────────────────────────────────────────
        h_norm = F.layer_norm(h_prev, (self.d_model,))
        r = torch.sigmoid(self.gate_r(h_norm))            # (B, S, d)

        # ── Memory injection ──────────────────────────────────────────────
        m_inj = self.proj_m(self._rms_norm(m))            # (B, S, d)

        # ── Residual update ───────────────────────────────────────────────
        h_new = h_prev + y + r * m_inj                    # (B, S, d)

        # ── Write gate & memory update ────────────────────────────────────
        sublayer_pos = layer_idx * 2 + sublayer
        depth_bias = self.depth_emb(
            torch.tensor(sublayer_pos, device=h_prev.device)
        )                                                  # (d,)
        alpha = torch.sigmoid(
            self.gate_alpha(y) + depth_bias               # (B, S, d)
        )
        # EMA update; gradients flow backwards across layers
        m_new = alpha * y + (1.0 - alpha) * m
        self.m = m_new

        return h_new