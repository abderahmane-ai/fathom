"""Recurrent Residual Cell for transformer layers.

Augments standard additive residual connections with a global shared gated memory
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


class RMSNorm(nn.Module):
    """RMS normalisation with a learnable scale parameter."""
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


class RecurrentResidualCell(nn.Module):
    """Gated recurrent residual connection with DeepNorm stability upgrades.

    Args:
        d_model: Hidden dimension ``d``.
        num_layers: Total transformer layers.  Used for DeepNorm alpha and
                    to allocate depth embeddings for ``num_layers * 2`` positions.
        eps: Epsilon for normalisation stability.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        eps: float = 1e-5,
        gate_r_bias: float = -3.0,
        gate_alpha_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps

        # ── DeepNorm constants ──────────────────────────────────────────
        # Alpha scales the residual branch (h_prev).
        # Calculated as (2 * N_sublayers)**0.25
        self.alpha = (2.0 * self.num_sublayers) ** 0.25

        # ── Reset gate: W_r · LN(h_prev) + b_r ──────────────────────────
        self.gate_r = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_r.weight)
        nn.init.constant_(self.gate_r.bias, gate_r_bias)

        # ── Memory projection: W_m · RMSNorm(m) ──────────────────────────
        self.norm_m = RMSNorm(d_model, eps=eps)
        self.proj_m = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj_m.weight)

        # ── Write gate: W_α · y + b_α + depth_emb ────────────────────────
        self.gate_alpha = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_alpha.weight)
        nn.init.constant_(self.gate_alpha.bias, gate_alpha_bias)

        # Learnable depth embedding — one per sublayer position.
        self.depth_emb = nn.Embedding(self.num_sublayers, d_model)
        nn.init.zeros_(self.depth_emb.weight)


        # ── Learnable initial memory ────────────────────────────────────
        # Starts at zero but learns a global prior.
        self.m_init = nn.Parameter(torch.zeros(d_model))

        # Memory: not a registered buffer; managed externally via reset_memory.
        self.m: torch.Tensor = torch.zeros(1, 1, d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_memory(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset the memory state to the learnable initial value.

        Must be called by ``TransformerDecoder`` at the start of every forward
        pass to ensure clean state for each new batch.
        """
        if device is None:
            device = self.m_init.device
        
        # Expand m_init to (B, S, d)
        self.m = self.m_init.view(1, 1, -1).expand(batch_size, seq_len, -1).contiguous()

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        layer_idx: int,
        sublayer: int = 0,
    ) -> torch.Tensor:
        """Compute recurrent residual update using DeepNorm scaling.

        Equations:
            h_norm = LayerNorm(h_prev)
            r      = sigmoid(W_r · h_norm + b_r)
            m_inj  = W_m · RMSNorm(m)
            h_new  = LayerNorm(alpha · h_prev + y + r ⊙ m_inj)  [DeepNorm]

            alpha  = sigmoid(W_α · y + b_α + depth_emb)
            m_new  = alpha ⊙ y + (1 − alpha) ⊙ m
        """
        m = self.m

        # ── Reset gate ────────────────────────────────────────────────────
        h_norm_pre = F.layer_norm(h_prev, (self.d_model,))
        r = torch.sigmoid(self.gate_r(h_norm_pre))

        # ── Memory injection ──────────────────────────────────────────────
        m_inj = self.proj_m(self.norm_m(m))

        # ── Residual update (DeepNorm) ───────────────────────────────────
        # Scale h_prev by alpha, add sublayer output and memory injection.
        # Then apply final LayerNorm as per DeepNorm / Post-LN stability.
        h_combined = self.alpha * h_prev + y + r * m_inj
        h_new = F.layer_norm(h_combined, (self.d_model,))

        # ── Write gate & memory update ────────────────────────────────────
        sublayer_pos = layer_idx * 2 + sublayer
        depth_bias = self.depth_emb(
            torch.tensor(sublayer_pos, device=h_prev.device)
        )
        alpha = torch.sigmoid(self.gate_alpha(y) + depth_bias)
        
        m_new = alpha * y + (1.0 - alpha) * m
        self.m = m_new

        return h_new