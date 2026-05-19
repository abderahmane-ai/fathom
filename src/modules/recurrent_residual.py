"""Recurrent Residual Cell for Transformer layers.

Augments standard additive residual connections with a global shared gated memory
that persists and evolves across the depth dimension. This architecture enables
long-range information persistence with O(1) memory and compute complexity.

Equations (per sublayer transition):
    h_norm  = LayerNorm(h_prev)
    r       = sigmoid(W_r · h_norm + b_r)          # Read gate
    m_inj   = W_m · RMSNorm(m)                     # Memory projection
    h_new   = LayerNorm(α · h_prev + y + r ⊙ m_inj) # DeepNorm update

    e_l     = depth_embedding[layer_idx * 2 + sublayer]
    α_gate  = sigmoid(W_α · y + b_α + e_l)         # Write gate
    m_new   = α_gate ⊙ y + (1 − α_gate) ⊙ m        # State transition

Design Rationale:
*   **Global Memory**: A single state 'm' is shared across all layers, enabling
    cross-layer information flow (Depth-as-RNN).
*   **DeepNorm Stability**: Residual branch scaling (α) and Post-LN normalization
    ensure gradient stability at extreme depths (1,000+ layers).
*   **DDP-Safe State**: Memory is managed as a transient instance attribute rather
    than a registered buffer to avoid DDP synchronisation overhead during updates.
*   **Depth Awareness**: Learnable embeddings provide distinct biases for each
    sublayer position (2N per stack), allowing position-dependent gating.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Normalization with learnable scaling."""
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
        gate_r_bias: Initial bias for the reset gate (default -3.0).
        gate_alpha_bias: Initial bias for the write gate (default -2.0).
        eps: Epsilon for normalisation stability.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        gate_r_bias: float = -3.0,
        gate_alpha_bias: float = -2.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps

        # ── DeepNorm constants ──────────────────────────────────────────
        self.alpha = (2.0 * self.num_sublayers) ** 0.25

        # ── Reset gate ──────────────────────────────────────────────────
        self.gate_r = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_r.weight)
        nn.init.constant_(self.gate_r.bias, gate_r_bias)

        # ── Memory projection ───────────────────────────────────────────
        self.norm_m = RMSNorm(d_model, eps=eps)
        self.proj_m = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj_m.weight)

        # ── Write gate ──────────────────────────────────────────────────
        self.gate_alpha = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_alpha.weight)
        nn.init.constant_(self.gate_alpha.bias, gate_alpha_bias)

        # Learnable depth embedding — one per sublayer position.
        self.depth_emb = nn.Embedding(self.num_sublayers, d_model)
        nn.init.zeros_(self.depth_emb.weight)

        # ── Learnable initial memory ────────────────────────────────────
        self.m_init = nn.Parameter(torch.zeros(d_model))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_initial_state(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the initial memory state expanded to batch size.

        Args:
            batch_size: Batch size ``B``.
            seq_len: Sequence length ``S``.
            device: Target device.

        Returns:
            Initial memory tensor of shape ``(B, S, d_model)``.
        """
        if device is None:
            device = self.m_init.device
        return self.m_init.view(1, 1, -1).expand(batch_size, seq_len, -1).contiguous()

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        layer_idx: int,
        sublayer: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute recurrent residual update.

        Args:
            h_prev: Previous hidden state ``(B, S, d_model)``.
            y: Sublayer output ``(B, S, d_model)``.
            m: Current memory state ``(B, S, d_model)``.
            layer_idx: Zero-based layer index.
            sublayer: 0 for Attn, 1 for FFN.

        Returns:
            Tuple of (updated hidden state, updated memory state).
        """
        # ── Reset gate ────────────────────────────────────────────────────
        h_norm_pre = F.layer_norm(h_prev, (self.d_model,))
        r = torch.sigmoid(self.gate_r(h_norm_pre))

        # ── Memory injection ──────────────────────────────────────────────
        m_inj = self.proj_m(self.norm_m(m))

        # ── Residual update (DeepNorm) ───────────────────────────────────
        h_combined = self.alpha * h_prev + y + r * m_inj
        h_new = F.layer_norm(h_combined, (self.d_model,))

        # ── Write gate & memory update ────────────────────────────────────
        sublayer_pos = layer_idx * 2 + sublayer
        depth_bias = self.depth_emb(
            torch.tensor(sublayer_pos, device=h_prev.device)
        )
        alpha = torch.sigmoid(self.gate_alpha(y) + depth_bias)
        self.last_alpha = alpha.detach().mean()
        
        m_new = alpha * y + (1.0 - alpha) * m

        return h_new, m_new