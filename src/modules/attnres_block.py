"""Block Attention Residual (Block-AttnRes) implementation.

Reference: "Attention Residuals", Kimi Team / Moonshot AI, arXiv:2603.15031.

This module implements content-aware cross-block aggregation, allowing layers
to attend back to previous block states via a learnable pseudo-query. This
mechanism prevents information dilution by maintaining a direct path to early
representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockAttnRes(nn.Module):
    """Gated aggregation over a sequence of architectural blocks.

    Replaces the standard residual addition with a softmax-weighted sum over
    historical block representations. Keys are RMS-normalized to ensure
    bounded logit magnitudes.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for RMSNorm stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable pseudo-query: Zero-init yields an equal-weight average.
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        
        # Learnable RMSNorm scale.
        self.norm_scale: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization with a learnable scale factor."""
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates history using attention weights derived from the pseudo-query."""
        if not blocks:
            raise ValueError("History list 'blocks' must not be empty.")

        # V: Value stack including current partial accumulation [N+1, B, S, d]
        V: torch.Tensor = torch.stack([*blocks, partial_block], dim=0)

        # K: RMS-normalized keys to stabilize attention scores.
        K: torch.Tensor = self._rms_norm(V)

        # Compute scores: Inner product between the pseudo-query and normalized keys.
        # logits shape: [N+1, B, S]
        logits: torch.Tensor = torch.einsum("d, n b s d -> n b s", self.pseudo_query, K)

        # Attention distribution over the block history.
        weights: torch.Tensor = logits.softmax(dim=0)

        # Return the weighted combination of block values.
        return torch.einsum("n b s, n b s d -> b s d", weights, V)