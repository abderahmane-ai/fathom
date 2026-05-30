"""Attention Residual modules — depth-wise aggregation over block history.

Reference: "Attention Residuals", Kimi Team / Moonshot AI, arXiv:2603.15031.

Both classes replace the standard residual addition with a softmax-weighted
sum over previous hidden states.  ``BlockAttnRes`` is the practical target;
``FullAttnRes`` stores every previous sublayer state and is kept as a
small-model diagnostic reference.

Math (BlockAttnRes):
    values  = stack([*blocks, partial_block])        # (N+1, B, S, d)
    keys    = RMSNorm(values)                        # normalized for bounded logits
    logits  = einsum("d, nbsd -> nbs", pseudo_query, keys)
    weights = softmax(logits, dim=0)                 # over depth axis
    output  = einsum("nbs, nbsd -> bsd", weights, values)

At init pseudo_query=0 → uniform weights → output equals the mean of all inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockAttnRes(nn.Module):
    """Gated aggregation over a sequence of completed architectural blocks.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for the internal RMS normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Zero-init → uniform-weight average at the start.
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.norm_scale:   nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate block history with a learned pseudo-query.

        Args:
            blocks: Completed block states including the embedding source.
                    Must be non-empty.
            partial_block: Current in-block residual state.

        Returns:
            Weighted aggregation of shape (B, S, d_model).
        """
        if not blocks:
            raise ValueError("History list 'blocks' must not be empty.")

        values: torch.Tensor = torch.stack([*blocks, partial_block], dim=0)
        keys:   torch.Tensor = self._rms_norm(values)
        logits: torch.Tensor = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys)
        weights: torch.Tensor = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)


class FullAttnRes(nn.Module):
    """Full Attention Residual over every stored sublayer state.

    Stores every hidden state since the embedding and aggregates them all.
    Memory cost is O(2L * d) which limits practical use to small models.
    Kept as a diagnostic reference for the block-based version.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for the internal RMS normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.norm_scale:   nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate all previous depth states.

        Args:
            states: Hidden states in depth order.  Must be non-empty.

        Returns:
            Weighted aggregation of shape (B, S, d_model).
        """
        if not states:
            raise ValueError("State history must not be empty.")
        values = torch.stack(states, dim=0)
        keys   = self._rms_norm(values)
        logits = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)
