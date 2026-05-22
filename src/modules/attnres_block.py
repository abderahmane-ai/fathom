"""Attention Residual modules.

Reference: "Attention Residuals", Kimi Team / Moonshot AI, arXiv:2603.15031.

These modules implement depth-wise aggregation over previous hidden states.
``BlockAttnRes`` is the practical benchmark target; ``FullAttnRes`` is kept as
small-model diagnostic reference because it stores every previous state.
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

        # Zero-init yields an equal-weight average.
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.norm_scale: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Tensor whose final dimension is ``d_model``.

        Returns:
            RMS-normalized tensor with a learnable scale.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate block history with a learned pseudo-query.

        Args:
            blocks: Completed block states, including the token embedding source.
            partial_block: Current in-block residual state.

        Returns:
            Weighted state of shape ``(B, S, d_model)``.

        Preconditions:
            ``blocks`` is non-empty and all tensors share shape and dtype.
        """
        if not blocks:
            raise ValueError("History list 'blocks' must not be empty.")

        values: torch.Tensor = torch.stack([*blocks, partial_block], dim=0)
        keys: torch.Tensor = self._rms_norm(values)
        logits: torch.Tensor = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys)
        weights: torch.Tensor = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)


class FullAttnRes(nn.Module):
    """Full Attention Residual over every stored depth state.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for RMSNorm stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))
        self.norm_scale: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Tensor whose final dimension is ``d_model``.

        Returns:
            RMS-normalized tensor with a learnable scale.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate all previous depth states.

        Args:
            states: Stored hidden states in depth order.

        Returns:
            Weighted state of shape ``(B, S, d_model)``.

        Preconditions:
            ``states`` is non-empty and all tensors share shape and dtype.
        """
        if not states:
            raise ValueError("State history must not be empty.")
        values = torch.stack(states, dim=0)
        keys = self._rms_norm(values)
        logits = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)
