"""Position-wise Feed-Forward Network (FFN).

Implements a standard two-layer projection with GELU activation and dropout.
Following GPT-2 conventions, dropout is applied post-activation but
pre-projection to maintain residual branch variance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Transformer FFN sublayer.

    Args:
        d_model: Input/Output dimension.
        ff_dim: Hidden expansion dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the non-linear expansion and subsequent projection."""
        return self.w2(self.dropout(F.gelu(self.w1(x))))
