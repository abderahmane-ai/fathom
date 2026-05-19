"""Feed-forward network (FFN) sublayer.

Implements the two-layer FFN used in each transformer layer with GELU
activation and proper dropout placement (post-activation, pre-projection).

Dropout is applied *between* the two linear layers (after the activation),
not after the final projection.  Applying it after ``w2`` would unnecessarily
zero out the residual branch contribution.

Reference: GPT-2 (Radford et al., 2019), §2.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Computes:  ``w2(dropout(gelu(w1(x))))``

    Args:
        d_model: Input / output dimension.
        ff_dim: Intermediate (expansion) dimension.  Typically ``4 * d_model``.
        dropout: Dropout probability applied after activation.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the FFN transformation.

        Args:
            x: Input tensor ``(B, S, d_model)``.

        Returns:
            Output tensor ``(B, S, d_model)``.
        """
        # Dropout is between w1 and w2, *not* after w2.
        return self.w2(self.dropout(F.gelu(self.w1(x))))