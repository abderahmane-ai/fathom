"""SwiGLU Feed-Forward Network.

To match a standard 4× GELU FFN in parameter count, set
    ff_dim = int(8/3 * d_model)
rounded to the nearest multiple of 64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU FFN: w2(dropout(silu(w1(x)) ⊙ w3(x))).

    Uses three projections following the PaLM / LLaMA convention.
    w1 is the gate branch, w3 is the up-projection, w2 is the down-projection.

    Args:
        d_model: Input and output dimension.
        ff_dim: Intermediate (up-projected) dimension.
        dropout: Dropout probability applied to the gated activation.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1_3 = nn.Linear(d_model, 2 * ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize each slice independently to match separate layer initialization exactly
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.w1_3.weight[:ff_dim], a=5**0.5)
            nn.init.kaiming_uniform_(self.w1_3.weight[ff_dim:], a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w1_3(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(gate) * up))
