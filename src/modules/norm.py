"""Normalization layers shared across modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with a learnable scale.

    Unlike LayerNorm, RMSNorm omits the mean-centering step.  This is
    faster and works well with the pre-norm transformer convention.

    Args:
        d: Feature dimension to normalize.
        eps: Numerical stability epsilon.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full fp32 norm; casting only the output avoids bf16 underflow in mean/rsqrt.
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.scale.float() * x_f32 * rms).to(dtype)
