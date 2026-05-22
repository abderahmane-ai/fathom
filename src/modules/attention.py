"""Multi-head self-attention with optimized Flash Attention kernels.

Utilizes ``torch.nn.functional.scaled_dot_product_attention`` for efficient
memory utilization (O(S) instead of O(S²)) and native causal masking.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    """Causal multi-head self-attention module.

    Args:
        d_model: Input dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection for single-pass matrix multiplication.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optimized SDPA kernel dispatch."""

        # Project and reshape into multi-head components.
        qkv = rearrange(
            self.qkv(x),
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.n_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash / memory-efficient scaled dot-product attention with causal mask.
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Concatenate heads and project back to d_model.
        out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.proj(out)
