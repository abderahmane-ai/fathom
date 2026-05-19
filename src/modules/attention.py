"""Multi-head self-attention with Flash Attention and causal masking.

Uses ``torch.nn.functional.scaled_dot_product_attention`` (PyTorch ≥ 2.0),
which dispatches to Flash Attention 2 on CUDA and a memory-efficient kernel
on CPU — eliminating the O(S²) materialized attention matrix.

Causal masking is handled natively by ``is_causal=True``, which avoids the
NaN hazard of filling with ``float("-inf")`` under mixed precision.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head causal self-attention.

    Args:
        d_model: Hidden dimension.  Must be divisible by ``n_heads``.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability (0.0 in eval mode).

    Raises:
        AssertionError: If ``d_model % n_heads != 0``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection — single matmul is faster than three.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout  # passed to F.scaled_dot_product_attention

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor ``(B, S, d_model)``.
            mask: Optional additive bias tensor broadcastable to
                  ``(B, n_heads, S, S)``.  Added to attention logits before
                  softmax (e.g. for padding masks).  Should contain ``0`` for
                  valid positions and ``-inf`` for masked ones.

        Returns:
            Output tensor ``(B, S, d_model)``.

        Notes:
            ``F.scaled_dot_product_attention`` with ``is_causal=True`` handles
            the upper-triangular mask internally — no manual ``triu`` needed.
            This avoids ``-inf`` → NaN issues under mixed precision and enables
            Flash Attention kernels on supported hardware.
        """
        B, S, _ = x.shape

        # Fused QKV projection and reshape to (B, n_heads, S, head_dim).
        qkv = (
            self.qkv(x)
            .view(B, S, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)  # each (B, n_heads, S, head_dim)

        # Flash / memory-efficient scaled dot-product attention.
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=True,
        )  # (B, n_heads, S, head_dim)

        # Merge heads and project back to d_model.
        out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.proj(out)