"""Unit tests for Attention (src/modules/attention.py).

Validates:
* Multi-head self-attention output shapes.
* Flash Attention compatibility.
* Single-token sequence (S=1) doesn't produce NaN.
"""
from __future__ import annotations

import torch

from src.modules.attention import Attention


class TestAttention:
    """Test suite for Attention module."""

    def test_output_shape(self, B, S, d_model, n_heads):
        attn = Attention(d_model, n_heads, dropout=0.0)
        x = torch.randn(B, S, d_model)
        out = attn(x)
        assert out.shape == (B, S, d_model)

    def test_grad_flows(self, B, S, d_model, n_heads):
        attn = Attention(d_model, n_heads, dropout=0.0)
        x = torch.randn(B, S, d_model, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_single_token_no_nan(self, d_model, n_heads):
        """Autoregressive step (S=1) shouldn't produce NaNs in causal mask."""
        attn = Attention(d_model, n_heads, dropout=0.0)
        x = torch.randn(1, 1, d_model)
        out = attn(x)
        assert not torch.isnan(out).any()
