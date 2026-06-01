"""Unit tests for FeedForward (src/modules/ffn.py) and RMSNorm (src/modules/norm.py)."""

from __future__ import annotations

import torch
from torch.testing import assert_close

from src.modules.ffn import FeedForward
from src.modules.norm import RMSNorm


class TestRMSNorm:
    """Test suite for RMSNorm module."""

    def test_output_shape(self, B, S, d_model):
        norm = RMSNorm(d_model)
        x = torch.randn(B, S, d_model)
        out = norm(x)
        assert out.shape == (B, S, d_model)

    def test_normalization_property(self, d_model):
        """RMSNorm output should have mean-square value close to 1 (scaled by scale parameter)."""
        eps = 1e-5
        norm = RMSNorm(d_model, eps=eps)
        # Fix scale parameter to all 2.0 to verify scaling behavior
        with torch.no_grad():
            norm.scale.fill_(2.0)

        x = torch.randn(10, d_model) * 5.0  # arbitrary scale
        out = norm(x)

        # Mean square on the last dimension should be 4.0 (scale^2 = 2.0^2 = 4.0)
        ms = out.pow(2).mean(dim=-1)
        expected = torch.full_like(ms, 4.0)
        assert_close(ms, expected, atol=1e-3, rtol=1e-3)

    def test_grad_flows(self, B, S, d_model):
        norm = RMSNorm(d_model)
        x = torch.randn(B, S, d_model, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None
        assert norm.scale.grad is not None


class TestFeedForward:
    """Test suite for SwiGLU FeedForward module."""

    def test_output_shape(self, B, S, d_model):
        ff_dim = 128
        ffn = FeedForward(d_model, ff_dim, dropout=0.0)
        x = torch.randn(B, S, d_model)
        out = ffn(x)
        assert out.shape == (B, S, d_model)

    def test_gating_behavior(self, d_model):
        """If the gate projection w1 yields zero, the output must be zero."""
        ff_dim = 64
        ffn = FeedForward(d_model, ff_dim=ff_dim, dropout=0.0)

        # Zero out the gate weight portion of the fused w1_3 linear layer
        with torch.no_grad():
            ffn.w1_3.weight[:ff_dim].zero_()

        x = torch.randn(5, d_model)
        out = ffn(x)

        # SwiGLU outputs must be zero
        assert_close(out, torch.zeros_like(out))

    def test_grad_flows(self, B, S, d_model):
        ff_dim = 128
        ffn = FeedForward(d_model, ff_dim, dropout=0.0)
        x = torch.randn(B, S, d_model, requires_grad=True)
        ffn(x).sum().backward()

        assert x.grad is not None
        assert ffn.w1_3.weight.grad is not None
        assert ffn.w2.weight.grad is not None
