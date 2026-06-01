"""Unit tests for src/modules/hyper_connections.py (mHC-Lite)."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from src.modules.hyper_connections import HyperConnection


@pytest.fixture
def d_model():
    return 32


@pytest.fixture
def m():
    return 2


class TestHyperConnectionInit:
    def test_invalid_num_channels_raises(self, d_model):
        with pytest.raises(ValueError, match="num_channels must be >= 1"):
            HyperConnection(d_model, num_channels=0)

    def test_pretrains_have_correct_shapes(self, d_model, m):
        hc = HyperConnection(d_model, num_channels=m)
        assert hc.W_pre.shape == (m, m)
        assert hc.W_post.shape == (m + 1, m)

    def test_static_input_adds_row_to_wpre(self, d_model, m):
        hc = HyperConnection(d_model, num_channels=m, use_static_input=True)
        assert hc.W_pre.shape == (m + 1, m)
        assert hc.W_post.shape == (m + 1, m)

    def test_w_pre_is_identity_at_init(self, d_model, m):
        hc = HyperConnection(d_model, num_channels=m)
        expected = torch.eye(m)
        assert_close(hc.W_pre.detach(), expected)

    def test_w_post_block_structure_at_init(self, d_model, m):
        """W_post should be identity on the first m rows, with row m adding to column 0."""
        hc = HyperConnection(d_model, num_channels=m)
        W = hc.W_post.detach()
        assert_close(W[:m, :], torch.eye(m))
        assert W[m, 0].item() == 1.0
        for c in range(1, m):
            assert W[m, c].item() == 0.0

    def test_static_gate_init(self, d_model, m):
        hc = HyperConnection(d_model, num_channels=m, use_static_input=True, init_static_gate=0.5)
        assert hc.W_pre[m, 0].item() == 0.5


class TestHyperConnectionPreMix:
    def test_identity_passthrough(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m)
        H = torch.randn(B, S, m, d_model)
        H_out = hc.pre_mix(H)
        assert_close(H_out, H)

    def test_with_static_input(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m, use_static_input=True)
        H = torch.randn(B, S, m, d_model)
        static = torch.randn(B, S, d_model)
        H_out = hc.pre_mix(H, static_input=static)
        assert H_out.shape == (B, S, m, d_model)
        assert_close(H_out[:, :, 0], H[:, :, 0] + 0.0 * static)

    def test_static_required_when_use_static_true(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m, use_static_input=True)
        H = torch.randn(B, S, m, d_model)
        with pytest.raises(ValueError, match="static_input is required"):
            hc.pre_mix(H)


class TestHyperConnectionPostMix:
    def test_init_post_mix_reduces_to_standard_residual(self, d_model, m, B, S):
        """At init, H_new[:, 0] = H_pre[:, 0] + y and H_new[:, 1] = H_pre[:, 1]."""
        hc = HyperConnection(d_model, num_channels=m)
        H_pre = torch.randn(B, S, m, d_model)
        y = torch.randn(B, S, d_model)
        H_new = hc.post_mix(H_pre, y)
        assert H_new.shape == (B, S, m, d_model)
        assert_close(H_new[:, :, 0], H_pre[:, :, 0] + y)
        for c in range(1, m):
            assert_close(H_new[:, :, c], H_pre[:, :, c])

    def test_post_mix_with_learned_offsets(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m)
        with torch.no_grad():
            hc.W_post[m, 0] = 0.5
            hc.W_post[m, 1] = 1.0
        H_pre = torch.randn(B, S, m, d_model)
        y = torch.randn(B, S, d_model)
        H_new = hc.post_mix(H_pre, y)
        assert_close(H_new[:, :, 0], H_pre[:, :, 0] + 0.5 * y)
        assert_close(H_new[:, :, 1], H_pre[:, :, 1] + 1.0 * y)


class TestHyperConnectionGradients:
    def test_gradients_flow_to_W_pre_and_W_post(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m)
        H = torch.randn(B, S, m, d_model, requires_grad=True)
        y = torch.randn(B, S, d_model, requires_grad=True)
        H_pre = hc.pre_mix(H)
        H_new = hc.post_mix(H_pre, y)
        H_new.sum().backward()
        assert hc.W_pre.grad is not None
        assert hc.W_post.grad is not None
        assert H.grad is not None
        assert y.grad is not None

    def test_gradients_with_static_input(self, d_model, m, B, S):
        hc = HyperConnection(d_model, num_channels=m, use_static_input=True)
        H = torch.randn(B, S, m, d_model, requires_grad=True)
        static = torch.randn(B, S, d_model, requires_grad=True)
        H_pre = hc.pre_mix(H, static_input=static)
        y = torch.randn(B, S, d_model, requires_grad=True)
        H_new = hc.post_mix(H_pre, y)
        H_new.sum().backward()
        assert static.grad is not None


class TestHyperConnectionIntegration:
    def test_off_diagonal_W_post_changes_output(self, d_model, m, B, S):
        """When the y-row of W_post routes y to channel 1 instead of channel 0,
        the post-mix output should reflect this change after a backward pass."""
        hc = HyperConnection(d_model, num_channels=m)
        H_pre = torch.randn(B, S, m, d_model)
        y = torch.randn(B, S, d_model)
        baseline = hc.post_mix(H_pre, y).clone()
        with torch.no_grad():
            hc.W_post[m, 0] = 0.0
            hc.W_post[m, 1] = 1.0
        rerouted = hc.post_mix(H_pre, y)
        assert not torch.allclose(rerouted, baseline)
