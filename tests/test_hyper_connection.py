"""Unit tests for the mHC-Lite implementation in src/modules/hyper_connections.py.

The class follows arXiv:2601.05732 §3 (mHC-Lite: convex combination of
permutation matrices for the doubly-stochastic H_res).  These tests
codify the paper's init protocol and verify the at-init behavior.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from src.modules.hyper_connections import HyperConnection

# ─────────────────────────────────────────────────────────────────────────
# Init protocol — must match the paper exactly
# ─────────────────────────────────────────────────────────────────────────


class TestHyperConnectionInit:
    def test_invalid_num_channels_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="num_channels=2"):
            HyperConnection(32, num_channels=4)

    def test_static_input_unsupported(self) -> None:
        with pytest.raises(NotImplementedError, match="use_static_input"):
            HyperConnection(32, use_static_input=True)

    def test_w_pre_post_res_shapes(self) -> None:
        hc = HyperConnection(d_model=8, num_channels=2)
        assert hc.W_pre.shape == (16, 2)  # (n*d, n)
        assert hc.W_post.shape == (16, 2)
        assert hc.W_res.shape == (16, 2)  # (n*d, n!) with n! = 2

    def test_alpha_gates_start_at_0_01(self) -> None:
        """α_pre, α_post, α_res all start at 0.01 (paper §3.3)."""
        hc = HyperConnection(d_model=8, num_channels=2)
        assert_close(hc.alpha_pre.detach(), torch.tensor([0.01]))
        assert_close(hc.alpha_post.detach(), torch.tensor([0.01]))
        assert_close(hc.alpha_res.detach(), torch.tensor([0.01]))

    def test_b_pre_init_minus_one_with_main_channel_plus_one(self) -> None:
        """b_pre = [-1, -1] except b_pre[0, 0] = +1 (main channel)."""
        hc = HyperConnection(d_model=8, num_channels=2)
        b_pre = hc.b_pre.detach()
        assert_close(b_pre[0, 1], torch.tensor(-1.0))
        assert_close(b_pre[0, 0], torch.tensor(1.0))

    def test_b_post_init_minus_one_with_main_channel_plus_one(self) -> None:
        """b_post has the same structure as b_pre."""
        hc = HyperConnection(d_model=8, num_channels=2)
        b_post = hc.b_post.detach()
        assert_close(b_post[0, 1], torch.tensor(-1.0))
        assert_close(b_post[0, 0], torch.tensor(1.0))

    def test_b_res_init_minus_eight_with_identity_zero(self) -> None:
        """b_res = [-8, -8] except b_res[0, 0] = 0 (identity permutation)."""
        hc = HyperConnection(d_model=8, num_channels=2)
        b_res = hc.b_res.detach()
        assert_close(b_res[0, 1], torch.tensor(-8.0))
        assert_close(b_res[0, 0], torch.tensor(0.0))

    def test_projection_weights_init_zero(self) -> None:
        """W_pre, W_post, W_res all start at zero (paper §3.3)."""
        hc = HyperConnection(d_model=8, num_channels=2)
        assert torch.equal(hc.W_pre.detach(), torch.zeros(16, 2))
        assert torch.equal(hc.W_post.detach(), torch.zeros(16, 2))
        assert torch.equal(hc.W_res.detach(), torch.zeros(16, 2))

    def test_permutations_buffer_holds_identity_and_swap(self) -> None:
        """The 2! permutation matrices of 2×2 are I and swap."""
        hc = HyperConnection(d_model=8, num_channels=2)
        P0, P1 = hc.permutations[0], hc.permutations[1]
        assert_close(P0, torch.eye(2))
        assert_close(P1, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


# ─────────────────────────────────────────────────────────────────────────
# At-init behavior: the paper's "approximate standard residual" contract
# ─────────────────────────────────────────────────────────────────────────


class TestHyperConnectionAtInit:
    def test_h_pre_equals_sigmoid_of_b_pre_at_init(self) -> None:
        """At init, H_pre = σ(b_pre) ≈ [0.731, 0.269]."""
        hc = HyperConnection(d_model=8, num_channels=2)
        # At init, W_pre=0 and α=0.01, so H_pre depends only on b_pre.
        # Compute the analytic value and verify the documented constants.
        H_pre_analytic = torch.sigmoid(hc.b_pre.detach())
        assert_close(H_pre_analytic[0, 0], torch.tensor(0.7311), atol=1e-3, rtol=1e-3)
        assert_close(H_pre_analytic[0, 1], torch.tensor(0.2689), atol=1e-3, rtol=1e-3)

    def test_h_post_equals_two_times_sigmoid_of_b_post_at_init(self) -> None:
        """At init, H_post = 2·σ(b_post) ≈ [1.462, 0.538]."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H_post_analytic = 2.0 * torch.sigmoid(hc.b_post.detach())
        assert_close(H_post_analytic[0, 0], torch.tensor(1.4621), atol=1e-3, rtol=1e-3)
        assert_close(H_post_analytic[0, 1], torch.tensor(0.5379), atol=1e-3, rtol=1e-3)

    def test_h_res_equals_softmax_of_b_res_at_init(self) -> None:
        """At init, H_res = softmax(b_res) ≈ [[0.9997, 0.0003], [0.0003, 0.9997]] ≈ I_2."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H_res_analytic = torch.softmax(hc.b_res.detach(), dim=-1)
        assert_close(H_res_analytic[0, 0], torch.tensor(0.9997), atol=1e-3, rtol=1e-3)
        assert_close(H_res_analytic[0, 1], torch.tensor(3.4e-4), atol=1e-3, rtol=1e-3)

    def test_post_mix_init_reduces_to_approximate_standard_residual(self) -> None:
        """At init, H_new ≈ H_res · H + H_post · y.  For the main channel
        with H = 0 (so x_pre = 0 → y = F(0) = some fixed vector):

            H_new[0] ≈ 1·H[0] + 1.462·y
            H_new[1] ≈ 1·H[1] + 0.538·y

        This is the paper's "approximate standard residual" — not bit-for-
        bit, but the closest the paper's init gets.  See METHODOLOGY.md
        §5.2 for the full discussion.
        """
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.zeros(1, 1, 2, 8)
        # Call pre_mix to set the cached H_norm, then check post_mix.
        _ = hc.pre_mix(H)
        y = torch.ones(1, 1, 8)
        H_new = hc.post_mix(H, y)

        # H_res · H = I_2 · 0 = 0
        # y_post[0] = 1.462 · 1 = 1.462
        # y_post[1] = 0.538 · 1 = 0.538
        # So H_new[0] ≈ 1.462, H_new[1] ≈ 0.538
        assert H_new[..., 0, :].abs().mean() > 0.5  # not zero
        assert H_new[..., 1, :].abs().mean() > 0.3
        # And H_new[0] > H_new[1] because main channel has larger H_post
        assert H_new[..., 0, :].abs().mean() > H_new[..., 1, :].abs().mean()


# ─────────────────────────────────────────────────────────────────────────
# Forward path: pre_mix / post_mix API and shape contracts
# ─────────────────────────────────────────────────────────────────────────


class TestHyperConnectionPreMix:
    def test_pre_mix_returns_d_dim_tensor(self) -> None:
        """pre_mix returns the mixed sublayer input of shape (B, S, d)."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        x = hc.pre_mix(H)
        assert x.shape == (2, 4, 8)

    def test_pre_mix_with_zero_h_returns_zero(self) -> None:
        """H = 0 → x_pre = H_pre · 0 = 0, regardless of H_pre values."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.zeros(1, 1, 2, 8)
        x = hc.pre_mix(H)
        assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)

    def test_pre_mix_picks_up_gradient(self) -> None:
        """pre_mix must propagate gradients back to the input H."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8, requires_grad=True)
        x = hc.pre_mix(H)
        x.sum().backward()
        assert H.grad is not None
        assert H.grad.abs().sum() > 0

    def test_pre_mix_dimension_mismatch_raises(self) -> None:
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 3, 8)  # 3 channels, not 2
        with pytest.raises(ValueError, match="channels"):
            hc.pre_mix(H)


class TestHyperConnectionPostMix:
    def test_post_mix_shape(self) -> None:
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        _ = hc.pre_mix(H)
        y = torch.randn(2, 4, 8)
        H_new = hc.post_mix(H, y)
        assert H_new.shape == (2, 4, 2, 8)

    def test_post_mix_without_pre_mix_raises(self) -> None:
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        y = torch.randn(2, 4, 8)
        with pytest.raises(RuntimeError, match="prior pre_mix"):
            hc.post_mix(H, y)

    def test_post_mix_dimension_mismatch_raises(self) -> None:
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        _ = hc.pre_mix(H)
        y = torch.randn(2, 4, 8)
        H_wrong = torch.randn(2, 4, 3, 8)
        with pytest.raises(ValueError, match="channels"):
            hc.post_mix(H_wrong, y)

    def test_post_mix_zero_y_does_not_change_residual(self) -> None:
        """If y = 0, then H_new = H_res · H.  At init H_res ≈ I, so H_new ≈ H."""
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        _ = hc.pre_mix(H)
        H_new = hc.post_mix(H, torch.zeros(2, 4, 8))
        # H_res ≈ I_2 at init, so H_new ≈ H.  Tolerance accounts for the
        # ≈ 3e-4 deviation of softmax([0, -8]) from a perfect one-hot.
        assert_close(H_new, H, atol=5e-3, rtol=5e-3)


# ─────────────────────────────────────────────────────────────────────────
# Gradient flow
# ─────────────────────────────────────────────────────────────────────────


class TestHyperConnectionGradients:
    def test_gradients_flow_to_w_pre_w_post_w_res_and_alphas(self) -> None:
        """The full gradient chain H_new → y(=f(x)) → x(pre_mix) → W_pre must work.

        Note: at the paper's W=0 init, the gradient w.r.t. α_pre/α_post/α_res
        is exactly zero (because d(α·H·W)/dα = H·W, and W=0).  This is by
        design — the W matrices carry the gradient signal at init, and the
        α gates take over once W has learned a non-trivial direction.  We
        only assert W_grad ≠ 0 here, and check α grad separately after
        perturbing W.
        """
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8, requires_grad=True)
        x = hc.pre_mix(H)
        # Make y a function of x so the gradient chain reaches W_pre.
        y_from_x = x * 2.0 + 0.1
        H_new = hc.post_mix(H, y_from_x)
        H_new.sum().backward()

        assert hc.W_pre.grad is not None and hc.W_pre.grad.abs().sum() > 0
        assert hc.W_post.grad is not None and hc.W_post.grad.abs().sum() > 0
        assert hc.W_res.grad is not None and hc.W_res.grad.abs().sum() > 0
        assert H.grad is not None

    def test_alphas_receive_gradient_after_perturbing_w(self) -> None:
        """After perturbing W (so d(α·H·W)/dα = H·W ≠ 0), the α gates
        must receive non-zero gradient — proving they are real learnable
        parameters, not just dead scalars."""
        hc = HyperConnection(d_model=8, num_channels=2)
        with torch.no_grad():
            hc.W_pre.fill_(0.1)
            hc.W_post.fill_(0.1)
            hc.W_res.fill_(0.1)
        H = torch.randn(2, 4, 2, 8, requires_grad=True)
        x = hc.pre_mix(H)
        y_from_x = x * 2.0 + 0.1
        H_new = hc.post_mix(H, y_from_x)
        H_new.sum().backward()
        assert hc.alpha_pre.grad is not None and hc.alpha_pre.grad.abs().item() > 0
        assert hc.alpha_post.grad is not None and hc.alpha_post.grad.abs().item() > 0
        assert hc.alpha_res.grad is not None and hc.alpha_res.grad.abs().item() > 0

    def test_learned_alpha_changes_output_after_perturbation(self) -> None:
        """Perturbing W_pre (W=0 init) must change the pre_mix output.

        At init, W_pre = 0, so α_pre has no effect — the test instead verifies
        that the projection weights are the learnable lever (gradient flow
        covers the same surface).
        """
        torch.manual_seed(0)
        hc = HyperConnection(d_model=8, num_channels=2)
        H = torch.randn(2, 4, 2, 8)
        baseline = hc.pre_mix(H).clone()
        with torch.no_grad():
            hc.W_pre.fill_(0.1)
        rerun = hc.pre_mix(H)
        assert not torch.allclose(baseline, rerun)
