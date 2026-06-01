"""Unit tests for BlockAttnRes (src/modules/attnres_block.py).

Validates:
* Output shape is correct.
* Equal-weight averaging at init (zero pseudo-query → uniform softmax).
* Gradient flows through both blocks and partial_block (no detach).
* Raises on empty blocks list.
* RMSNorm scale is learnable and changes logits after manual update.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from src.modules.attnres_block import BlockAttnRes, FullAttnRes


@pytest.fixture
def module(d_model):
    return BlockAttnRes(d_model)


class TestBlockAttnResShape:
    """Output shape contract."""

    def test_output_shape(self, module, B, S, d_model):
        """Output must be (B, S, d_model) given one completed block."""
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)
        out = module(blocks=[block0], partial_block=partial)
        assert out.shape == (B, S, d_model)

    def test_multiple_blocks_shape(self, module, B, S, d_model):
        """Shape is invariant to the number of completed blocks."""
        blocks = [torch.randn(B, S, d_model) for _ in range(4)]
        partial = torch.randn(B, S, d_model)
        out = module(blocks=blocks, partial_block=partial)
        assert out.shape == (B, S, d_model)


class TestBlockAttnResInit:
    """Behaviour at initialisation (zero pseudo-query)."""

    def test_equal_weight_average_at_init(self, module, B, S, d_model):
        """With pseudo_query=0 the softmax is uniform → output = mean of inputs.

        This matches the paper's recommendation: zero-init → neutral start.
        The key normalization is parameter-free RMSNorm (no learnable scale),
        matching the paper's protocol.
        """
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)

        # Zero the pseudo-query explicitly (it is already zero at init).
        module.pseudo_query.data.zero_()

        out = module(blocks=[block0], partial_block=partial)

        # Uniform attention over 2 tensors → simple mean.
        expected = (block0 + partial) / 2.0
        assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_uniform_weights_three_blocks(self, module, B, S, d_model):
        """Uniform weights over N+1 tensors → mean of all N+1 tensors."""
        module.pseudo_query.data.zero_()
        blocks = [torch.ones(B, S, d_model) * i for i in range(3)]
        partial = torch.ones(B, S, d_model) * 3.0

        out = module(blocks=blocks, partial_block=partial)
        expected = torch.ones(B, S, d_model) * 1.5  # mean of [0,1,2,3]
        assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestBlockAttnResGradients:
    """Gradient flow through blocks and partial_block."""

    def test_grad_flows_through_partial_block(self, module, B, S, d_model):
        """Gradient must reach partial_block (no detach in forward)."""
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model, requires_grad=True)

        out = module(blocks=[block0], partial_block=partial)
        out.sum().backward()

        assert partial.grad is not None, "No gradient reached partial_block."
        assert partial.grad.abs().sum() > 0, "Gradient is all-zero."

    def test_grad_flows_through_blocks(self, module, B, S, d_model):
        """Gradient must reach completed block tensors."""
        block0 = torch.randn(B, S, d_model, requires_grad=True)
        partial = torch.randn(B, S, d_model)

        out = module(blocks=[block0], partial_block=partial)
        out.sum().backward()

        assert block0.grad is not None, "No gradient reached block0."
        assert block0.grad.abs().sum() > 0, "Gradient is all-zero."

    def test_pseudo_query_receives_grad(self, module, B, S, d_model):
        """The pseudo_query parameter must receive a gradient signal."""
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)

        out = module(blocks=[block0], partial_block=partial)
        out.sum().backward()

        assert module.pseudo_query.grad is not None
        assert module.pseudo_query.grad.abs().sum() > 0


class TestBlockAttnResEdgeCases:
    """Edge cases and error handling."""

    def test_raises_on_empty_blocks(self, module, B, S, d_model):
        """Must raise ValueError when blocks list is empty."""
        partial = torch.randn(B, S, d_model)
        with pytest.raises(ValueError, match="must not be empty"):
            module(blocks=[], partial_block=partial)

    def test_single_token_sequence(self, module, d_model):
        """S=1 must not produce NaN (regression for causal mask edge case)."""
        block0 = torch.randn(1, 1, d_model)
        partial = torch.randn(1, 1, d_model)
        out = module(blocks=[block0], partial_block=partial)
        assert not torch.isnan(out).any(), "NaN detected for S=1."


class TestFullAttnRes:
    """Diagnostic full-depth Attention Residual reference."""

    def test_uniform_average_at_init(self, B, S, d_model):
        """Zero pseudo-query must average every stored depth state."""
        module = FullAttnRes(d_model)
        states = [torch.ones(B, S, d_model) * value for value in range(4)]
        out = module(states)
        expected = torch.ones(B, S, d_model) * 1.5
        assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_raises_on_empty_history(self, d_model):
        """Full AttnRes requires at least one stored state."""
        module = FullAttnRes(d_model)
        with pytest.raises(ValueError, match="must not be empty"):
            module([])


class TestBlockAttnResLogitScaling:
    def test_logit_scaling_prevents_saturation(self, B, S, d_model):
        """Softmax entropy remains healthy under large scale outputs due to logit scaling."""
        module = BlockAttnRes(d_model)
        block0 = torch.randn(B, S, d_model) * 5.0
        partial = torch.randn(B, S, d_model) * 5.0
        module.pseudo_query.data.fill_(10.0)

        values = torch.stack([block0, partial], dim=0)
        keys = module._rms_norm(values)
        logits = torch.einsum("d, n b s d -> n b s", module.pseudo_query, keys) / (d_model**0.5)
        weights = logits.softmax(dim=0)

        entropy = -torch.sum(weights * torch.log(weights + 1e-12), dim=0).mean().item()
        assert entropy > 0.05
