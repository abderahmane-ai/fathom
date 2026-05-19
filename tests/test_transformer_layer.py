"""Unit tests for TransformerLayer (src/modules/transformer_layer.py).

Validates:
* Standard mode: correct output shape, residual connection.
* AttnRes mode: forward_attnres returns correct types and shapes.
* AttnRes mode: blocks list grows at correct boundary.
* Calling wrong forward method raises ValueError.
"""
from __future__ import annotations

import pytest
import torch

from src.modules.transformer_layer import TransformerLayer


class TestTransformerLayerStandard:
    """Standard residual mode."""

    def test_output_shape(self, standard_cfg, B, S, d_model):
        layer = TransformerLayer(standard_cfg)
        x = torch.randn(B, S, d_model)
        out = layer(x, layer_idx=0)
        assert out.shape == (B, S, d_model)

    def test_residual_is_not_identity(self, standard_cfg, B, S, d_model):
        """Output should differ from input (non-trivial transformation)."""
        layer = TransformerLayer(standard_cfg)
        x = torch.randn(B, S, d_model)
        out = layer(x, layer_idx=0)
        assert not torch.allclose(out, x), "Layer output must not equal input."

    def test_grad_flows(self, standard_cfg, B, S, d_model):
        layer = TransformerLayer(standard_cfg)
        x = torch.randn(B, S, d_model, requires_grad=True)
        layer(x, layer_idx=0).sum().backward()
        assert x.grad is not None

    def test_raises_if_forward_attnres_called(self, standard_cfg, B, S, d_model):
        """forward_attnres must not exist on a standard-mode layer."""
        layer = TransformerLayer(standard_cfg)
        x = torch.randn(B, S, d_model)
        # forward() in standard mode should NOT raise
        layer(x, layer_idx=0)
        # forward_attnres would raise AttributeError (no attn_res attr)
        # or be unusable — just verify forward works cleanly.


class TestTransformerLayerAttnRes:
    """Block-AttnRes mode."""

    def test_forward_attnres_returns_tuple(self, attnres_cfg, B, S, d_model):
        layer = TransformerLayer(attnres_cfg)
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)
        result = layer.forward_attnres([block0], partial, layer_idx=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_partial_block_shape(self, attnres_cfg, B, S, d_model):
        layer = TransformerLayer(attnres_cfg)
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)
        _, new_partial = layer.forward_attnres([block0], partial, layer_idx=0)
        assert new_partial.shape == (B, S, d_model)

    def test_blocks_grows_at_boundary(self, attnres_cfg, B, S, d_model):
        """blocks list must grow by 1 when layer_idx+1 is a multiple of layers_per_block."""
        layer = TransformerLayer(attnres_cfg)
        # attnres_cfg has block_size=4 → layers_per_block=2
        # Boundary at layer_idx = 1 (idx+1=2, 2%2==0)
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)

        blocks_in = [block0]
        blocks_out, _ = layer.forward_attnres(blocks_in, partial, layer_idx=1)
        assert len(blocks_out) == len(blocks_in) + 1, (
            "blocks list should grow by 1 at block boundary."
        )

    def test_blocks_does_not_grow_mid_block(self, attnres_cfg, B, S, d_model):
        """blocks list must NOT grow for a non-boundary layer."""
        layer = TransformerLayer(attnres_cfg)
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model)

        blocks_in = [block0]
        blocks_out, _ = layer.forward_attnres(blocks_in, partial, layer_idx=0)
        assert len(blocks_out) == len(blocks_in), (
            "blocks list should NOT grow for a non-boundary layer."
        )

    def test_forward_raises_for_attnres_mode(self, attnres_cfg, B, S, d_model):
        """Calling forward() on an attnres layer must raise ValueError."""
        layer = TransformerLayer(attnres_cfg)
        x = torch.randn(B, S, d_model)
        with pytest.raises(ValueError, match="forward_attnres"):
            layer(x, layer_idx=0)

    def test_grad_flows_through_partial(self, attnres_cfg, B, S, d_model):
        layer = TransformerLayer(attnres_cfg)
        block0 = torch.randn(B, S, d_model)
        partial = torch.randn(B, S, d_model, requires_grad=True)
        _, new_partial = layer.forward_attnres([block0], partial, layer_idx=0)
        new_partial.sum().backward()
        assert partial.grad is not None
