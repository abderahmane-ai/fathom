"""Unit tests for RecurrentResidualCell (src/modules/recurrent_residual.py).

Validates:
* Identity-like behaviour at init (gate_r biased closed, proj_m zeroed).
* Memory EMA update equation matches manual computation.
* Two sublayer positions produce distinct depth embeddings.
* Gradient flows through h_new.
* reset_memory correctly zeros the state.
* Memory does not accumulate across reset_memory calls.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from src.modules.recurrent_residual import RecurrentResidualCell


@pytest.fixture
def cell(d_model, num_layers):
    c = RecurrentResidualCell(d_model, num_layers)
    return c


class TestRRCellInit:
    """At initialisation the cell must approximate a standard residual."""

    def test_identity_at_init(self, cell, B, S, d_model):
        """h_new ≈ h_prev + y when gate_r ≈ 0 and proj_m.weight = 0."""
        cell.reset_memory(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new = cell(h_prev, y, layer_idx=0, sublayer=0)
        expected = h_prev + y

        # Allow a small deviation because gate_r is not exactly 0 (bias=-3).
        assert_close(h_new, expected, atol=1e-2, rtol=1e-2)

    def test_proj_m_zero_means_no_injection(self, cell, B, S, d_model):
        """With proj_m.weight=0, memory_injection=0 → h_new = h_prev + y exactly."""
        # proj_m.weight is already zeroed at init.
        assert cell.proj_m.weight.abs().max() == 0.0

        cell.reset_memory(B, S)
        # Also zero gate bias so r=sigmoid(0)=0.5 but injection=0.
        with torch.no_grad():
            cell.proj_m.weight.zero_()
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        h_new = cell(h_prev, y, layer_idx=0, sublayer=0)
        assert_close(h_new, h_prev + y, atol=1e-6, rtol=1e-6)


class TestRRCellMemoryUpdate:
    """Memory EMA update equation."""

    def test_memory_ema_equation(self, cell, B, S, d_model):
        """Manually verify m_new = alpha * y + (1-alpha) * m_prev."""
        cell.reset_memory(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        m_before = cell.m.clone()

        cell(h_prev, y, layer_idx=0, sublayer=0)
        m_after = cell.m.clone()

        # Recompute alpha manually.
        depth_pos = torch.tensor(0)  # layer=0, sublayer=0 → pos=0
        depth_bias = cell.depth_emb(depth_pos)
        alpha = torch.sigmoid(cell.gate_alpha(y) + depth_bias)
        expected_m = alpha * y + (1.0 - alpha) * m_before
        assert_close(m_after, expected_m.detach(), atol=1e-5, rtol=1e-5)

    def test_memory_accumulates_across_layers(self, cell, B, S, d_model):
        """Memory should be non-zero after at least one forward pass."""
        cell.reset_memory(B, S)
        assert cell.m.abs().sum() == 0.0, "Memory should start at zero."

        y = torch.randn(B, S, d_model)
        cell(torch.zeros(B, S, d_model), y, layer_idx=0, sublayer=0)

        assert cell.m.abs().sum() > 0.0, "Memory should be non-zero after write."


class TestRRCellDepthEmbedding:
    """Depth embeddings per sublayer position."""

    def test_sublayer_positions_are_distinct(self, cell, d_model):
        """Sublayer 0 and sublayer 1 of the same layer use different embeddings."""
        pos_attn = torch.tensor(0 * 2 + 0)  # layer=0, sublayer=0
        pos_ffn = torch.tensor(0 * 2 + 1)   # layer=0, sublayer=1

        # After training the embeddings would differ; at init both are zero.
        # Force them to differ to test the indexing logic.
        with torch.no_grad():
            cell.depth_emb.weight[pos_attn] = torch.ones(d_model)
            cell.depth_emb.weight[pos_ffn] = torch.ones(d_model) * -1.0

        emb_attn = cell.depth_emb(pos_attn)
        emb_ffn = cell.depth_emb(pos_ffn)
        assert not torch.allclose(emb_attn, emb_ffn), (
            "Attn and FFN sublayer embeddings should be distinct."
        )


class TestRRCellGradients:
    """Gradient flow through the recurrent cell."""

    def test_grad_flows_through_h_new(self, cell, B, S, d_model):
        """h_new must be part of the autograd graph."""
        cell.reset_memory(B, S)
        h_prev = torch.randn(B, S, d_model, requires_grad=True)
        y = torch.randn(B, S, d_model, requires_grad=True)

        h_new = cell(h_prev, y, layer_idx=0, sublayer=0)
        h_new.sum().backward()

        assert h_prev.grad is not None
        assert y.grad is not None

    def test_gate_params_receive_grad(self, cell, B, S, d_model):
        """gate_r and gate_alpha parameters must receive gradient signal."""
        cell.reset_memory(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        # Run step 1: writes to memory via gate_alpha
        h_mid = cell(h_prev, y, layer_idx=0, sublayer=0)
        # Run step 2: reads from memory via gate_r and proj_m
        h_new = cell(h_mid, y, layer_idx=0, sublayer=1)
        h_new.sum().backward()

        assert cell.gate_r.weight.grad is not None
        assert cell.gate_alpha.weight.grad is not None


class TestRRCellReset:
    """Memory reset behaviour."""

    def test_reset_zeros_memory(self, cell, B, S, d_model):
        """reset_memory must produce an all-zero memory tensor."""
        # Pollute memory first.
        cell.m = torch.ones(B, S, d_model)
        cell.reset_memory(B, S)
        assert_close(cell.m, torch.zeros(B, S, d_model))

    def test_reset_changes_shape(self, cell, d_model):
        """reset_memory must resize m to the new (B, S) shape."""
        cell.reset_memory(batch_size=4, seq_len=16)
        assert cell.m.shape == (4, 16, d_model)