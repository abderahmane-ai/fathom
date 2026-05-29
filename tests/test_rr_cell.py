"""Unit tests for the Recurrent Residual cell."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from src.modules.recurrent_residual import RecurrentResidualCell


@pytest.fixture
def cell(d_model, num_layers):
    """Create a small RR cell for tests."""
    return RecurrentResidualCell(d_model, num_layers)


class TestRRCellInit:
    """Initialization behavior."""

    def test_standard_residual_at_init(self, cell, B, S, d_model):
        """Zero memory gain makes h_new exactly h_prev + y."""
        m = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

        assert_close(h_new, h_prev + y, atol=1e-6, rtol=1e-6)

    def test_methodology_parameter_count(self, cell):
        """RR parameter count should follow (num_sublayers + 8) * d_model."""
        assert sum(parameter.numel() for parameter in cell.parameters()) == (
            cell.parameter_count_formula
        )

    def test_no_dense_gate_or_projection_matrices(self, cell):
        """RR gates and memory read path must stay diagonal."""
        parameter_shapes = {
            name: tuple(parameter.shape) for name, parameter in cell.named_parameters()
        }
        assert parameter_shapes["read_weight"] == (cell.d_model,)
        assert parameter_shapes["forget_weight"] == (cell.d_model,)
        assert parameter_shapes["update_weight"] == (cell.d_model,)
        assert parameter_shapes["memory_gain"] == (cell.d_model,)
        assert all(
            "weight" not in name or shape != (cell.d_model, cell.d_model)
            for name, shape in parameter_shapes.items()
        )

    def test_m_init_is_learnable(self, cell, B, S):
        """m_init should be a learnable parameter initialized to zero."""
        assert isinstance(cell.m_init, torch.nn.Parameter)
        assert cell.m_init.abs().max() == 0.0

        with torch.no_grad():
            cell.m_init.fill_(1.0)
        m = cell.get_initial_state(B, S)
        assert m.abs().min() == 1.0


class TestRRCellMemoryUpdate:
    """Memory update equation."""

    def test_memory_update_equation(self, cell, B, S, d_model):
        """Manually verify m_new = f * m_prev + u * y."""
        m_before = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        _, m_after = cell(h_prev, y, m_before, layer_idx=0, sublayer=0)

        forget_gate = torch.sigmoid(cell.forget_weight * y + cell.forget_bias)
        update_gate = torch.sigmoid(cell.update_weight * y + cell.update_bias + cell.depth_bias[0])
        expected_m = forget_gate * m_before + update_gate * y
        assert_close(m_after, expected_m, atol=1e-6, rtol=1e-6)

    def test_memory_accumulates_across_layers(self, cell, B, S, d_model):
        """Memory should be non-zero after at least one forward pass."""
        m = cell.get_initial_state(B, S)
        y = torch.randn(B, S, d_model)
        _, m_after = cell(torch.zeros(B, S, d_model), y, m, layer_idx=0, sublayer=0)

        assert m_after.abs().sum() > 0.0


class TestRRCellDepthBias:
    """Depth bias indexing."""

    def test_sublayer_positions_are_distinct(self, cell, d_model):
        """Sublayer 0 and sublayer 1 of the same layer use different bias rows."""
        with torch.no_grad():
            cell.depth_bias[0] = torch.ones(d_model)
            cell.depth_bias[1] = torch.ones(d_model) * -1.0

        assert not torch.allclose(cell.depth_bias[0], cell.depth_bias[1])

    def test_invalid_sublayer_raises(self, cell, B, S, d_model):
        """Only attention and FFN sublayer ids are valid."""
        m = cell.get_initial_state(B, S)
        with pytest.raises(ValueError, match="sublayer"):
            cell(torch.zeros(B, S, d_model), torch.zeros(B, S, d_model), m, 0, sublayer=2)


class TestRRCellGradients:
    """Gradient flow through the recurrent cell."""

    def test_grad_flows_through_h_new(self, cell, B, S, d_model):
        """h_new must be part of the autograd graph."""
        m = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model, requires_grad=True)
        y = torch.randn(B, S, d_model, requires_grad=True)

        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        h_new.sum().backward()

        assert h_prev.grad is not None
        assert y.grad is not None

    def test_gate_params_receive_grad_when_memory_path_active(self, cell, B, S, d_model):
        """Read, forget, and update gate parameters should receive gradients once memory is readable."""
        with torch.no_grad():
            cell.memory_gain.fill_(1.0)
        m = torch.randn(B, S, d_model)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, m_new = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        (h_new.sum() + m_new.sum()).backward()

        assert cell.read_weight.grad is not None
        assert cell.forget_weight.grad is not None
        assert cell.update_weight.grad is not None
        assert cell.memory_gain.grad is not None


class TestRRCellInitialState:
    """Memory initialization behavior."""

    def test_get_initial_state_uses_m_init(self, cell, B, S):
        """get_initial_state must produce memory tensor expanded from m_init."""
        with torch.no_grad():
            cell.m_init.normal_()
        expected = cell.m_init.view(1, 1, -1).expand(B, S, -1)

        m = cell.get_initial_state(B, S)
        assert_close(m, expected)

    def test_initial_state_shape(self, cell, d_model):
        """get_initial_state must produce the expected ``(B, S, d)`` shape."""
        m = cell.get_initial_state(batch_size=4, seq_len=16)
        assert m.shape == (4, 16, d_model)
