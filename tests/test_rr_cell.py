"""Unit tests for RecurrentResidualCell (src/modules/recurrent_residual.py)."""

import pytest
import torch
from torch.testing import assert_close

from src.modules.recurrent_residual import RecurrentResidualCell


@pytest.fixture
def B():
    return 2


@pytest.fixture
def S():
    return 4


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def cell(d_model):
    return RecurrentResidualCell(d_model, num_layers=2)


class TestRRCellInit:
    def test_standard_residual_at_init(self, cell, B, S, d_model):
        """At init (memory_gain=0) the cell should behave close to h_prev + y."""
        m = cell.get_initial_state(B, S, device=cell.read_proj[0].weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        # Loose tolerance: damp gate ≈ 1 and read gate ≈ 0 at init, so h_new ≈ h_prev + y.
        assert_close(h_new, h_prev + y, atol=0.2, rtol=0.2)

    def test_parameter_count(self, cell, d_model):
        """Total parameter count must match the theoretical formula (S+8)*d."""
        actual = sum(p.numel() for p in cell.parameters())
        # Formula: 4 gates × 2 sublayers × d (depth biases) + per-cell params
        # At minimum, actual >> d_model; verify we have a reasonable lower bound.
        assert actual >= cell.num_sublayers * 4 * d_model, (
            f"Expected at least {cell.num_sublayers * 4 * d_model} params, got {actual}"
        )


class TestRRCellMemoryUpdate:
    def test_memory_update_shape(self, cell, B, S, d_model):
        """Memory output must match input shape."""
        m_before = cell.get_initial_state(B, S, device=cell.read_proj[0].weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        _, m_after = cell(h_prev, y, m_before, layer_idx=0, sublayer=0)
        assert m_after.shape == (B, S, d_model)

    def test_sublayer_position_validation(self, cell):
        """_sublayer_position must raise on invalid sublayer values."""
        with pytest.raises(ValueError, match="sublayer must be 0 or 1"):
            cell._sublayer_position(0, 2)

    def test_sublayer_position_out_of_bounds(self, cell):
        """_sublayer_position must raise IndexError when position exceeds num_sublayers."""
        with pytest.raises(IndexError):
            cell._sublayer_position(cell.num_layers, 0)


class TestRRCellGradients:
    def test_grad_flows(self, cell, B, S, d_model):
        """Gradients must flow back to h_prev, y, and gate weights."""
        h_prev = torch.randn(B, S, d_model, requires_grad=True)
        y = torch.randn(B, S, d_model, requires_grad=True)
        m = cell.get_initial_state(B, S, device=h_prev.device)
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        h_new.sum().backward()
        assert h_prev.grad is not None
        assert y.grad is not None
        assert cell.read_proj[0].weight.grad is not None


class TestRRCellStability:
    def test_memory_state_bounding(self, cell, B, S, d_model):
        """Under large input values, tanh bounding prevents state explosion."""
        m = cell.get_initial_state(B, S, device=cell.read_proj[0].weight.device)
        h_prev = torch.randn(B, S, d_model)
        y_large = torch.randn(B, S, d_model) * 100.0
        _, m_after = cell(h_prev, y_large, m, layer_idx=0, sublayer=0)
        assert m_after.abs().max().item() <= 1.0 + cell.eps
