"""Unit tests for the Sliding-Window Depth Attention with Low-Rank History (SWDA-LR) cell."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from src.modules.swda_lr import SWDALRCell


@pytest.fixture
def cell(d_model, num_layers):
    """Create a small SWDA-LR cell for tests."""
    return SWDALRCell(d_model, num_layers, window_size=4, rank=8)


class TestSWDALRCellInit:
    """Initialization behavior."""

    def test_standard_residual_at_init(self, cell, B, S, d_model):
        """Zero memory gain makes h_new exactly h_prev + y."""
        m = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

        assert_close(h_new, h_prev + y, atol=1e-6, rtol=1e-6)

    def test_get_initial_state(self, cell, B, S, d_model):
        """get_initial_state must produce empty FIFO, and zero S and z tensors."""
        fifo, S_init, z_init = cell.get_initial_state(B, S)
        assert len(fifo) == 0
        assert S_init.shape == (B, S, d_model, cell.rank)
        assert z_init.shape == (B, S, cell.rank)
        assert S_init.abs().max() == 0.0
        assert z_init.abs().max() == 0.0


class TestSWDALRCellForward:
    """State updates and calculations."""

    def test_state_recurrent_updates(self, cell, B, S, d_model):
        """Verify state progression and FIFO growth."""
        m = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model)
        y1 = torch.randn(B, S, d_model)
        y2 = torch.randn(B, S, d_model)

        # First transition
        _, m_new = cell(h_prev, y1, m, layer_idx=0, sublayer=0)
        fifo, S_new, z_new = m_new

        assert len(fifo) == 1
        assert_close(fifo[0], y1)

        # Second transition
        _, m_new2 = cell(h_prev, y2, m_new, layer_idx=0, sublayer=1)
        fifo2, S_new2, z_new2 = m_new2

        assert len(fifo2) == 2
        assert_close(fifo2[0], y1)
        assert_close(fifo2[1], y2)

        # FIFO window limit check (window_size is 4)
        m_curr = m_new2
        for _ in range(5):
            y_dummy = torch.randn(B, S, d_model)
            _, m_curr = cell(h_prev, y_dummy, m_curr, layer_idx=1, sublayer=0)

        assert len(m_curr[0]) == cell.window_size


class TestSWDALRCellGradients:
    """Gradient flow through the SWDA-LR cell."""

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
        """Read and projection parameters should receive gradients once memory path is active."""
        # Enable memory gain to make memory read path active
        with torch.no_grad():
            cell.memory_gain.fill_(1.0)

        # Mock non-zero historical states
        fifo = [torch.randn(B, S, d_model)]
        S_prev = torch.randn(B, S, d_model, cell.rank)
        z_prev = torch.randn(B, S, cell.rank).abs() + 1.0  # avoid division by zero
        m = (fifo, S_prev, z_prev)

        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, m_new = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        (h_new.sum() + m_new[1].sum() + m_new[2].sum()).backward()

        assert cell.read_weight.grad is not None
        assert cell.memory_gain.grad is not None
        assert cell.q_local_proj.weight.grad is not None
        assert cell.q_deep_proj.weight.grad is not None
        assert cell.v_proj.weight.grad is not None
        assert cell.decay_bias.grad is not None
        assert cell.key_decay_bias.grad is not None
