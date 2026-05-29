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

        # Force read gate to 0 so h_new exactly matches h_prev + y
        with torch.no_grad():
            cell.gate_biases[0].fill_(-100.0)

        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

        assert_close(h_new, h_prev + y, atol=1e-6, rtol=1e-6)

    def test_get_initial_state(self, cell, B, S, d_model):
        """get_initial_state must produce empty FIFO tensor, index 0, and zero S and z tensors."""
        fifo_buf, fifo_norm_buf, fifo_idx, S_init, z_init = cell.get_initial_state(B, S)
        assert fifo_buf.shape == (B, S, cell.window_size, d_model)
        assert fifo_norm_buf.shape == (B, S, cell.window_size, d_model)
        assert fifo_idx == 0
        assert S_init.shape == (B, S, cell.n_heads, cell.r_head, cell.d_head)
        assert z_init.shape == (B, S, cell.n_heads, cell.r_head)
        assert fifo_buf.abs().max() == 0.0
        assert fifo_norm_buf.abs().max() == 0.0
        assert S_init.abs().max() == 0.0
        assert z_init.abs().max() == 0.0

    def test_timescale_initialization(self, cell):
        """Verify that state decay parameters are initialized with logarithmic spacing."""
        assert cell.decay_bias.shape == (cell.num_sublayers, cell.rank)
        assert cell.key_decay_bias.shape == (cell.num_sublayers, cell.rank)
        
        # Verify sigmoid values mapped from initialized logit bias are in bounds
        sig_decay = torch.sigmoid(cell.decay_bias)
        assert (sig_decay >= 0.0009).all()
        assert (sig_decay <= 0.999).all()
        
        # Values should be strictly sorted within each sublayer due to linspace initialization
        for i in range(cell.num_sublayers):
            vals = sig_decay[i]
            diffs = vals[1:] - vals[:-1]
            assert (diffs >= 0.0).all()


class TestSWDALRCellForward:
    """State updates and calculations."""

    def test_state_recurrent_updates(self, cell, B, S, d_model):
        """Verify state progression and FIFO index wrapping."""
        m = cell.get_initial_state(B, S)
        h_prev = torch.randn(B, S, d_model)
        y1 = torch.randn(B, S, d_model)
        y2 = torch.randn(B, S, d_model)

        # First transition
        _, m_new = cell(h_prev, y1, m, layer_idx=0, sublayer=0)
        fifo_buf, fifo_norm_buf, fifo_idx, S_new, z_new = m_new

        assert fifo_idx == 1
        assert_close(fifo_buf[:, :, 0, :], y1)

        # Second transition
        _, m_new2 = cell(h_prev, y2, m_new, layer_idx=0, sublayer=1)
        fifo_buf2, fifo_norm_buf2, fifo_idx2, S_new2, z_new2 = m_new2

        assert fifo_idx2 == 2
        assert_close(fifo_buf2[:, :, 0, :], y1)
        assert_close(fifo_buf2[:, :, 1, :], y2)

        # FIFO window wrapping check (window_size is 4)
        m_curr = m_new2
        for _ in range(5):
            y_dummy = torch.randn(B, S, d_model)
            _, m_curr = cell(h_prev, y_dummy, m_curr, layer_idx=1, sublayer=0)

        # Wrapping count: fifo_idx should be 7
        assert m_curr[2] == 7


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
        """Read, Write, and projection parameters should receive gradients once memory path is active."""
        # Enable memory read path by filling read bias
        with torch.no_grad():
            cell.gate_biases[0].fill_(3.0)

        # Mock non-zero historical states
        fifo_buf = torch.randn(B, S, cell.window_size, d_model)
        fifo_norm_buf = torch.randn(B, S, cell.window_size, d_model)
        fifo_idx = torch.tensor(1, dtype=torch.long)
        S_prev = torch.randn(B, S, cell.n_heads, cell.r_head, cell.d_head)
        z_prev = torch.randn(B, S, cell.n_heads, cell.r_head).abs() + 1.0  # avoid division by zero
        m = (fifo_buf, fifo_norm_buf, fifo_idx, S_prev, z_prev)

        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, m_new = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        (h_new.sum() + m_new[3].sum() + m_new[4].sum()).backward()

        assert cell.gate_weights.grad is not None
        assert cell.q_local_proj.weight.grad is not None
        assert cell.fifo_depth_bias.grad is not None
        assert cell.kv_deep_proj.weight.grad is not None
        assert cell.q_deep_proj.weight.grad is not None
        assert cell.query_bias.grad is not None
        assert cell.decay_bias.grad is not None
        assert cell.key_decay_bias.grad is not None
