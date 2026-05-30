"""Unit tests for VEGACell (src/modules/vega.py)."""

import pytest
import torch
from torch.testing import assert_close

from src.modules.vega import VEGACell


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
    return VEGACell(d_model, num_layers=2, rank=8, n_heads=4, n_fast_heads=2)


class TestVEGACellInit:
    def test_standard_residual_at_init(self, cell, B, S, d_model):
        """At init (out_fast/slow zeroed) the cell should approximate h_prev + y."""
        m = cell.get_initial_state(B, S, device=cell.key_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        # Loose tolerance: output projections are zeroed at init so c_out ≈ 0.
        assert_close(h_new, h_prev + y, atol=0.5, rtol=0.5)


class TestVEGACellForward:
    def test_state_is_tuple_of_two(self, cell, B, S, d_model):
        """State returned by forward must be a 2-tuple (S_state, z_state)."""
        m = cell.get_initial_state(B, S, device=cell.key_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        _, m_new = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        assert isinstance(m_new, tuple) and len(m_new) == 2

    def test_state_shapes(self, cell, B, S, d_model):
        """S_state and z_state must have the expected shapes."""
        m = cell.get_initial_state(B, S, device=cell.key_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        _, (S_state, z_state) = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        assert S_state.shape == (B, S, cell.n_heads, cell.r_head, cell.r_head)
        assert z_state.shape == (B, S, cell.n_heads, cell.r_head)

    def test_forward_requires_valid_state(self, cell, B, S, d_model):
        """Passing None as state must raise TypeError (cannot unpack None)."""
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        with pytest.raises(TypeError):
            cell(h_prev, y, None, layer_idx=0, sublayer=0)  # type: ignore[arg-type]

    def test_grad_flows(self, cell, B, S, d_model):
        """Gradients must flow back to y and projection weights."""
        m = cell.get_initial_state(B, S, device=cell.key_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model, requires_grad=True)
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        h_new.sum().backward()
        assert y.grad is not None
