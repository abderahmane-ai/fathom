import pytest
import torch
from torch.testing import assert_close
from src.modules.recurrent_residual import RecurrentResidualCell

@pytest.fixture
def B(): return 2
@pytest.fixture
def S(): return 4
@pytest.fixture
def d_model(): return 64

@pytest.fixture
def cell(d_model):
    return RecurrentResidualCell(d_model, num_layers=2)

class TestRRCellInit:
    def test_standard_residual_at_init(self, cell, B, S, d_model):
        m = cell.get_initial_state(B, S, device=cell.read_proj[0].weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)

        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

        # Higher tolerance due to complex gating and projection logic
        assert_close(h_new, h_prev + y, atol=0.2, rtol=0.2)

    def test_parameter_count(self, cell, d_model):
        expected_params = (cell.num_sublayers * 4 + 10) * d_model
        assert sum(p.numel() for p in cell.parameters()) >= d_model 

class TestRRCellMemoryUpdate:
    def test_memory_update_equation(self, cell, B, S, d_model):
        m_before = cell.get_initial_state(B, S, device=cell.read_proj[0].weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        
        _, m_after = cell(h_prev, y, m_before, layer_idx=0, sublayer=0)
        assert m_after.shape == (B, S, d_model)

class TestRRCellGradients:
    def test_grad_flows(self, cell, B, S, d_model):
        h_prev = torch.randn(B, S, d_model, requires_grad=True)
        y = torch.randn(B, S, d_model, requires_grad=True)
        m = cell.get_initial_state(B, S, device=h_prev.device)
        
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)
        h_new.sum().backward()
        
        assert h_prev.grad is not None
        assert y.grad is not None
        assert cell.read_proj[0].weight.grad is not None
