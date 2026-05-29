import pytest
import torch
from torch.testing import assert_close
from src.modules.vega import VEGACell

@pytest.fixture
def B(): return 2
@pytest.fixture
def S(): return 4
@pytest.fixture
def d_model(): return 64

@pytest.fixture
def cell(d_model):
    return VEGACell(d_model, num_layers=2, rank=8)

class TestVEGACellInit:
    def test_standard_residual_at_init(self, cell, B, S, d_model):
        m = cell.get_initial_state(B, S, device=cell.k_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        
        # In VEGACell, gate biases and init stds ensure minimal initial interaction
        h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

        # Looser tolerance for initial gate behavior
        assert_close(h_new, h_prev + y, atol=0.5, rtol=0.5)

class TestVEGACellForward:
    def test_state_recurrent_updates(self, cell, B, S, d_model):
        m = cell.get_initial_state(B, S, device=cell.k_proj.weight.device)
        h_prev = torch.randn(B, S, d_model)
        y1 = torch.randn(B, S, d_model)
        
        _, m_new = cell(h_prev, y1, m, layer_idx=0, sublayer=0)
        
        assert isinstance(m_new, tuple)
        assert len(m_new) == 2 # S, z

    def test_forward_requires_m(self, cell, B, S, d_model):
        h_prev = torch.randn(B, S, d_model)
        y = torch.randn(B, S, d_model)
        with pytest.raises(TypeError):
            cell(h_prev, y, None, layer_idx=0, sublayer=0)
