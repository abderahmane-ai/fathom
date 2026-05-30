"""Unit tests for TransformerLayer (src/modules/transformer_layer.py)."""

import pytest
import torch
from omegaconf import OmegaConf

from src.modules.transformer_layer import TransformerLayer


@pytest.fixture
def config():
    return OmegaConf.create({
        "d_model": 64,
        "n_heads": 4,
        "ff_dim": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "residual_mode": "recurrent_residual",
        "recurrent_residual": {
            "read_gate_bias": -3.0,
            "forget_gate_bias": 3.0,
            "update_gate_bias": -2.0,
            "damp_gate_bias": 3.0,
            "eps": 1e-5,
            "gate_init_std": 0.01,
            "memory_gain_init": 0.0,
        }
    })


def test_grad_flows_through_rr_path(config):
    """Gradients must flow back to x and all gate projection weights."""
    B, S, d_model = 2, 4, config.d_model
    layer = TransformerLayer(config)

    x = torch.randn(B, S, d_model, requires_grad=True)
    m = layer.rr_cell.get_initial_state(B, S, device=x.device)

    h_new, _ = layer(x, layer_idx=0, m=m)
    h_new.sum().backward()

    assert x.grad is not None
    assert layer.rr_cell.read_proj[0].weight.grad is not None
    assert layer.rr_cell.update_proj[0].weight.grad is not None


def test_forward_with_memory_flow(config):
    """forward() must return (h_out, m_out) with correct shapes."""
    B, S, d_model = 2, 4, config.d_model
    layer = TransformerLayer(config)

    x = torch.randn(B, S, d_model)
    m_in = layer.rr_cell.get_initial_state(B, S, device=x.device)

    h_out, m_out = layer(x, layer_idx=0, m=m_in)

    assert h_out.shape == (B, S, d_model)
    assert m_out is not None
    assert m_out.shape == (B, S, d_model)


def test_forward_raises_for_attnres_modes(config):
    """forward() must raise ValueError when called in block_attnres or full_attnres mode."""
    config.residual_mode = "block_attnres"
    config.attnres_block = {"block_size": 2}
    layer = TransformerLayer(config)
    x = torch.randn(2, 4, config.d_model)
    with pytest.raises(ValueError):
        layer(x, layer_idx=0)
