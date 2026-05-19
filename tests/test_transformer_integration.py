"""Integration tests for TransformerDecoder across different residual modes."""
import pytest
import torch
from omegaconf import OmegaConf

from src.modules.transformer import TransformerDecoder


@pytest.fixture
def base_config():
    return OmegaConf.create({
        "d_model": 128,
        "n_heads": 4,
        "ff_dim": 512,
        "num_layers": 4,
        "max_seq_len": 64,
        "vocab_size": 1000,
        "dropout": 0.1,
        "residual_mode": "standard"
    })


@pytest.mark.parametrize("mode", ["standard", "recurrent_residual", "attnres_block"])
def test_transformer_forward_modes(base_config, mode):
    """Verify all residual modes produce correct output shapes and handle forward passes."""
    config = base_config
    config.residual_mode = mode
    
    if mode == "attnres_block":
        config.attnres_block = {"block_size": 4} # 2 layers per block
    
    model = TransformerDecoder(config)
    
    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    
    logits = model(input_ids)
    
    assert logits.shape == (B, S, config.vocab_size)
    assert not torch.isnan(logits).any(), f"NaN detected in {mode} logits"


def test_recurrent_residual_memory_persistence(base_config):
    """Verify that memory is shared and flows across layers in RR mode."""
    config = base_config
    config.residual_mode = "recurrent_residual"
    
    model = TransformerDecoder(config)
    rr_cell = model.rr_cell
    
    B, S = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    
    # Run forward
    _ = model(input_ids)
    
    # After forward, memory should be non-zero (unless everything is gated off, which is unlikely with random weights)
    assert rr_cell.m.abs().sum() > 0.0, "Memory should be non-zero after forward pass"
    
    # Reset should clear it back to m_init
    rr_cell.reset_memory(B, S)
    expected_m = rr_cell.m_init.view(1, 1, -1).expand(B, S, -1)
    torch.testing.assert_close(rr_cell.m, expected_m)
