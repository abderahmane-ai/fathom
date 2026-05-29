"""Integration tests for TransformerDecoder across different residual modes."""

import pytest
import torch
from omegaconf import OmegaConf

from src.modules.transformer import TransformerDecoder


@pytest.fixture
def base_config():
    return OmegaConf.create(
        {
            "d_model": 128,
            "n_heads": 4,
            "ff_dim": 512,
            "num_layers": 4,
            "max_seq_len": 64,
            "vocab_size": 1000,
            "dropout": 0.1,
            "residual_mode": "standard",
            "recurrent_residual": {
                "read_gate_bias": -3.0,
                "update_gate_bias": -2.0,
                "gate_init_std": 0.01,
                "memory_gain_init": 0.0,
                "eps": 1e-5,
            },
            "full_attnres": {"max_layers": 24},
        }
    )


@pytest.mark.parametrize(
    "mode",
    ["standard", "recurrent_residual", "vega", "block_attnres", "full_attnres"],
)
def test_transformer_forward_modes(base_config, mode):
    """Verify all residual modes produce correct output shapes and handle forward passes."""
    config = base_config
    config.residual_mode = mode
    if mode == "block_attnres":
        config.attnres_block = {"block_size": 4}
    if mode == "vega":
        config.vega = {
            "window_size": 4,
            "rank": 8,
            "n_heads": 4,
            "decay_bias_init": 3.0,
            "read_gate_bias": -3.0,
            "write_gate_bias": -2.0,
            "gate_init_std": 0.01,
            "eps": 1e-5,
        }

    model = TransformerDecoder(config)

    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    logits = model(input_ids)

    assert logits.shape == (B, S, config.vocab_size)
    assert not torch.isnan(logits).any(), f"NaN detected in {mode} logits"


def test_recurrent_residual_memory_persistence(base_config):
    """Verify that memory is initialized correctly in RR mode."""
    config = base_config
    config.residual_mode = "recurrent_residual"
    # Add required RR config
    config.recurrent_residual = {
        "read_gate_bias": -3.0,
        "update_gate_bias": -2.0,
        "gate_init_std": 0.01,
        "memory_gain_init": 0.0,
        "eps": 1e-5,
    }

    model = TransformerDecoder(config)
    rr_cell = model.rr_cell

    B, S = 1, 8

    # Check initial state
    m_init = rr_cell.get_initial_state(B, S)
    expected_m = rr_cell.m_init.view(1, 1, -1).expand(B, S, -1)
    torch.testing.assert_close(m_init, expected_m)


@pytest.mark.skip(reason="Legacy test logic needs refactor")
def test_legacy_attnres_mode_is_rejected(base_config):
    """Only the canonical block_attnres residual mode should be accepted."""
    base_config.residual_mode = "invalid_mode"
    base_config.attnres_block = {"block_size": 4}
    with pytest.raises(ValueError, match="Unsupported residual_mode"):
        TransformerDecoder(base_config)
