"""Integration tests for TransformerDecoder across all residual modes."""

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
            "dropout": 0.0,
            "residual_mode": "standard",
            "recurrent_residual": {
                "read_gate_bias": -3.0,
                "forget_gate_bias": 3.0,
                "update_gate_bias": -2.0,
                "damp_gate_bias": 3.0,
                "gate_init_std": 0.01,
                "memory_gain_init": 0.0,
                "eps": 1e-5,
            },
            "vega": {
                "rank": 8,
                "n_heads": 4,
                "n_fast_heads": 2,
                "read_gate_bias": -3.0,
                "write_gate_bias": -2.0,
                "damp_gate_bias": 3.0,
                "gate_init_std": 0.01,
                "eps": 1e-5,
            },
            "attnres_block": {"block_size": 4},
            "full_attnres": {"max_layers": 24},
        }
    )


@pytest.mark.parametrize(
    "mode",
    ["standard", "recurrent_residual", "vega", "block_attnres", "full_attnres"],
)
def test_transformer_decoder_shapes(base_config, mode):
    """All residual modes must produce logits of the correct shape with no NaNs."""
    base_config.residual_mode = mode
    model = TransformerDecoder(base_config)

    B, S = 2, 16
    input_ids = torch.randint(0, base_config.vocab_size, (B, S))
    logits = model(input_ids)

    assert logits.shape == (B, S, base_config.vocab_size)
    assert not torch.isnan(logits).any(), f"NaN in {mode} logits"


def test_recurrent_residual_initial_state(base_config):
    """RR initial state must match the expanded m_init parameter."""
    base_config.residual_mode = "recurrent_residual"
    model = TransformerDecoder(base_config)
    rr_cell = model.rr_cell
    assert rr_cell is not None

    B, S = 1, 8
    m_init = rr_cell.get_initial_state(B, S, device=rr_cell.m_init.device)
    expected = rr_cell.m_init.view(1, 1, -1).expand(B, S, -1)
    torch.testing.assert_close(m_init, expected)


def test_unknown_residual_mode_raises(base_config):
    """TransformerLayer must raise ValueError for unsupported residual modes."""
    base_config.residual_mode = "invalid_mode"
    # forward() raises on unknown mode — instantiation itself is fine.
    model = TransformerDecoder(base_config)
    x = torch.randint(0, base_config.vocab_size, (2, 8))
    with pytest.raises(ValueError, match="Unknown residual_mode"):
        model(x)


def test_full_attnres_exceeds_max_layers_raises(base_config):
    """full_attnres must raise if num_layers exceeds max_layers cap."""
    base_config.residual_mode = "full_attnres"
    base_config.full_attnres = {"max_layers": 2}  # fewer than the 4 layers in base_config
    with pytest.raises(ValueError, match="limited to"):
        TransformerDecoder(base_config)
