"""Integration tests for TransformerDecoder (src/modules/transformer.py).

Validates:
* Forward pass shapes for all residual modes (standard, RR, AttnRes).
* Global memory flow in Recurrent Residual mode.
* Weight tying between embeddings and LM head.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

from src.modules.transformer import TransformerDecoder


@pytest.fixture
def config():
    return DictConfig(
        {
            "d_model": 64,
            "n_heads": 4,
            "ff_dim": 128,
            "num_layers": 4,
            "max_seq_len": 32,
            "vocab_size": 100,
            "dropout": 0.0,
            "residual_mode": "standard",
            "attnres_block": {"block_size": 4},
            "recurrent_residual": {
                "read_gate_bias": -3.0,
                "forget_gate_bias": 3.0,
                "update_gate_bias": -2.0,
                "damp_gate_bias": 3.0,
                "eps": 1e-5,
                "gate_init_std": 0.01,
                "memory_gain_init": 0.0,
            },
            "vega": {
                "rank": 8,
                "n_heads": 2,
                "n_fast_heads": 1,
                "read_gate_bias": -3.0,
                "write_gate_bias": -2.0,
                "damp_gate_bias": 3.0,
                "gate_init_std": 0.01,
                "eps": 1e-5,
            },
            "full_attnres": {"max_layers": 24},
        }
    )


def test_transformer_decoder_shapes(config):
    """Verify output logits shape across all modes."""
    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    for mode in ["standard", "recurrent_residual", "vega", "block_attnres", "full_attnres"]:
        config.residual_mode = mode
        model = TransformerDecoder(config)
        logits = model(input_ids)

        assert logits.shape == (B, S, config.vocab_size)


def test_weight_tying(config):
    """Verify that token embeddings and LM head share the same weights."""
    model = TransformerDecoder(config)
    assert model.lm_head.weight is model.token_embeddings.weight


def test_recurrent_residual_memory_flow(config):
    """Verify that memory flows across layers in RR mode."""
    config.residual_mode = "recurrent_residual"
    model = TransformerDecoder(config)
    B, S = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    from unittest.mock import MagicMock

    for layer in model.layers:
        layer.forward = MagicMock(side_effect=layer.forward)

    _ = model(input_ids)

    # Check that each layer received m and returned a new m
    for idx, layer in enumerate(model.layers):
        # pyrefly: ignore [missing-attribute]
        args, kwargs = layer.forward.call_args
        # args[2] is m
        assert args[2] is not None, f"Layer {idx} did not receive memory state m"
        assert args[2].shape == (B, S, config.d_model)


def test_vega_memory_flow(config):
    """Verify that memory flows across layers in VEGA mode."""
    config.residual_mode = "vega"
    model = TransformerDecoder(config)
    B, S = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    from unittest.mock import MagicMock

    for layer in model.layers:
        layer.forward = MagicMock(side_effect=layer.forward)

    _ = model(input_ids)

    # Check that each layer received m tuple (S, z) and returned a new m
    for idx, layer in enumerate(model.layers):
        # pyrefly: ignore [missing-attribute]
        args, kwargs = layer.forward.call_args
        m = args[2]
        assert m is not None, f"Layer {idx} did not receive memory state m"
        S_state, z_state = m
        n_heads = config.vega.get("n_heads", 4)
        r_head = config.vega.rank // n_heads
        assert S_state.shape == (B, S, n_heads, r_head, r_head)
        assert z_state.shape == (B, S, n_heads, r_head)


def test_shared_rr_cell_params(config):
    """Verify that all layers share the same rr_cell instance in RR mode."""
    config.residual_mode = "recurrent_residual"
    model = TransformerDecoder(config)

    rr_cell = model.rr_cell
    assert rr_cell is not None

    for layer in model.layers:
        assert layer.rr_cell is rr_cell, "Layers are not sharing the same RR cell instance"


def test_shared_vega_cell_params(config):
    """Verify that all layers share the same vega_cell instance in VEGA mode."""
    config.residual_mode = "vega"
    model = TransformerDecoder(config)

    vega_cell = model.vega_cell
    assert vega_cell is not None

    for layer in model.layers:
        assert layer.vega_cell is vega_cell, "Layers are not sharing the same VEGA cell instance"
