"""End-to-end tests for mHC mode in TransformerDecoder and TransformerLayer."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close

from src.modules import TransformerDecoder, TransformerLayer


@pytest.fixture
def mhc_cfg(d_model, n_heads, ff_dim, num_layers):
    return OmegaConf.create(
        {
            "d_model": d_model,
            "n_heads": n_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "max_seq_len": 32,
            "vocab_size": 256,
            "dropout": 0.0,
            "residual_mode": "hyper_connection",
            "hyper_connection": {
                "num_channels": 2,
                "use_static_input": False,
                "init_static_gate": 0.0,
            },
        }
    )


class TestTransformerLayerMHC:
    def test_layer_constructs_hc(self, mhc_cfg):
        layer = TransformerLayer(mhc_cfg)
        assert layer.residual_mode == "hyper_connection"
        assert hasattr(layer, "hc")
        assert layer.hc.num_channels == 2

    def test_layer_wrong_mode_raises(self, standard_cfg):
        layer = TransformerLayer(standard_cfg)
        with pytest.raises(ValueError, match="forward_hyperconnection called"):
            layer.forward_hyperconnection(torch.randn(2, 4, 2, 64), 0)


class TestTransformerDecoderMHC:
    def test_decoder_constructs_in_mhc_mode(self, mhc_cfg):
        decoder = TransformerDecoder(mhc_cfg)
        assert decoder.residual_mode == "hyper_connection"
        assert decoder.hc_channels == 2

    def test_decoder_forward_shape(self, mhc_cfg, B, S, d_model):
        decoder = TransformerDecoder(mhc_cfg)
        decoder.eval()
        input_ids = torch.randint(0, 256, (B, S))
        with torch.no_grad():
            logits = decoder(input_ids)
        assert logits.shape == (B, S, 256)

    def test_decoder_main_channel_matches_standard_at_init(
        self, mhc_cfg, standard_cfg, B, S, d_model
    ):
        """At init, the mHC main channel should follow the same trajectory as
        a standard transformer with identical sublayer weights."""
        torch.manual_seed(0)
        mhc_decoder = TransformerDecoder(mhc_cfg)
        torch.manual_seed(0)
        std_decoder = TransformerDecoder(standard_cfg)
        copy_sublayer_weights(mhc_decoder, std_decoder)

        mhc_decoder.eval()
        std_decoder.eval()
        input_ids = torch.randint(0, 256, (B, S))

        with torch.no_grad():
            mhc_out = mhc_decoder(input_ids)
            std_out = std_decoder(input_ids)

        assert_close(mhc_out, std_out, atol=1e-4, rtol=1e-4)

    def test_decoder_gradient_flows(self, mhc_cfg, B, S, d_model):
        decoder = TransformerDecoder(mhc_cfg)
        decoder.train()
        input_ids = torch.randint(0, 256, (B, S))
        logits = decoder(input_ids)
        loss = logits.float().sum()
        loss.backward()
        for layer in decoder.layers:
            assert layer.hc.W_pre.grad is not None
            assert layer.hc.W_post.grad is not None


def copy_sublayer_weights(mhc_decoder, std_decoder) -> None:
    """Copy attention, FFN, and norm weights from the standard decoder to the
    mHC decoder so that the two have identical sublayer parameters.

    The mHC W_pre / W_post remain at their zero-init values, which reduce
    mHC to standard Pre-LN on channel 0 — so the two decoders should
    produce identical outputs at init.
    """
    assert len(mhc_decoder.layers) == len(std_decoder.layers)
    for mhc_layer, std_layer in zip(mhc_decoder.layers, std_decoder.layers, strict=True):
        mhc_layer.attn.load_state_dict(std_layer.attn.state_dict())
        mhc_layer.ffn.load_state_dict(std_layer.ffn.state_dict())
        mhc_layer.ln_1.load_state_dict(std_layer.ln_1.state_dict())
        mhc_layer.ln_2.load_state_dict(std_layer.ln_2.state_dict())
    mhc_decoder.ln_f.load_state_dict(std_decoder.ln_f.state_dict())
