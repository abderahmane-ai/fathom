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

    def test_decoder_main_channel_close_to_standard_at_init(
        self, mhc_cfg, standard_cfg, B, S, d_model
    ):
        """At init, the mHC main channel should approximately follow the
        standard transformer's trajectory.  This is *not* bit-for-bit
        because the paper's init protocol gives H_post ≈ [1.462, 0.538]
        (not exactly [1, 0]) — see METHODOLOGY.md §5.2.
        """
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

        # The mHC decoder at init has y-gain 1.462 on the main channel and
        # 0.538 on the shadow channel.  The H_pre mix (≈[0.731, 0.269])
        # feeds the sublayer a mix of channels 0 and 1.  So the outputs
        # are close but not bit-equal; the gap accumulates over depth.
        # Tolerance 5e-1 is loose enough to absorb the accumulated drift
        # through 2 layers of attention + FFN, but tight enough to still
        # fail if the two decoders were doing completely different things.
        assert_close(mhc_out, std_out, atol=5e-1, rtol=5e-1)

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
            assert layer.hc.W_res.grad is not None


def copy_sublayer_weights(mhc_decoder, std_decoder) -> None:
    """Copy attention, FFN, and norm weights from the standard decoder to the
    mHC decoder so that the two have identical sublayer parameters.

    The mHC W_pre / W_post / W_res remain at their zero-init values, which
    combined with the paper's bias protocol give H_res ≈ I_2, H_pre ≈
    [0.731, 0.269], and H_post ≈ [1.462, 0.538] — the closest the paper
    gets to a standard residual on channel 0 (see METHODOLOGY.md §5.2).
    """
    assert len(mhc_decoder.layers) == len(std_decoder.layers)
    for mhc_layer, std_layer in zip(mhc_decoder.layers, std_decoder.layers, strict=True):
        mhc_layer.attn.load_state_dict(std_layer.attn.state_dict())
        mhc_layer.ffn.load_state_dict(std_layer.ffn.state_dict())
        mhc_layer.ln_1.load_state_dict(std_layer.ln_1.state_dict())
        mhc_layer.ln_2.load_state_dict(std_layer.ln_2.state_dict())
    mhc_decoder.ln_f.load_state_dict(std_decoder.ln_f.state_dict())
