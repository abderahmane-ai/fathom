"""Unit tests for TransformerLayer (src/modules/transformer_layer.py)."""

import pytest
import torch
from omegaconf import OmegaConf

from src.modules.transformer_layer import TransformerLayer


@pytest.fixture
def config():
    return OmegaConf.create(
        {
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
        }
    )


def test_grad_flows_through_rr_path(config):
    """Gradients must flow back to x and all gate projection weights."""
    B, S, d_model = 2, 4, config.d_model
    layer = TransformerLayer(config)

    x = torch.randn(B, S, d_model, requires_grad=True)
    m = layer.rr_cell.get_initial_state(B, S, device=x.device)

    h_new, _ = layer(x, layer_idx=0, m=m)
    h_new.sum().backward()

    assert x.grad is not None
    assert layer.rr_cell.y_gates_down.weight.grad is not None


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


def test_grad_flows_through_vega_path(config):
    """Gradients must flow back to x and all VEGA projections."""
    config.residual_mode = "vega"
    B, S, d_model = 2, 4, config.d_model
    layer = TransformerLayer(config)

    x = torch.randn(B, S, d_model, requires_grad=True)
    m = layer.vega_cell.get_initial_state(B, S, device=x.device)

    h_new, _ = layer(x, layer_idx=0, m=m)
    h_new.sum().backward()

    assert x.grad is not None
    assert layer.vega_cell.qkv_proj.weight.grad is not None


def test_forward_with_vega_memory_flow(config):
    """forward() must return (h_out, m_out) in VEGA mode with correct shapes."""
    config.residual_mode = "vega"
    B, S, d_model = 2, 4, config.d_model
    layer = TransformerLayer(config)

    x = torch.randn(B, S, d_model)
    m_in = layer.vega_cell.get_initial_state(B, S, device=x.device)

    h_out, m_out = layer(x, layer_idx=0, m=m_in)

    assert h_out.shape == (B, S, d_model)
    assert m_out is not None
    S_state, z_state = m_out
    if layer.vega_cell.use_vector_state:
        assert S_state.shape == (B, S, layer.vega_cell.n_heads, layer.vega_cell.r_head)
    else:
        assert S_state.shape == (
            B,
            S,
            layer.vega_cell.n_heads,
            layer.vega_cell.r_head,
            layer.vega_cell.r_head,
        )
    assert z_state.shape == (B, S, layer.vega_cell.n_heads, layer.vega_cell.r_head)


def _build_layer_for_mode(config, mode: str) -> TransformerLayer:
    """Configure extra config fields a mode needs, then build the layer."""
    config.residual_mode = mode
    if mode == "block_attnres":
        config.attnres_block = {"block_size": 2}
    elif mode == "hyper_connection":
        config.hyper_connection = {
            "num_channels": 2,
            "use_static_input": False,
            "init_static_gate": 0.0,
        }
    return TransformerLayer(config)


def _call_layer_for_mode(layer: TransformerLayer, mode: str, config):
    """Invoke the unified layer(...) with mode-appropriate args; return the raw output."""
    B, S, d = 2, 4, config.d_model
    x = torch.randn(B, S, d)
    if mode == "standard":
        return layer(x, layer_idx=0)
    if mode == "recurrent_residual":
        m = layer.rr_cell.get_initial_state(B, S, device=x.device)
        return layer(x, layer_idx=0, m=m)
    if mode == "vega":
        m = layer.vega_cell.get_initial_state(B, S, device=x.device)
        return layer(x, layer_idx=0, m=m)
    if mode == "block_attnres":
        return layer([x], x.clone(), 0)
    if mode == "full_attnres":
        return layer([x], x)
    if mode == "hyper_connection":
        H = x.unsqueeze(-2).expand(B, S, 2, d).contiguous()
        return layer(H, 0)
    raise AssertionError(f"unhandled mode: {mode}")


def _assert_mode_output_shape(out, mode: str, config) -> None:
    """Verify the unified dispatch produced the shape the mode-specific method promises."""
    B, S, d = 2, 4, config.d_model
    if mode == "standard":
        h, m = out
        assert h.shape == (B, S, d)
        assert m is None
    elif mode == "recurrent_residual":
        h, m = out
        assert h.shape == (B, S, d)
        assert m.shape == (B, S, d)
    elif mode == "vega":
        h, m = out
        assert h.shape == (B, S, d)
        assert m is not None  # tuple of (S_state, z_state)
    elif mode == "block_attnres":
        blocks, partial = out
        assert isinstance(blocks, list)
        assert partial.shape == (B, S, d)
    elif mode == "full_attnres":
        history, h_new = out
        assert isinstance(history, list)
        assert h_new.shape == (B, S, d)
    elif mode == "hyper_connection":
        assert out.shape == (B, S, 2, d)
    else:
        raise AssertionError(f"unhandled mode: {mode}")


@pytest.mark.parametrize(
    "mode",
    [
        "standard",
        "recurrent_residual",
        "vega",
        "block_attnres",
        "full_attnres",
        "hyper_connection",
    ],
)
def test_forward_dispatch_routes_to_mode_specific_method_and_fires_hooks(config, mode: str) -> None:
    """Unified forward() must dispatch to the correct mode-specific method for every
    residual mode, and must trigger PyTorch forward hooks — the whole purpose of
    the routing refactor (commit e9a1a41) is to make hooks fire uniformly so the
    per-layer grad / activation trackers in benchmarks/common/ see every mode.
    """
    layer = _build_layer_for_mode(config, mode)

    hook_calls: list[object] = []
    layer.register_forward_hook(lambda _module, _inputs, output: hook_calls.append(output))

    out = _call_layer_for_mode(layer, mode, config)

    assert len(hook_calls) == 1, f"forward hook did not fire for mode '{mode}'"
    _assert_mode_output_shape(out, mode, config)
