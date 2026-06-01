"""Design-ladder tests: verify the init contract for every residual mechanism.

The five mechanisms in this project form a **design ladder** of progressively
richer approximations of the same operation (letting a layer reach back into
the history of depth states).  See METHODOLOGY.md §1.1 for the full framing.

Every rung of the ladder is required to satisfy an **init contract** — the
property that at initialization, the alternative reduces to a known, well-
defined baseline.  This file codifies that contract rung by rung:

  | Rung | Mechanism | At init, h_new is            | Strict?             |
  |-----:|-----------|------------------------------|---------------------|
  |    0 | standard  | h_prev + (sublayer outputs)  | yes (trivial)       |
  |    1 | rr        | σ(+3)·h_prev + y ≈ 0.953·h + y | no (soft)          |
  |    2 | vega      | σ(+3)·h_prev + y ≈ 0.953·h + y | no (soft)          |
  |    3 | mhc       | h_prev + y (channel 0)        | yes (bit-for-bit)  |
  |    4 | attnres   | mean([*blocks, partial])      | n/a (uniform mean)  |

The parametrised summary at the bottom is a **documentation test** — it
always passes, but its parametrise block reads as a table of the exact
init contract each mechanism satisfies.
"""

from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.testing import assert_close

from src.modules.attnres_block import BlockAttnRes
from src.modules.hyper_connections import HyperConnection
from src.modules.recurrent_residual import RecurrentResidualCell
from src.modules.transformer_layer import TransformerLayer
from src.modules.vega import VEGACell

# σ(+3) = 1 / (1 + exp(-3)) — the analytical damp-gate value at init for
# both RR and VEGA, since both initialize the damp bias to +3.
SOFT_DAMP_EXACT: float = 1.0 / (1.0 + math.exp(-3.0))  # ≈ 0.9526

# Tolerance for the soft zero-start assertions: the damp gate is computed
# as σ(damp_proj(y_norm) + 3) where damp_proj is initialised with std 0.01,
# so the projection contribution is O(0.01·√d).  With d=64 this is ≈ 0.08,
# which moves σ by ≈ 2e-3.  We use 5e-2 to be safe across seeds.
SOFT_ZERO_START_TOL: float = 5e-2


def _make_layer_config(residual_mode: str) -> OmegaConf:
    """Build a minimal TransformerLayer config for the given residual mode."""
    return OmegaConf.create(
        {
            "d_model": 64,
            "n_heads": 4,
            "ff_dim": 128,
            "num_layers": 1,
            "dropout": 0.0,
            "residual_mode": residual_mode,
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
                "n_heads": 4,
                "n_fast_heads": 2,
                "read_gate_bias": -3.0,
                "write_gate_bias": -2.0,
                "damp_gate_bias": 3.0,
                "gate_init_std": 0.01,
                "eps": 1e-5,
            },
            "attnres_block": {"block_size": 2},
        }
    )


# ─────────────────────────────────────────────────────────────────────────
# Rung 0: Standard — trivial passthrough
# ─────────────────────────────────────────────────────────────────────────


def test_standard_residual_is_well_defined_at_init() -> None:
    """Standard mode: h_new = h_prev + (sublayer outputs), well-defined and finite.

    The standard mode has no "cell" to test in isolation; the layer wrapper
    does the addition explicitly.  This test verifies the layer is well-
    defined and produces a finite output of the correct shape.
    """
    layer = TransformerLayer(_make_layer_config("standard"))
    h_prev = torch.randn(2, 4, 64)
    h_new, m_new = layer(h_prev, layer_idx=0)

    assert h_new.shape == h_prev.shape
    assert m_new is None
    assert torch.isfinite(h_new).all()


# ─────────────────────────────────────────────────────────────────────────
# Rung 1: RR — soft zero-start
# ─────────────────────────────────────────────────────────────────────────


def test_rr_zero_start_at_init() -> None:
    """RR at init produces h_new ≈ σ(+3)·h_prev + y, not h_prev + y.

    The memory is *written* (update_gate ≈ 0.119 is not closed) but never
    *read* (read_gate ≈ 0.047 and memory_gain = 0).  This is the "soft"
    zero-start documented in METHODOLOGY.md §3.2.
    """
    cell = RecurrentResidualCell(d_model=64, num_layers=2)
    device = cell.y_gates_down.weight.device
    m = cell.get_initial_state(2, 4, device=device)
    h_prev = torch.randn(2, 4, 64)
    y = torch.randn(2, 4, 64)

    h_new, m_new = cell(h_prev, y, m, layer_idx=0, sublayer=0)

    # The read term is exactly zero (memory_gain=0, read_gate proj is non-zero
    # but multiplied by 0 gain).  So h_new = damp_gate * h_prev + y.
    expected = SOFT_DAMP_EXACT * h_prev + y
    assert_close(h_new, expected, atol=SOFT_ZERO_START_TOL, rtol=SOFT_ZERO_START_TOL)

    # The memory is non-trivially written: |m_new| should be in the range
    # [0, |tanh(y)| * 0.12] since update_gate ≈ 0.119 at init.
    assert m_new.abs().mean() < 0.5


def test_rr_init_biases_match_documented_values() -> None:
    """The four gate biases must be {read: -3, damp: +3, forget: +3, update: -2}.

    See METHODOLOGY.md §3.2 for the documented values and their effect on
    the soft zero-start.
    """
    cell = RecurrentResidualCell(d_model=64, num_layers=2)
    d = cell.d_model
    # y_gates_up bias is split into [read, damp, update] chunks of size d.
    assert_close(cell.y_gates_up.bias[:d], torch.full((d,), -3.0), atol=1e-5, rtol=1e-5)
    assert_close(cell.y_gates_up.bias[d : 2 * d], torch.full((d,), 3.0), atol=1e-5, rtol=1e-5)
    assert_close(cell.y_gates_up.bias[2 * d :], torch.full((d,), -2.0), atol=1e-5, rtol=1e-5)
    # forget_proj is a separate Sequential; its bias lives in forget_proj[1].
    assert_close(cell.forget_proj[1].bias, torch.full((d,), 3.0), atol=1e-5, rtol=1e-5)
    # memory_gain is exactly zero (the read term is therefore *exactly* zero
    # at init, not just approximately).
    assert_close(cell.memory_gain, torch.zeros(d), atol=1e-6, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Rung 2: VEGA — soft zero-start with the same form as RR
# ─────────────────────────────────────────────────────────────────────────


def test_vega_zero_start_at_init() -> None:
    """VEGA at init produces h_new ≈ σ(+3)·h_prev + y, not h_prev + y.

    The retrieval c_out is *exactly* zero because out_proj.weight is
    zero-initialized (verified below).  This is the soft zero-start
    documented in METHODOLOGY.md §4.5.
    """
    cell = VEGACell(d_model=64, num_layers=2, rank=8, n_heads=4, n_fast_heads=2)

    # The key init condition: out_proj.weight must be *exactly* zero.
    assert torch.equal(cell.out_proj.weight, torch.zeros_like(cell.out_proj.weight))

    device = cell.key_proj.weight.device
    m = cell.get_initial_state(2, 4, device=device)
    h_prev = torch.randn(2, 4, 64)
    y = torch.randn(2, 4, 64)

    h_new, _ = cell(h_prev, y, m, layer_idx=0, sublayer=0)

    # The read term is exactly zero (c_out = 0).  So h_new = damp_gate * h_prev + y.
    expected = SOFT_DAMP_EXACT * h_prev + y
    assert_close(h_new, expected, atol=SOFT_ZERO_START_TOL, rtol=SOFT_ZERO_START_TOL)


def test_vega_init_damp_bias_matches_documented_value() -> None:
    """The damp-bias parameter must be +3 so that damp_gate ≈ σ(+3) at init."""
    cell = VEGACell(d_model=64, num_layers=2, rank=8, n_heads=4, n_fast_heads=2)
    d = cell.d_model
    assert_close(cell.damp_bias, torch.full((d,), 3.0), atol=1e-5, rtol=1e-5)
    # damp_weight starts with std=0.01 — small, so damp_gate ≈ σ(0 + 3).
    assert cell.damp_weight.abs().max() < 0.1


# ─────────────────────────────────────────────────────────────────────────
# Rung 3: mHC — strict zero-start on the main channel
# ─────────────────────────────────────────────────────────────────────────


def test_mhc_strict_zero_start_at_init() -> None:
    """mHC at init: channel 0 == h_prev + y *exactly* (bit-for-bit).

    W_pre is identity on the m channels and W_post is identity on the m
    pre-mixed channels with the y-row adding to channel 0.  This is the
    **strictest** zero-start of any mechanism in the ladder.
    """
    mhc = HyperConnection(d_model=64, num_channels=2)
    H = torch.randn(2, 4, 2, 64)
    y = torch.randn(2, 4, 64)

    H_pre = mhc.pre_mix(H)
    H_post = mhc.post_mix(H_pre, y)

    # Main channel: H_pre[0] = H[0] (pre-mix is identity on the main channel).
    assert_close(H_pre[..., 0, :], H[..., 0, :], atol=1e-6, rtol=1e-6)
    # H_post[0] = H_pre[0] + y (post-mix adds y to the main channel only).
    assert_close(H_post[..., 0, :], H_pre[..., 0, :] + y, atol=1e-6, rtol=1e-6)
    # Combined: H_post[0] = H[0] + y — the exact standard residual.
    assert_close(H_post[..., 0, :], H[..., 0, :] + y, atol=1e-6, rtol=1e-6)
    # Shadow channel: H_post[1] = H_pre[1] = H[1] (pure carry-over at init).
    assert_close(H_post[..., 1, :], H[..., 1, :], atol=1e-6, rtol=1e-6)


def test_mhc_init_mix_matrices_have_expected_block_structure() -> None:
    """At init: W_pre = I_m, W_post = [[I_m], [e_0]] where e_0 = (1, 0, ..., 0).

    The off-diagonal entries of W_pre (routing from non-main channels into
    the main sublayer input) and the non-main rows of W_post (routing y
    into shadow channels) start at zero.  These are the only learnable
    routing parameters; they grow during training.
    """
    mhc = HyperConnection(d_model=64, num_channels=2)

    # W_pre (num_channels, num_channels) at init: identity.
    assert_close(mhc.W_pre.detach(), torch.eye(2), atol=1e-6, rtol=1e-6)

    # W_post (num_channels + 1, num_channels) at init:
    #   top num_channels rows: identity (channels → channels)
    #   bottom row: e_0 (y → main channel only)
    expected_W_post = torch.tensor(
        [
            [1.0, 0.0],  # channel 0 → channel 0
            [0.0, 1.0],  # channel 1 → channel 1
            [1.0, 0.0],  # y → main channel (channel 0)
        ]
    )
    assert_close(mhc.W_post.detach(), expected_W_post, atol=1e-6, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Rung 4: AttnRes — uniform mean at init
# ─────────────────────────────────────────────────────────────────────────


def test_attnres_uniform_mean_at_init() -> None:
    """AttnRes at init: pseudo_query=0 → uniform softmax → output = mean of inputs.

    This is the *weakest* zero-start of the alternatives — the cell is
    fully active, just with content-blind weights.
    """
    module = BlockAttnRes(d_model=64)
    # Verify the init condition explicitly.
    assert torch.equal(module.pseudo_query, torch.zeros(64))

    block0 = torch.randn(2, 4, 64)
    partial = torch.randn(2, 4, 64)
    out = module(blocks=[block0], partial_block=partial)

    # Uniform softmax over 2 tensors → simple mean.
    expected = (block0 + partial) / 2.0
    assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_attnres_uniform_mean_generalises_to_n_blocks() -> None:
    """The uniform-mean init contract holds for any number of completed blocks."""
    module = BlockAttnRes(d_model=64)
    blocks = [torch.ones(2, 4, 64) * i for i in range(3)]
    partial = torch.ones(2, 4, 64) * 3.0

    out = module(blocks=blocks, partial_block=partial)
    expected = torch.ones(2, 4, 64) * 1.5  # mean of [0, 1, 2, 3]
    assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# Cross-cutting: parametrised init-contract table
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rung", "mechanism", "init_form", "strict"),
    [
        (0, "standard", "h_prev + (sublayer outputs)", True),
        (1, "rr", "σ(+3)·h_prev + y ≈ 0.953·h_prev + y", False),
        (2, "vega", "σ(+3)·h_prev + y ≈ 0.953·h_prev + y", False),
        (3, "mhc", "h_prev + y (channel 0 only)", True),
        (4, "attnres", "mean([*blocks, partial_block])", None),
    ],
)
def test_design_ladder_init_contract_summary(
    rung: int, mechanism: str, init_form: str, strict: bool | None
) -> None:
    """Every rung's init contract, in one parametrised table.

    This test always passes — its purpose is to make the design-ladder
    design intent **greppable** and **visible in the test summary**.

    Run ``pytest tests/test_design_ladder.py -k design_ladder -v`` to see
    the table; the parametrise block above is the source of truth for
    which contract each rung satisfies.
    """
    assert 0 <= rung <= 4
    assert mechanism in {"standard", "rr", "vega", "mhc", "attnres"}
    assert isinstance(init_form, str) and len(init_form) > 0
    assert strict in {True, False, None}
