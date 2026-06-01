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
  |    3 | mhc       | H_res·H + 2·σ(±1)·y ≈ H + 1.462·y[0]+0.538·y[1] | no (paper's approximate) |
  |    4 | attnres   | mean([*blocks, partial])      | n/a (uniform mean)  |

  mHC has two algorithm variants (see METHODOLOGY.md §5.2):
  - ``algorithm="sinkhorn_knopp"`` (default) — the original DeepSeek mHC
    paper (arXiv:2512.24880), H_res computed via 20 Sinkhorn-Knopp iterations.
  - ``algorithm="permutation_convex"`` — the mHC-Lite variant
    (arXiv:2601.05732), H_res computed via convex combination of permutation
    matrices (Birkhoff-von Neumann, exact doubly-stochasticity for n=2).

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
# Rung 3: mHC — paper-faithful (DeepSeek mHC with Sinkhorn-Knopp,
# arXiv:2512.24880, the canonical implementation) approximate zero-start
# ─────────────────────────────────────────────────────────────────────────


def test_mhc_approximate_zero_start_at_init() -> None:
    """mHC at init: H_new = H_res · H + H_post · y, with the paper's init
    protocol giving H_pre ≈ [0.731, 0.269], H_post ≈ [1.462, 0.538], and
    H_res ≈ I_2.  This is **approximate** (not bit-for-bit) zero-start:
    channel 0 still gets close to a standard residual, but with a y-gain
    of 1.462 instead of 1.  See METHODOLOGY.md §5.2 for the rationale.

    Uses the canonical Sinkhorn-Knopp algorithm (default) — see
    ``test_mhc_lite_approximate_zero_start_at_init`` for the mHC-Lite
    variant (arXiv:2601.05732).
    """
    mhc = HyperConnection(d_model=64, num_channels=2, algorithm="sinkhorn_knopp")
    H = torch.randn(2, 4, 2, 64)
    y = torch.randn(2, 4, 64)

    # pre_mix / post_mix route through the dynamic init.
    x = mhc.pre_mix(H)
    H_new = mhc.post_mix(H, y)

    # pre_mix returns the mixed sublayer input, shape (B, S, d).
    assert x.shape == (2, 4, 64)
    # post_mix returns the new n-channel residual state, shape (B, S, n, d).
    assert H_new.shape == (2, 4, 2, 64)

    # 1) H_res ≈ I_2 → H_res · H ≈ H.
    # 2) H_post ≈ [1.462, 0.538] → y_post[0] ≈ 1.462·y, y_post[1] ≈ 0.538·y.
    # Tolerance 1e-1 absorbs the larger deviation of H_res from I_2 under
    # 20 SK iterations (compared to mHC-Lite's exact construction).
    assert_close(H_new[..., 0, :], H[..., 0, :] + 1.4621 * y, atol=1e-1, rtol=1e-1)
    assert_close(H_new[..., 1, :], H[..., 1, :] + 0.5379 * y, atol=1e-1, rtol=1e-1)


def test_mhc_lite_approximate_zero_start_at_init() -> None:
    """mHC-Lite at init: same approximate zero-start as mHC, but H_res is
    exactly I_2 at init (no SK approximation gap).  See arXiv:2601.05732.
    """
    mhc = HyperConnection(d_model=64, num_channels=2, algorithm="permutation_convex")
    H = torch.randn(2, 4, 2, 64)
    y = torch.randn(2, 4, 64)
    _ = mhc.pre_mix(H)
    H_new = mhc.post_mix(H, y)
    # mHC-Lite is exact: H_res = softmax([0,-8])·[I,swap] ≈ I_2.
    assert_close(H_new[..., 0, :], H[..., 0, :] + 1.4621 * y, atol=5e-3, rtol=5e-3)
    assert_close(H_new[..., 1, :], H[..., 1, :] + 0.5379 * y, atol=5e-3, rtol=5e-3)


def test_mhc_init_contract_at_init() -> None:
    """At init, the dynamic mix matrices are fixed by the paper's bias-only
    protocol (W=0, α=0.01).  This is what gives the approximate zero-start
    in ``test_mhc_approximate_zero_start_at_init`` and its mHC-Lite sibling.

    Verifies the SK variant (default); the mHC-Lite variant has the same
    init for b_pre / b_post / W_pre / W_post / α_* — only b_res shape
    differs (see test_mhc_init_contract_at_init_lite).
    """
    mhc = HyperConnection(d_model=64, num_channels=2, algorithm="sinkhorn_knopp")

    # W_pre / W_post shape (n*d, n) = (128, 2) for d=64, n=2.
    assert torch.equal(mhc.W_pre.detach(), torch.zeros(128, 2))
    assert torch.equal(mhc.W_post.detach(), torch.zeros(128, 2))
    # W_res shape (n*d, n^2) = (128, 4) for SK.
    assert torch.equal(mhc.W_res.detach(), torch.zeros(128, 4))

    # All α-gates start at 0.01 (the paper's learnable temperature).
    assert_close(mhc.alpha_pre.detach(), torch.tensor([0.01]))
    assert_close(mhc.alpha_post.detach(), torch.tensor([0.01]))
    assert_close(mhc.alpha_res.detach(), torch.tensor([0.01]))

    # Biases: b_pre = b_post = [1, -1] (main channel +1, shadow −1).
    assert_close(mhc.b_pre.detach()[0, 0], torch.tensor(1.0))
    assert_close(mhc.b_pre.detach()[0, 1], torch.tensor(-1.0))
    assert_close(mhc.b_post.detach()[0, 0], torch.tensor(1.0))
    assert_close(mhc.b_post.detach()[0, 1], torch.tensor(-1.0))

    # b_res is a 2x2 matrix for SK: 0 at the identity-matrix [0,0] entry,
    # −8 everywhere else.  After exp and 20 SK iterations this converges
    # close to I_2 (the mHC paper's init intent).
    b_res = mhc.b_res.detach()
    assert b_res.shape == (1, 2, 2)
    assert_close(b_res[0, 0, 0], torch.tensor(0.0))
    assert_close(b_res[0, 0, 1], torch.tensor(-8.0))
    assert_close(b_res[0, 1, 0], torch.tensor(-8.0))
    assert_close(b_res[0, 1, 1], torch.tensor(-8.0))


def test_mhc_init_contract_at_init_lite() -> None:
    """mHC-Lite: b_res is a 2-vector (0 on identity-perm, -8 on swap),
    instead of a 2x2 matrix like the SK variant.  All other init values
    match the SK variant.
    """
    mhc = HyperConnection(d_model=64, num_channels=2, algorithm="permutation_convex")
    # W_res shape (n*d, n!) = (128, 2) for mHC-Lite.
    assert torch.equal(mhc.W_res.detach(), torch.zeros(128, 2))
    # b_res is a 2-vector.
    b_res = mhc.b_res.detach()
    assert b_res.shape == (1, 2)
    assert_close(b_res[0, 0], torch.tensor(0.0))
    assert_close(b_res[0, 1], torch.tensor(-8.0))


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
        (3, "mhc", "H_res·H + 2σ(±1)·y ≈ H + 1.462·y[0] + 0.538·y[1]", False),
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
