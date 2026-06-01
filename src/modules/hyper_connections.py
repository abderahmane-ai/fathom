"""Multi-Head Hyper-Connections (mHC) — m-channel residual mixing.

Reference: "mHC: Manifold-Constrained Hyper-Connections", DeepSeek-AI, arXiv:2512.24880.

The residual stream is duplicated into ``m`` parallel channels of shape
``(B, S, m, d)``. Two learned matrices re-mix these channels at every
sublayer:

    W_pre  : (m + 1) x m    # mixes previous m channels with sublayer input
    W_post : (m + 1) x m    # mixes the m pre-mixed channels with the new y

mHC-Lite (m=2) is the practical, parameter-efficient variant. At init
W_pre is identity on the m channels and W_post is identity on the m
pre-mixed channels with the y-row adding to channel 0, so mHC reduces
to standard Pre-LN residuals on channel 0 while channel 1 remains an
untouched scratch state. Off-diagonal entries are learned during
training to route information between channels and across depth.

Design-ladder role (see METHODOLOGY.md §1.1, §5.4):
    mHC is **Rung 3** of the design ladder, but sits *orthogonally* to the
    other history-aggregation rungs (RR / VEGA / AttnRes).  Where the
    other three rungs ask "how do we let a layer reach back into the
    history of previous hidden states?", mHC asks "how do we let a single
    sublayer mix across ``m`` parallel residual channels?".  The two
    ideas are composable in principle but kept separate in this project
    so each can be evaluated in isolation.  In the benchmark suite, mHC
    is the **recently-published reference baseline** — included as the
    "what does the latest concurrent work propose?" comparison point,
    with the mHC-Lite (m=2) variant chosen to keep parameter overhead
    negligible (O(m²·d) per sublayer, i.e. 4d for m=2).

Init contract (verified by
tests/test_mhc_integration.py::test_decoder_main_channel_matches_standard_at_init):
    mHC has the **strictest zero-start** of any mechanism in the ladder.
    At init, W_pre = I and W_post = I except for the main-channel row
    (which is ``1, 0, 0, ..., 0``), so the main channel obeys
    ``H_l[0] = h_{l-1} + y_l`` *bit-for-bit* — an exact standard Pre-LN
    residual.  The other ``m - 1`` channels are pure carry-over
    (``H_l[k] = H_{l-1}[k]``).  Off-diagonal entries of W_pre and W_post
    are the only learnable parameters; they start at zero and grow during
    training to route information between channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    """m-channel residual mixer for one transformer layer (mHC-Lite default m=2).

    Provides ``pre_mix`` and ``post_mix`` that the transformer layer composes
    around its attention/FFN sublayers. The module owns no sublayer of its
    own.

    Args:
        d_model: Hidden dimension of each channel.
        num_channels: m. The number of parallel residual channels.
        use_static_input: When True, the layer's main-channel input (the
            first channel of the previous state) is concatenated to the
            pre-mix input as a static anchor. Disable for mHC-Lite.
        init_static_gate: Initial value for the static-input row of W_pre.
            When 0, the static input is ignored at init (zero-start protocol).
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int = 2,
        use_static_input: bool = False,
        init_static_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be >= 1.")
        self.d_model = d_model
        self.num_channels = num_channels
        self.use_static_input = use_static_input

        pre_in = num_channels + (1 if use_static_input else 0)
        self.W_pre = nn.Parameter(torch.zeros(pre_in, num_channels))
        self.W_post = nn.Parameter(torch.zeros(num_channels + 1, num_channels))

        with torch.no_grad():
            for i in range(num_channels):
                self.W_pre[i, i] = 1.0
                self.W_post[i, i] = 1.0
            if use_static_input:
                self.W_pre[num_channels, 0] = float(init_static_gate)
            self.W_post[num_channels, 0] = 1.0

    def pre_mix(
        self,
        H: torch.Tensor,
        static_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix previous channels into the sublayer input.

        Args:
            H: Previous residual state, shape (B, S, m, d).
            static_input: Optional static input, shape (B, S, d). Required
                when ``use_static_input`` is True.

        Returns:
            Pre-mixed state of shape (B, S, m, d).
        """
        if self.use_static_input:
            if static_input is None:
                raise ValueError("static_input is required when use_static_input is True.")
            H_in = torch.cat([H, static_input.unsqueeze(-2)], dim=-2)
        else:
            H_in = H
        return torch.einsum("bsid, ic -> bscd", H_in, self.W_pre)

    def post_mix(
        self,
        H_pre: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Mix pre-mixed channels with the new sublayer output.

        Args:
            H_pre: Pre-mixed state, shape (B, S, m, d).
            y: New sublayer output, shape (B, S, d).

        Returns:
            New residual state of shape (B, S, m, d).
        """
        y_expanded = y.unsqueeze(-2)
        H_in = torch.cat([H_pre, y_expanded], dim=-2)
        return torch.einsum("bsid, ic -> bscd", H_in, self.W_post)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_channels={self.num_channels}, "
            f"use_static_input={self.use_static_input}"
        )
