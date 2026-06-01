"""Universal transformer layer supporting all residual modes.

Each layer wraps one Attention sublayer and one FFN sublayer.  The residual
connection style is determined by ``config.residual_mode``:

    "standard"           — plain h = h + y (Pre-LN)
    "recurrent_residual" — RR gated memory cell (shared across layers)
    "vega"               — VEGA EMA depth-memory cell (shared across layers)
    "block_attnres"      — BlockAttnRes softmax aggregation over block history
    "full_attnres"       — FullAttnRes softmax aggregation over all sublayer states
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attention import Attention
from .attnres_block import BlockAttnRes, FullAttnRes
from .ffn import FeedForward
from .hyper_connections import HyperConnection
from .norm import RMSNorm
from .recurrent_residual import RecurrentResidualCell
from .vega import VEGACell


class TransformerLayer(nn.Module):
    """One transformer layer with a pluggable residual mechanism.

    Args:
        config: OmegaConf config object (see ``conf/model/``).
        rr_cell: Pre-built RecurrentResidualCell to share; created from config if None.
        vega_cell: Pre-built VEGACell to share; created from config if None.
    """

    def __init__(
        self,
        config: Any,
        rr_cell: RecurrentResidualCell | None = None,
        vega_cell: VEGACell | None = None,
    ) -> None:
        super().__init__()

        self.ln_1 = RMSNorm(config.d_model)
        self.ln_2 = RMSNorm(config.d_model)
        self.attn = Attention(config.d_model, config.n_heads, getattr(config, "dropout", 0.1))
        self.ffn = FeedForward(config.d_model, config.ff_dim, getattr(config, "dropout", 0.1))

        self.rr_cell: RecurrentResidualCell | None = None
        self.vega_cell: VEGACell | None = None

        if config.residual_mode == "recurrent_residual":
            if rr_cell is None:
                rr_cfg = config.recurrent_residual
                rr_cell = RecurrentResidualCell(
                    config.d_model,
                    config.num_layers,
                    read_gate_bias=rr_cfg.read_gate_bias,
                    forget_gate_bias=getattr(rr_cfg, "forget_gate_bias", 3.0),
                    update_gate_bias=rr_cfg.update_gate_bias,
                    damp_gate_bias=getattr(rr_cfg, "damp_gate_bias", 3.0),
                    eps=rr_cfg.eps,
                    gate_init_std=rr_cfg.gate_init_std,
                    memory_gain_init=rr_cfg.memory_gain_init,
                )
            self.rr_cell = rr_cell

        elif config.residual_mode == "vega":
            if vega_cell is None:
                vega_cfg = config.vega
                vega_cell = VEGACell(
                    config.d_model,
                    config.num_layers,
                    rank=vega_cfg.rank,
                    n_heads=getattr(vega_cfg, "n_heads", 4),
                    n_fast_heads=getattr(vega_cfg, "n_fast_heads", 2),
                    fast_decay_range=tuple(getattr(vega_cfg, "fast_decay_range", (0.0, 1.2))),
                    slow_decay_range=tuple(getattr(vega_cfg, "slow_decay_range", (2.0, 4.5))),
                    read_gate_bias=vega_cfg.read_gate_bias,
                    write_gate_bias=getattr(vega_cfg, "write_gate_bias", -2.0),
                    damp_gate_bias=getattr(vega_cfg, "damp_gate_bias", 3.0),
                    eps=vega_cfg.eps,
                    gate_init_std=vega_cfg.gate_init_std,
                )
            self.vega_cell = vega_cell

        elif config.residual_mode == "block_attnres":
            block_size: int = config.attnres_block.block_size
            if block_size < 2 or block_size % 2 != 0:
                raise ValueError("attnres_block.block_size must be an even sublayer count ≥ 2.")
            self.layers_per_block: int = block_size // 2
            self.attn_res = BlockAttnRes(config.d_model)
            self.ffn_res = BlockAttnRes(config.d_model)

        elif config.residual_mode == "full_attnres":
            self.full_attn_res = FullAttnRes(config.d_model)
            self.full_ffn_res = FullAttnRes(config.d_model)

        elif config.residual_mode == "hyper_connection":
            hc_cfg = config.hyper_connection
            self.hc = HyperConnection(
                d_model=config.d_model,
                num_channels=int(getattr(hc_cfg, "num_channels", 2)),
                use_static_input=bool(getattr(hc_cfg, "use_static_input", False)),
                init_static_gate=float(getattr(hc_cfg, "init_static_gate", 0.0)),
            )

        self.residual_mode: str = config.residual_mode

    # ── Standard / RR / VEGA forward ────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        m: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass for standard, recurrent_residual, and vega modes.

        Args:
            x: Input hidden state, shape (B, S, d_model).
            layer_idx: 0-based layer index (used for depth biases).
            m: Memory state — None for standard, tensor for RR, tuple for VEGA.

        Returns:
            ``(h_new, m_new)`` where m_new is None for standard mode.

        Raises:
            ValueError: If called in block_attnres or full_attnres mode.
        """
        if self.residual_mode in {"block_attnres", "full_attnres"}:
            raise ValueError(
                f"Use forward_attnres / forward_full_attnres for mode '{self.residual_mode}'."
            )

        # Attention sublayer
        x_norm = self.ln_1(x)
        y_attn = self.attn(x_norm)

        if self.residual_mode == "standard":
            h_mid, m_mid = x + y_attn, None
        elif self.residual_mode == "recurrent_residual":
            assert m is not None and self.rr_cell is not None
            h_mid, m_mid = self.rr_cell(x, y_attn, m, layer_idx, sublayer=0)
        elif self.residual_mode == "vega":
            assert m is not None and self.vega_cell is not None
            h_mid, m_mid = self.vega_cell(x, y_attn, m, layer_idx, sublayer=0)
        else:
            raise ValueError(f"Unknown residual_mode: '{self.residual_mode}'")

        # FFN sublayer
        h_norm = self.ln_2(h_mid)
        y_ffn = self.ffn(h_norm)

        if self.residual_mode == "standard":
            h_new, m_new = h_mid + y_ffn, None
        elif self.residual_mode == "recurrent_residual":
            assert m_mid is not None and self.rr_cell is not None
            h_new, m_new = self.rr_cell(h_mid, y_ffn, m_mid, layer_idx, sublayer=1)
        elif self.residual_mode == "vega":
            assert m_mid is not None and self.vega_cell is not None
            h_new, m_new = self.vega_cell(h_mid, y_ffn, m_mid, layer_idx, sublayer=1)
        else:
            raise ValueError(f"Unknown residual_mode: '{self.residual_mode}'")

        return h_new, m_new

    # ── Block AttnRes forward ────────────────────────────────────────────────

    def forward_attnres(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer_idx: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass for block_attnres mode.

        Aggregates over completed blocks before each sublayer, then appends
        the current partial block to the history when a block boundary is reached.

        Args:
            blocks: Completed block states (including the embedding source).
            partial_block: Current in-block accumulated state.
            layer_idx: 0-based layer index (determines block boundaries).

        Returns:
            ``(blocks, partial_block)`` — updated history and current state.
        """
        # Pre-attention aggregation across block history.
        h_in = self.attn_res(blocks, partial_block)
        y_attn = self.attn(self.ln_1(h_in))
        partial_block = partial_block + y_attn

        # Pre-FFN aggregation across block history.
        h_in = self.ffn_res(blocks, partial_block)
        y_ffn = self.ffn(self.ln_2(h_in))
        partial_block = partial_block + y_ffn

        if (layer_idx + 1) % self.layers_per_block == 0:
            blocks = [*blocks, partial_block]

        return blocks, partial_block

    # ── Full AttnRes forward ─────────────────────────────────────────────────

    def forward_full_attnres(
        self,
        history: list[torch.Tensor],
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass for full_attnres mode.

        Appends every intermediate state to the growing history list.

        Args:
            history: All hidden states produced so far (including the embedding).
            x: Current hidden state entering this layer.

        Returns:
            ``(history, h_new)`` — extended history and updated hidden state.
        """
        h_in = self.full_attn_res([*history, x])
        y_attn = self.attn(self.ln_1(h_in))
        h_mid = x + y_attn
        history = [*history, h_mid]

        h_in = self.full_ffn_res(history)
        y_ffn = self.ffn(self.ln_2(h_in))
        h_new = h_mid + y_ffn
        history = [*history, h_new]

        return history, h_new

    # ── mHC forward ─────────────────────────────────────────────────────────

    def forward_hyperconnection(
        self,
        H: torch.Tensor,
        _layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass for hyper_connection (mHC) mode.

        Applies pre/post-mix around each sublayer. At init, channel 0
        accumulates standard Pre-LN residuals while channel 1 stays at its
        initial state (the embedding broadcast). Off-diagonal entries of
        W_pre / W_post are learned during training to route information
        between channels.

        Args:
            H: m-channel residual state, shape (B, S, m, d).
            _layer_idx: 0-based layer index (unused; reserved for future
                per-layer modulation of W_pre / W_post).

        Returns:
            New m-channel residual state of shape (B, S, m, d).
        """
        if self.residual_mode != "hyper_connection":
            raise ValueError(f"forward_hyperconnection called in mode '{self.residual_mode}'.")

        H_pre = self.hc.pre_mix(H)
        x_main = H_pre.select(dim=-2, index=0)
        y_attn = self.attn(self.ln_1(x_main))
        H_after_attn = self.hc.post_mix(H_pre, y_attn)
        x_mid_main = H_after_attn.select(dim=-2, index=0)
        y_ffn = self.ffn(self.ln_2(x_mid_main))
        H_new = self.hc.post_mix(H_after_attn, y_ffn)
        return H_new
