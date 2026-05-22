"""Transformer layer orchestrator with multi-mode residual support.

Supported Architectures:
1.  **Standard / RecurrentResidual**:
    Standard Pre-LN flow with two sublayers (Attn, FFN), each followed by
    either a simple addition or a gated RR cell transition.
2.  **Block-AttnRes**:
    Implements cross-block aggregation where sublayers attend to previous
    block states stored in a shared list.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attention import Attention
from .ffn import FeedForward
from .recurrent_residual import RecurrentResidualCell


class TransformerLayer(nn.Module):
    """Universal Transformer layer supporting gated and block residuals.

    Args:
        config: Configuration exposing d_model, n_heads, ff_dim, and residual_mode.
    """

    def __init__(self, config: Any, rr_cell: RecurrentResidualCell | None = None) -> None:
        super().__init__()

        # Pre-sublayer normalisation layers.
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = Attention(config.d_model, config.n_heads, getattr(config, "dropout", 0.1))
        self.ffn = FeedForward(config.d_model, config.ff_dim, getattr(config, "dropout", 0.1))

        # ── Mode-specific residual modules ──────────────────────────────────
        self.rr_cell: nn.Module | None = None

        if config.residual_mode == "recurrent_residual":
            if rr_cell is None:
                rr_cfg = config.recurrent_residual
                rr_cell = RecurrentResidualCell(
                    config.d_model,
                    config.num_layers,
                    read_gate_bias=rr_cfg.read_gate_bias,
                    update_gate_bias=rr_cfg.update_gate_bias,
                    eps=rr_cfg.eps,
                    gate_init_std=rr_cfg.gate_init_std,
                    memory_gain_init=rr_cfg.memory_gain_init,
                )
            self.rr_cell = rr_cell

        elif config.residual_mode == "block_attnres":
            from .attnres_block import BlockAttnRes

            block_size: int = config.attnres_block.block_size
            if block_size < 2 or block_size % 2 != 0:
                raise ValueError("attnres_block.block_size must be an even sublayer count.")
            self.layers_per_block: int = block_size // 2

            self.attn_res = BlockAttnRes(config.d_model)
            self.ffn_res = BlockAttnRes(config.d_model)

        elif config.residual_mode == "full_attnres":
            from .attnres_block import FullAttnRes

            self.full_attn_res = FullAttnRes(config.d_model)
            self.full_ffn_res = FullAttnRes(config.d_model)

        self.residual_mode: str = config.residual_mode

    # ------------------------------------------------------------------
    # Standard / RecurrentResidual forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        m: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for ``standard`` and ``recurrent_residual`` modes.

        Args:
            x: Input hidden state ``(B, S, d_model)``.
            layer_idx: Zero-based layer index used by RR depth bias.
            m: Current memory state for RR mode.

        Returns:
            Tuple of (updated hidden state, updated memory state).
        """
        if self.residual_mode in {"block_attnres", "full_attnres"}:
            raise ValueError(
                "Use the Attention Residual forward path for this residual mode."
            )

        # ── Attention sublayer ────────────────────────────────────────────
        y_attn = self.attn(self.ln_1(x))

        if self.residual_mode == "standard":
            h_mid = x + y_attn
            m_mid = None
        else:  # recurrent_residual
            assert m is not None, "Memory state m is required for RR mode."
            h_mid, m_mid = self.rr_cell(x, y_attn, m, layer_idx, sublayer=0)

        # ── FFN sublayer ──────────────────────────────────────────────────
        y_ffn = self.ffn(self.ln_2(h_mid))

        if self.residual_mode == "standard":
            h_new = h_mid + y_ffn
            m_new = None
        else:
            assert m_mid is not None
            h_new, m_new = self.rr_cell(h_mid, y_ffn, m_mid, layer_idx, sublayer=1)

        return h_new, m_new

    def forward_attnres(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer_idx: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass for sparse block-wise attention residuals.

        Args:
            blocks: Completed block states, including source embeddings.
            partial_block: Current block residual state.
            layer_idx: Zero-based transformer layer index.

        Returns:
            Updated block history and partial block state.
        """
        # Pre-Attn aggregation from previous blocks.
        h_in = self.attn_res(blocks, partial_block)
        y_attn = self.attn(self.ln_1(h_in))
        partial_block = partial_block + y_attn

        # Pre-FFN aggregation from previous blocks.
        h_in = self.ffn_res(blocks, partial_block)
        y_ffn = self.ffn(self.ln_2(h_in))
        partial_block = partial_block + y_ffn

        if (layer_idx + 1) % self.layers_per_block == 0:
            blocks = [*blocks, partial_block]

        return blocks, partial_block

    def forward_full_attnres(
        self,
        history: list[torch.Tensor],
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass for diagnostic full Attention Residuals.

        Args:
            history: Previous hidden states in depth order.
            x: Current hidden state.

        Returns:
            Updated history and hidden state.

        Preconditions:
            ``history`` contains at least the token embedding state.
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
