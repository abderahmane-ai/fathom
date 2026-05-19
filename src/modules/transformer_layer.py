"""Transformer Layer orchestrator with multi-mode residual support.

Supported Architectures:
1.  **Standard / RecurrentResidual**:
    Standard Pre-LN flow with two sublayers (Attn, FFN), each followed by
    either a simple addition or a gated RR cell transition.
2.  **Block-AttnRes**:
    Implements cross-block aggregation where sublayers attend to previous
    block states stored in a shared list.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .attention import Attention
from .ffn import FeedForward


class TransformerLayer(nn.Module):
    """Universal Transformer layer supporting gated and block residuals.

    Args:
        config: Configuration exposing d_model, n_heads, ff_dim, and residual_mode.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        # Pre-sublayer normalisation layers.
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = Attention(config.d_model, config.n_heads,
                               getattr(config, "dropout", 0.1))
        self.ffn = FeedForward(config.d_model, config.ff_dim,
                               getattr(config, "dropout", 0.1))

        # ── Mode-specific residual modules ──────────────────────────────────
        self.rr_cell: Optional[nn.Module] = None

        if config.residual_mode == "recurrent_residual":
            from .recurrent_residual import RecurrentResidualCell
            # Defaults to per-layer cell; TransformerDecoder can override this
            # with a shared instance to enable global memory flow.
            self.rr_cell = RecurrentResidualCell(config.d_model, config.num_layers)

        elif config.residual_mode == "attnres_block":
            from .attnres_block import BlockAttnRes
            block_size: int = config.attnres_block.block_size
            assert block_size % 2 == 0, "attnres_block.block_size must be even."
            self.layers_per_block: int = block_size // 2

            # Cross-block projection heads.
            self.attn_res = BlockAttnRes(config.d_model)
            self.ffn_res = BlockAttnRes(config.d_model)

        self.residual_mode: str = config.residual_mode

    # ------------------------------------------------------------------
    # Standard / RecurrentResidual forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        m: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for ``standard`` and ``recurrent_residual`` modes.

        Args:
            x: Input hidden state ``(B, S, d_model)``.
            layer_idx: Zero-based layer index used by RR depth bias.
            m: Current memory state for RR mode.

        Returns:
            Tuple of (updated hidden state, updated memory state).
        """
        if self.residual_mode == "attnres_block":
            raise ValueError(
                "Use forward_attnres() for attnres_block mode."
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
        """Forward pass for sparse block-wise attention residuals."""
        # Pre-Attn aggregation from previous blocks.
        h_in = self.attn_res(blocks, partial_block)
        y_attn = self.attn(self.ln_1(h_in))
        partial_block = partial_block + y_attn

        # Pre-FFN aggregation from previous blocks.
        h_in = self.ffn_res(blocks, partial_block)
        y_ffn = self.ffn(self.ln_2(h_in))
        partial_block = partial_block + y_ffn

        # Handle block completion and state storage.
        if (layer_idx + 1) % self.layers_per_block == 0:
            blocks = [*blocks, partial_block]

        return blocks, partial_block

