"""Transformer layer with configurable residual mode.

Architecture
------------
All modes use standard Pre-LN with **two independent sublayers** (Attn + FFN),
each with their own LayerNorm and residual connection:

    Standard / RecurrentResidual
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    y_attn = Attn(LN1(x))
    h_mid  = residual(x,     y_attn, layer_idx, sublayer=0)
    y_ffn  = FFN(LN2(h_mid))
    h_new  = residual(h_mid, y_ffn,  layer_idx, sublayer=1)

    Block-AttnRes
    ~~~~~~~~~~~~~
    h_in   = BlockAttnRes(blocks, partial_block)   # pre-Attn aggregation
    y_attn = Attn(LN1(h_in))
    partial_block += y_attn                        # intra-block accumulation
    h_in   = BlockAttnRes(blocks, partial_block)   # pre-FFN aggregation
    y_ffn  = FFN(LN2(h_in))
    partial_block += y_ffn
    [push partial_block -> blocks at block boundary]

The Block-AttnRes forward signature differs from the other modes (it takes and
returns ``(blocks, partial_block)`` instead of a plain tensor ``x``).
``TransformerDecoder`` calls the appropriate path via ``forward`` for standard /
RR modes and ``forward_attnres`` for Block-AttnRes mode.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .attention import Attention
from .ffn import FeedForward


class TransformerLayer(nn.Module):
    """Single transformer layer with configurable residual connection.

    Args:
        config: Configuration object exposing at minimum:
            - ``d_model`` (int): Hidden dimension.
            - ``n_heads`` (int): Number of attention heads.
            - ``ff_dim``  (int): FFN intermediate dimension.
            - ``dropout`` (float): Dropout probability.
            - ``residual_mode`` (str): ``"standard"`` | ``"recurrent_residual"``
              | ``"attnres_block"``.
            - ``num_layers`` (int): Total layers (needed for RR depth bias).
            - ``attnres_block.block_size`` (int): Sublayers per block (needed
              for attnres_block mode).  Must be even (Attn + FFN = 2 per layer).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        # Pre-Attn and pre-FFN layer norms (separate, as in standard Pre-LN).
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = Attention(config.d_model, config.n_heads,
                               getattr(config, "dropout", 0.1))
        self.ffn = FeedForward(config.d_model, config.ff_dim,
                               getattr(config, "dropout", 0.1))

        self.residual_mode: str = config.residual_mode

        # -- Mode-specific residual modules ----------------------------------
        if config.residual_mode == "attnres_block":
            from .attnres_block import BlockAttnRes

            block_size: int = config.attnres_block.block_size
            # block_size counts *sublayers* (Attn + FFN = 2 per transformer layer)
            assert block_size % 2 == 0, (
                f"attnres_block.block_size must be even (sublayers), got {block_size}."
            )
            self.layers_per_block: int = block_size // 2

            # Two independent projections - one per sublayer, as in the paper.
            self.attn_res = BlockAttnRes(config.d_model)
            self.ffn_res = BlockAttnRes(config.d_model)

    # ------------------------------------------------------------------
    # Standard / RecurrentResidual forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        rr_cell: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Forward pass for ``standard`` and ``recurrent_residual`` modes.

        Args:
            x: Input hidden state ``(B, S, d_model)``.
            layer_idx: Zero-based layer index used by RR depth bias.
            rr_cell: Optional shared ``RecurrentResidualCell``.

        Returns:
            Updated hidden state ``(B, S, d_model)``.

        Raises:
            ValueError: If called in ``attnres_block`` mode (use
                        ``forward_attnres`` instead).
        """
        if self.residual_mode == "attnres_block":
            raise ValueError(
                "Use forward_attnres() for attnres_block mode."
            )

        # -- Attention sublayer --------------------------------------------
        y_attn = self.attn(self.ln_1(x))

        if self.residual_mode == "standard":
            h_mid = x + y_attn
        else:  # recurrent_residual
            assert rr_cell is not None, "rr_cell required for recurrent_residual mode."
            h_mid = rr_cell(x, y_attn, layer_idx, sublayer=0)

        # -- FFN sublayer --------------------------------------------------
        y_ffn = self.ffn(self.ln_2(h_mid))

        if self.residual_mode == "standard":
            h_new = h_mid + y_ffn
        else:
            h_new = rr_cell(h_mid, y_ffn, layer_idx, sublayer=1)

        return h_new

    # ------------------------------------------------------------------
    # Block-AttnRes forward
    # ------------------------------------------------------------------

    def forward_attnres(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer_idx: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass for ``attnres_block`` mode.

        Args:
            blocks: List of completed block tensors ``(B, S, d_model)``.
                    Block-0 is the initial token embedding.  Caller must pass
                    the same list reference through all layers; this method
                    **appends** to it at block boundaries without mutating
                    existing entries.
            partial_block: Running intra-block accumulation ``(B, S, d_model)``.
            layer_idx: Zero-based layer index.

        Returns:
            Tuple ``(blocks, partial_block)`` with the (possibly extended)
            block list and the updated partial accumulation.

        Preconditions:
            * ``len(blocks) >= 1`` (block-0 must be the token embedding).
        """
        # -- Pre-Attn: cross-block aggregation ----------------------------
        h_in = self.attn_res(blocks, partial_block)
        y_attn = self.attn(self.ln_1(h_in))
        partial_block = partial_block + y_attn

        # -- Pre-FFN: cross-block aggregation -----------------------------
        h_in = self.ffn_res(blocks, partial_block)
        y_ffn = self.ffn(self.ln_2(h_in))
        partial_block = partial_block + y_ffn

        # -- Block boundary: push completed block --------------------------
        # A block completes at the last layer of each block group.
        if (layer_idx + 1) % self.layers_per_block == 0:
            # Append without in-place mutation so autograd sees a new list node.
            blocks = [*blocks, partial_block]

        return blocks, partial_block
