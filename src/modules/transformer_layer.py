from __future__ import annotations
from typing import Any, cast
import torch
import torch.nn as nn

# Import RMSNorm from SWDALRCell module
from .vega import RMSNorm, VEGACell
from .recurrent_residual import RecurrentResidualCell
from .attention import Attention
from .ffn import FeedForward

class TransformerLayer(nn.Module):
    """Universal Transformer layer supporting gated and block residuals."""

    def __init__(
        self,
        config: Any,
        rr_cell: RecurrentResidualCell | None = None,
        vega_cell: VEGACell | None = None,
    ) -> None:
        super().__init__()

        # Use RMSNorm to match modern backbones and the internal cell norms
        self.ln_1 = RMSNorm(config.d_model)
        self.ln_2 = RMSNorm(config.d_model)

        self.attn = Attention(config.d_model, config.n_heads, getattr(config, "dropout", 0.1))
        self.ffn = FeedForward(config.d_model, config.ff_dim, getattr(config, "dropout", 0.1))

        self.rr_cell: nn.Module | None = None
        self.vega_cell: VEGACell | None = None

        if config.residual_mode == "recurrent_residual":
            if rr_cell is None:
                rr_cfg = config.recurrent_residual
                rr_cell = RecurrentResidualCell(
                    config.d_model, config.num_layers,
                    read_gate_bias=rr_cfg.read_gate_bias,
                    forget_gate_bias=getattr(rr_cfg, "forget_gate_bias", 3.0),
                    update_gate_bias=rr_cfg.update_gate_bias,
                    eps=rr_cfg.eps, gate_init_std=rr_cfg.gate_init_std,
                    memory_gain_init=rr_cfg.memory_gain_init,
                )
            self.rr_cell = rr_cell

        elif config.residual_mode == "vega":
            if vega_cell is None:
                vega_cfg = config.vega
                vega_cell = VEGACell(
                    config.d_model, config.num_layers,
                    rank=vega_cfg.rank,
                    n_heads=getattr(vega_cfg, "n_heads", 4),
                    n_fast_heads=getattr(vega_cfg, "n_fast_heads", 2),
                    read_gate_bias=vega_cfg.read_gate_bias,
                    write_gate_bias=getattr(vega_cfg, "write_gate_bias", -2.0),
                    damp_gate_bias=getattr(vega_cfg, "damp_gate_bias", 3.0),
                    eps=vega_cfg.eps, gate_init_std=vega_cfg.gate_init_std,
                )
            self.vega_cell = vega_cell

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

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        m: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass for standard, recurrent_residual, and swda_lr modes."""
        if self.residual_mode in {"block_attnres", "full_attnres"}:
            raise ValueError("Use the specific Attention Residual forward path for this mode.")

        # ── Attention sublayer ────────────────────────────────────────────
        x_norm = self.ln_1(x)
        y_attn = self.attn(x_norm)

        if self.residual_mode == "standard":
            h_mid = x + y_attn
            m_mid = None
        elif self.residual_mode == "recurrent_residual":
            assert m is not None, "Memory state m is required"
            # pyrefly: ignore [not-callable]
            h_mid, m_mid = self.rr_cell(x, y_attn, m, layer_idx, sublayer=0, h_norm=x_norm)
        elif self.residual_mode == "vega":
            assert m is not None and self.vega_cell is not None
            # pyrefly: ignore [not-callable]
            h_mid, m_mid = self.vega_cell(x, y_attn, m, layer_idx, sublayer=0, h_norm=x_norm)

        # ── FFN sublayer ──────────────────────────────────────────────────
        h_norm = self.ln_2(h_mid)
        y_ffn = self.ffn(h_norm)

        if self.residual_mode == "standard":
            h_new = h_mid + y_ffn
            m_new = None
        elif self.residual_mode == "recurrent_residual":
            assert m_mid is not None
            # pyrefly: ignore [not-callable]
            h_new, m_new = self.rr_cell(h_mid, y_ffn, m_mid, layer_idx, sublayer=1, h_norm=h_norm)
        elif self.residual_mode == "vega":
            assert m_mid is not None
            # pyrefly: ignore [not-callable]
            h_new, m_new = self.vega_cell(h_mid, y_ffn, m_mid, layer_idx, sublayer=1, h_norm=h_norm)
            
        return h_new, m_new

    def forward_attnres(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer_idx: int,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor
    ]:
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

        output = (blocks, partial_block)
        for hook in self._forward_hooks.values():
            hook(self, (blocks, partial_block, layer_idx), output)

        return blocks, partial_block

    def forward_full_attnres(
        self,
        history: list[torch.Tensor],
        x: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor
    ]:
        h_in = self.full_attn_res([*history, x])
        y_attn = self.attn(self.ln_1(h_in))
        h_mid = x + y_attn
        history = [*history, h_mid]

        h_in = self.full_ffn_res(history)
        y_ffn = self.ffn(self.ln_2(h_in))
        h_new = h_mid + y_ffn
        history = [*history, h_new]

        output = (history, h_new)
        for hook in self._forward_hooks.values():
            hook(self, (history, x), output)

        return history, h_new
