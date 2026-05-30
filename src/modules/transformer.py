"""Decoder-only transformer for causal language modelling.

Supports five residual modes selectable via ``config.residual_mode``:
    "standard"           — plain Pre-LN residuals (baseline)
    "recurrent_residual" — RR gated depth memory
    "vega"               — VEGA multi-scale EMA depth memory
    "block_attnres"      — BlockAttnRes softmax over block history
    "full_attnres"       — FullAttnRes softmax over all sublayer states
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .norm import RMSNorm
from .recurrent_residual import RecurrentResidualCell
from .transformer_layer import TransformerLayer
from .vega import VEGACell


class TransformerDecoder(nn.Module):
    """Causal decoder-only transformer with weight-tied LM head.

    The shared depth-memory cells (RR or VEGA) are instantiated once here
    and injected into every layer so all layers share the same parameters.

    Args:
        config: OmegaConf config object (see ``conf/model/``).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        # Shared depth-memory cells — one instance, passed to every layer.
        self.rr_cell:   RecurrentResidualCell | None = None
        self.vega_cell: VEGACell | None = None

        if self.residual_mode == "recurrent_residual":
            rr_cfg = config.recurrent_residual
            self.rr_cell = RecurrentResidualCell(
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

        elif self.residual_mode == "vega":
            vega_cfg = config.vega
            self.vega_cell = VEGACell(
                config.d_model,
                config.num_layers,
                rank=vega_cfg.rank,
                n_heads=getattr(vega_cfg, "n_heads", 4),
                n_fast_heads=getattr(vega_cfg, "n_fast_heads", 2),
                read_gate_bias=vega_cfg.read_gate_bias,
                write_gate_bias=getattr(vega_cfg, "write_gate_bias", -2.0),
                damp_gate_bias=getattr(vega_cfg, "damp_gate_bias", 3.0),
                eps=vega_cfg.eps,
                gate_init_std=vega_cfg.gate_init_std,
            )

        elif self.residual_mode == "full_attnres":
            max_full_layers = int(config.full_attnres.max_layers)
            if config.num_layers > max_full_layers:
                raise ValueError(
                    f"full_attnres is limited to {max_full_layers} layers "
                    f"(requested {config.num_layers})."
                )

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerLayer(config, rr_cell=self.rr_cell, vega_cell=self.vega_cell)
                for _ in range(config.num_layers)
            ]
        )

        self.ln_f    = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and LM-head share the same matrix.
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-style Pre-LN weight initialization.

        Shared cell parameters are excluded to preserve their careful zero-start
        / log-scale initializations.  Output projections (attention proj, FFN w2)
        are scaled by 1/sqrt(2L) following the residual branch scaling convention
        of Wang & Komatsuzaki (2021).
        """
        num_layers = len(self.layers)

        excluded_ids: set[int] = set()
        if self.rr_cell is not None:
            excluded_ids.update(id(m) for m in self.rr_cell.modules())
        if self.vega_cell is not None:
            excluded_ids.update(id(m) for m in self.vega_cell.modules())

        for module in self.modules():
            if id(module) in excluded_ids:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.scale)

        # Residual branch output scaling: std *= 1/sqrt(2L)
        scale = (2.0 * num_layers) ** -0.5
        for layer in self.layers:
            if hasattr(layer.attn, "proj"):
                nn.init.normal_(layer.attn.proj.weight, mean=0.0, std=0.02 * scale)
            if hasattr(layer.ffn, "w2"):
                nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token prediction logits.

        Args:
            input_ids: Token indices, shape (B, S).

        Returns:
            Logits of shape (B, S, vocab_size).
        """
        B, S = input_ids.shape
        device = input_ids.device

        h = self.emb_drop(self.token_embeddings(input_ids))

        if self.residual_mode in {"recurrent_residual", "vega"}:
            cell = self.rr_cell if self.residual_mode == "recurrent_residual" else self.vega_cell
            assert cell is not None
            m = cell.get_initial_state(B, S, device=device)
            for idx, layer in enumerate(self.layers):
                h, m = layer(h, idx, m)

        elif self.residual_mode == "block_attnres":
            blocks: list[torch.Tensor] = [h]
            partial_block: torch.Tensor = h
            for idx, layer in enumerate(self.layers):
                blocks, partial_block = layer.forward_attnres(blocks, partial_block, idx)
            h = partial_block

        elif self.residual_mode == "full_attnres":
            history: list[torch.Tensor] = [h]
            for layer in self.layers:
                history, h = layer.forward_full_attnres(history, h)

        else:  # standard
            for idx, layer in enumerate(self.layers):
                h, _ = layer(h, idx)

        h = self.ln_f(h)
        return self.lm_head(h)
