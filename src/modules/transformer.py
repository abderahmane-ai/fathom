"""Decoder-only Transformer with multi-mode residual architectures.

Supported Modes:
*   ``standard``: Classic Pre-LN Transformer.
*   ``recurrent_residual``: Global gated-memory state ('m') shared across depth.
*   ``attnres_block``: Block-wise attention residuals for sparse cross-depth retrieval.
"""
from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn

from .transformer_layer import TransformerLayer


class TransformerDecoder(nn.Module):
    """Causal language model decoder with configurable residual logic.

    Args:
        config: Configuration exposing d_model, n_heads, ff_dim, num_layers,
                max_seq_len, vocab_size, and residual_mode.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        self.layers: nn.ModuleList = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )

        # Shared Recurrent Residual cell (Depth-RNN orchestrator).
        self.rr_cell: Optional[nn.Module] = None
        if self.residual_mode == "recurrent_residual":
            from .recurrent_residual import RecurrentResidualCell
            self.rr_cell = RecurrentResidualCell(
                d_model=config.d_model,
                num_layers=config.num_layers,
                gate_r_bias=getattr(config, "gate_r_bias", -3.0),
                gate_alpha_bias=getattr(config, "gate_alpha_bias", -2.0),
            )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (GPT-2 convention).
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise parameters using DeepNorm and GPT-2 conventions."""
        num_layers = len(self.layers)
        num_sublayers = num_layers * 2
        
        # DeepNorm beta constant scales residual projections to ensure variance stability.
        beta = (8.0 * num_sublayers) ** -0.25

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if self.residual_mode == "recurrent_residual":
            assert self.rr_cell is not None
            # DeepNorm initialization for residual branches.
            for layer in self.layers:
                layer_typed = cast(TransformerLayer, layer)
                nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * beta)
                nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * beta)
                
            if hasattr(self.rr_cell, "proj_m"):
                nn.init.normal_(self.rr_cell.proj_m.weight, mean=0.0, std=0.02 * beta)
        else:
            # Megatron-LM style initialization scaling.
            scale = (2.0 * num_layers) ** -0.5
            for layer in self.layers:
                layer_typed = cast(TransformerLayer, layer)
                nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * scale)
                nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for next-token prediction."""
        B, S = input_ids.shape
        device = input_ids.device

        # Token + Positional embeddings.
        pos_ids = torch.arange(S, device=device)
        h = self.emb_drop(
            self.token_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        )

        # Process through layer stack based on residual mode.
        if self.residual_mode == "recurrent_residual":
            assert self.rr_cell is not None
            self.rr_cell.reset_memory(B, S, device=device)
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                h = layer_typed(h, idx, rr_cell=self.rr_cell)

        elif self.residual_mode == "attnres_block":
            blocks: list[torch.Tensor] = [h]
            partial_block: torch.Tensor = h
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                blocks, partial_block = layer_typed.forward_attnres(
                    blocks, partial_block, idx
                )
            h = partial_block

        else:
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                h = layer_typed(h, idx)

        h = self.ln_f(h)
        return self.lm_head(h)