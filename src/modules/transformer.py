"""Decoder-only transformer with three residual modes.

Modes
-----
``standard``
    Classic Pre-LN transformer.  Each layer: ``h = h + Attn(LN1(h)) + FFN(LN2(...))``.

``recurrent_residual``
    Gated-memory residual via ``RecurrentResidualCell``.  Memory is reset at
    the start of every forward pass (one sequence / batch at a time).

``attnres_block``
    Block Attention Residuals (arXiv:2603.15031).  A ``blocks`` list and a
    ``partial_block`` tensor are threaded through all layers.  The token
    embedding serves as block-0 so every layer can attend back to raw input.

Usage
-----
    model = TransformerDecoder(cfg)
    logits = model(input_ids)          # (B, S, vocab_size)
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.modules.transformer_layer import TransformerLayer


class TransformerDecoder(nn.Module):
    """Decoder-only transformer for causal language modelling.

    Args:
        config: Configuration object exposing:
            - ``d_model``      (int):  Hidden dimension.
            - ``n_heads``      (int):  Attention heads.
            - ``ff_dim``       (int):  FFN intermediate width.
            - ``num_layers``   (int):  Number of transformer layers.
            - ``max_seq_len``  (int):  Maximum sequence length.
            - ``vocab_size``   (int):  Vocabulary size.
            - ``dropout``      (float): Embedding / output dropout.
            - ``residual_mode`` (str): ``"standard"`` | ``"recurrent_residual"``
              | ``"attnres_block"``.
            - ``attnres_block.block_size`` (int): Required for attnres_block mode.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding ↔ LM head output projection.
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights following GPT-2 conventions.

        - Embeddings: N(0, 0.02).
        - Linear layers: N(0, 0.02), bias → 0.
        - LayerNorm: weight → 1, bias → 0.
        - Residual projections scaled by 1/√(2·num_layers) to keep output
          variance stable at depth (Megatron-LM convention).
        """
        num_layers = len(self.layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual projection weights (attn.proj, ffn.w2).
        scale = (2.0 * num_layers) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token prediction logits.

        Args:
            input_ids: Token ID tensor of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.

        Preconditions:
            * ``S <= config.max_seq_len``.
            * ``input_ids`` values in ``[0, vocab_size)``.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # ── Embeddings ────────────────────────────────────────────────────
        pos_ids = torch.arange(S, device=device)
        h = self.emb_drop(
            self.token_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        )  # (B, S, d)

        # ── Layer stack ───────────────────────────────────────────────────
        if self.residual_mode == "recurrent_residual":
            # Reset per-layer memory to zeros for this batch.
            for layer in self.layers:
                layer.rr_cell.reset_memory(B, S, device=device)
            for idx, layer in enumerate(self.layers):
                h = layer(h, idx)

        elif self.residual_mode == "attnres_block":
            # Block-0 = token embedding (lets every layer attend to raw input).
            blocks: list[torch.Tensor] = [h]
            partial_block: torch.Tensor = h
            for idx, layer in enumerate(self.layers):
                blocks, partial_block = layer.forward_attnres(
                    blocks, partial_block, idx
                )
            h = partial_block

        else:  # standard
            for idx, layer in enumerate(self.layers):
                h = layer(h, idx)

        # ── Head ──────────────────────────────────────────────────────────
        h = self.ln_f(h)
        return self.lm_head(h)