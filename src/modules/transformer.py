"""Decoder-only Transformer with multi-mode residual architectures.

Supported Modes:
*   ``standard``: Classic Pre-LN Transformer.
*   ``recurrent_residual``: Global gated-memory state ('m') shared across depth.
*   ``block_attnres``: Block-wise attention residuals for sparse cross-depth retrieval.
*   ``full_attnres``: Small diagnostic full-depth attention residuals.
"""
from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from .transformer_layer import TransformerLayer


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
              | ``"block_attnres"`` | ``"attnres_block"`` | ``"full_attnres"``.
            - ``attnres_block.block_size`` (int): Required for block_attnres mode.
            - ``recurrent_residual.gate_r_bias`` (float): Optional RR config.
            - ``recurrent_residual.gate_alpha_bias`` (float): Optional RR config.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        # ── Global Recurrent Cell (Shared across layers) ──────────────────
        if self.residual_mode == "recurrent_residual":
            from .recurrent_residual import RecurrentResidualCell

            rr_cfg = getattr(config, "recurrent_residual", {})
            self.rr_cell = RecurrentResidualCell(
                config.d_model,
                config.num_layers,
                gate_r_bias=getattr(rr_cfg, "gate_r_bias", -3.0),
                gate_alpha_bias=getattr(rr_cfg, "gate_alpha_bias", -2.0),
                eps=getattr(rr_cfg, "eps", 1e-5),
            )
        else:
            self.rr_cell = None

        self.layers: nn.ModuleList = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )

        # If shared RR cell is used, inject it into all layers.
        if self.rr_cell is not None:
            for layer in self.layers:
                cast(TransformerLayer, layer).rr_cell = self.rr_cell

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding ↔ LM head output projection.
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights following GPT-2 and DeepNorm conventions.

        - Embeddings: N(0, 0.02).
        - Linear layers: N(0, 0.02), bias → 0.
        - LayerNorm: weight → 1, bias → 0.
        - Standard/AttnRes: Residual projections scaled by 1/√(2·num_layers).
        - Recurrent Residual: Residual projections scaled by DeepNorm beta = (8N)^-0.25.
        """
        num_layers = len(self.layers)
        num_sublayers = num_layers * 2

        # DeepNorm beta constant
        beta = (8.0 * num_sublayers) ** -0.25

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
        if self.residual_mode == "recurrent_residual":
            # DeepNorm initialization
            for layer in self.layers:
                layer_typed = cast(TransformerLayer, layer)
                nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * beta)
                nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * beta)

            # Also scale the shared memory projection
            nn.init.normal_(self.rr_cell.proj_m.weight, mean=0.0, std=0.02 * beta)
        else:
            # Megatron-LM / Standard initialization
            scale = (2.0 * num_layers) ** -0.5
            for layer in self.layers:
                layer_typed = cast(TransformerLayer, layer)
                nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * scale)
                nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * scale)

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
            # Global memory flows across all layers.
            m = self.rr_cell.get_initial_state(B, S, device=device)
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                h, m = layer_typed(h, idx, m)

        elif self.residual_mode in {"attnres_block", "block_attnres"}:
            # Block-0 = token embedding (lets every layer attend to raw input).
            blocks: list[torch.Tensor] = [h]
            partial_block: torch.Tensor = h
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                blocks, partial_block = layer_typed.forward_attnres(
                    blocks, partial_block, idx
                )
            h = partial_block

        elif self.residual_mode == "full_attnres":
            history: list[torch.Tensor] = [h]
            for layer in self.layers:
                layer_typed = cast(TransformerLayer, layer)
                history, h = layer_typed.forward_full_attnres(history, h)

        else:  # standard
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                h, _ = layer_typed(h, idx)

        # ── Head ──────────────────────────────────────────────────────────
        h = self.ln_f(h)
        return self.lm_head(h)
