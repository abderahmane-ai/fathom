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
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

from .recurrent_residual import RecurrentResidualCell
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
              | ``"block_attnres"`` | ``"full_attnres"``.
            - ``attnres_block.block_size`` (int): Required for block_attnres mode.
            - ``recurrent_residual``: Required RR gate initialization config.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        if self.residual_mode not in {
            "standard",
            "recurrent_residual",
            "block_attnres",
            "full_attnres",
        }:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        # ── Global Recurrent Cell (Shared across layers) ──────────────────
        self.rr_cell: RecurrentResidualCell | None
        if self.residual_mode == "recurrent_residual":
            rr_cfg = config.recurrent_residual
            self.rr_cell = RecurrentResidualCell(
                config.d_model,
                config.num_layers,
                read_gate_bias=rr_cfg.read_gate_bias,
                update_gate_bias=rr_cfg.update_gate_bias,
                eps=rr_cfg.eps,
                gate_init_std=rr_cfg.gate_init_std,
                memory_gain_init=rr_cfg.memory_gain_init,
            )
        else:
            self.rr_cell = None

        if self.residual_mode == "full_attnres":
            max_full_layers = int(config.full_attnres.max_layers)
            if config.num_layers > max_full_layers:
                raise ValueError(
                    "full_attnres stores every depth state and is limited to "
                    f"{max_full_layers} layers by config."
                )

        self.layers: nn.ModuleList = nn.ModuleList(
            [TransformerLayer(config, rr_cell=self.rr_cell) for _ in range(config.num_layers)]
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
        """Initialise weights following GPT-style Pre-LN conventions.

        - Embeddings: N(0, 0.02).
        - Linear layers: N(0, 0.02), bias → 0.
        - LayerNorm: weight → 1, bias → 0.
        - Residual projections: scaled by 1/√(2·num_layers).
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

        scale = (2.0 * num_layers) ** -0.5
        for layer in self.layers:
            layer_typed = cast(TransformerLayer, layer)
            nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @jaxtyped(typechecker=beartype)
    def forward(
        self, input_ids: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch seq vocab_size"]:
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

        elif self.residual_mode == "block_attnres":
            # Block-0 = token embedding (lets every layer attend to raw input).
            blocks: list[torch.Tensor] = [h]
            partial_block: torch.Tensor = h
            for idx, layer in enumerate(self.layers):
                layer_typed = cast(TransformerLayer, layer)
                blocks, partial_block = layer_typed.forward_attnres(blocks, partial_block, idx)
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
