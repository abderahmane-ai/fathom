from __future__ import annotations
from typing import Any, cast
import torch
import torch.nn as nn

from .swda_lr import RMSNorm, SWDALRCell
from .recurrent_residual import RecurrentResidualCell
from .transformer_layer import TransformerLayer

class TransformerDecoder(nn.Module):
    """Decoder-only transformer for causal language modelling."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.residual_mode: str = config.residual_mode

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Only use absolute positional embeddings if NOT using RoPE in Attention
        # If your Attention module uses RoPE, set use_absolute_pe to False
        self.use_absolute_pe = getattr(config, "use_absolute_pe", False)
        if self.use_absolute_pe:
            self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
            
        self.emb_drop = nn.Dropout(getattr(config, "dropout", 0.1))

        if self.residual_mode not in {"standard", "recurrent_residual", "swda_lr", "block_attnres", "full_attnres"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        # ── Global Shared Cells (Weight-Tied Depth Routing) ───────────────
        self.rr_cell: RecurrentResidualCell | None = None
        self.swda_lr_cell: SWDALRCell | None = None
        
        if self.residual_mode == "recurrent_residual":
            rr_cfg = config.recurrent_residual
            self.rr_cell = RecurrentResidualCell(
                config.d_model, config.num_layers,
                read_gate_bias=rr_cfg.read_gate_bias,
                forget_gate_bias=getattr(rr_cfg, "forget_gate_bias", 3.0),
                update_gate_bias=rr_cfg.update_gate_bias,
                eps=rr_cfg.eps, gate_init_std=rr_cfg.gate_init_std,
                memory_gain_init=rr_cfg.memory_gain_init,
            )
        elif self.residual_mode == "swda_lr":
            swda_cfg = config.swda_lr
            self.swda_lr_cell = SWDALRCell(
                config.d_model, config.num_layers,
                window_size=swda_cfg.window_size, rank=swda_cfg.rank,
                n_heads=getattr(swda_cfg, "n_heads", 4), v_dim=getattr(swda_cfg, "v_dim", None),
                decay_bias_init=swda_cfg.decay_bias_init, read_gate_bias=swda_cfg.read_gate_bias,
                write_gate_bias=getattr(swda_cfg, "write_gate_bias", -2.0),
                eps=swda_cfg.eps, gate_init_std=swda_cfg.gate_init_std,
            )

        if self.residual_mode == "full_attnres":
            max_full_layers = int(config.full_attnres.max_layers)
            if config.num_layers > max_full_layers:
                raise ValueError(f"full_attnres is limited to {max_full_layers} layers.")

        # Pass the globally shared cells to all layers
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerLayer(config, rr_cell=self.rr_cell, swda_lr_cell=self.swda_lr_cell)
                for _ in range(config.num_layers)
            ]
        )

        # Use RMSNorm for the final pre-head normalization
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following GPT-style Pre-LN conventions."""
        num_layers = len(self.layers)

        # Exclude the shared cells so we don't overwrite their delicate zero-start/log-scale inits
        excluded_ids: set[int] = set()
        if self.rr_cell is not None:
            excluded_ids.update(id(m) for m in self.rr_cell.modules())
        if self.swda_lr_cell is not None:
            excluded_ids.update(id(m) for m in self.swda_lr_cell.modules())

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
                # Handle parameter-free vs parameterized RMSNorm
                if hasattr(module, 'scale'):
                    nn.init.ones_(module.scale)

        # Pre-LN residual scaling (Wortsman et al., 2021)
        scale = (2.0 * num_layers) ** -0.5
        for layer in self.layers:
            layer_typed = cast(TransformerLayer, layer)
            # Ensure your Attention and FFN classes expose these exact attribute names
            if hasattr(layer_typed.attn, 'proj'):
                nn.init.normal_(layer_typed.attn.proj.weight, mean=0.0, std=0.02 * scale)
            if hasattr(layer_typed.ffn, 'w2'):
                nn.init.normal_(layer_typed.ffn.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token prediction logits."""
        B, S = input_ids.shape
        device = input_ids.device

        h = self.token_embeddings(input_ids)
        
        if self.use_absolute_pe:
            pos_ids = torch.arange(S, device=device)
            h = h + self.pos_embeddings(pos_ids)
            
        h = self.emb_drop(h)

        # ── Layer stack ───────────────────────────────────────────────────
        if self.residual_mode in {"recurrent_residual", "swda_lr"}:
            cell = self.rr_cell if self.residual_mode == "recurrent_residual" else self.swda_lr_cell
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
