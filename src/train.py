"""Training entry point for Recurrent Residuals experiments.

Orchestrates model, data, and trainer via Hydra config + PyTorch Lightning.

Usage
-----
    # Standard residual baseline
    python src/train.py

    # Block-AttnRes mode
    python src/train.py model=attnres

    # Recurrent Residual mode, custom LR
    python src/train.py model=recurrent_residual trainer.optimizer.lr=1e-4

    # Multi-GPU (2 GPUs, DDP)
    python src/train.py trainer.devices=2 trainer.strategy=ddp
"""
from __future__ import annotations

import math
import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    GradientAccumulationScheduler,
)

from src.modules import TransformerDecoder
from src.data import LanguageModelDataModule

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class LanguageModel(L.LightningModule):
    """LightningModule wrapping TransformerDecoder for causal LM training.

    Implements:
    * Cross-entropy LM loss (next-token prediction).
    * AdamW optimiser with cosine LR schedule + linear warmup.
    * Per-step logging of loss, perplexity, LR, and gradient norm.
    * WandB logging of per-layer output magnitudes every N steps.

    Args:
        model_cfg: Model configuration (``conf/model/*.yaml``).
        trainer_cfg: Trainer configuration (``conf/trainer/default.yaml``).
    """

    def __init__(self, model_cfg: DictConfig, trainer_cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(model_cfg, resolve=True))
        self.trainer_cfg = trainer_cfg
        self.model = TransformerDecoder(model_cfg)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model parameters: %s M", f"{n_params / 1e6:.1f}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute next-token cross-entropy loss.

        Args:
            batch: Integer tensor ``(B, S)`` of token IDs.

        Returns:
            Scalar cross-entropy loss averaged over valid positions.
        """
        input_ids = batch[:, :-1].contiguous()   # (B, S-1)
        labels = batch[:, 1:].contiguous()        # (B, S-1)  — shifted targets
        logits = self.model(input_ids)             # (B, S-1, vocab_size)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Single training step.

        Args:
            batch: Token ID tensor ``(B, S)``.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss.detach())
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/ppl", ppl, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Single validation step.

        Args:
            batch: Token ID tensor ``(B, S)``.
            batch_idx: Index of the current batch.
        """
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ppl", ppl, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Single test step.

        Args:
            batch: Token ID tensor ``(B, S)``.
            batch_idx: Index of the current batch.
        """
        loss = self._compute_loss(batch)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/ppl", torch.exp(loss), on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW + cosine LR schedule with linear warmup.

        Returns:
            Lightning optimizer/scheduler dict.
        """
        opt_cfg = self.trainer_cfg.optimizer
        sch_cfg = self.trainer_cfg.scheduler

        # Weight decay only on 2-D+ tensors (skip bias, LN, embeddings).
        decay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]
        param_groups = [
            {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            fused=torch.cuda.is_available(),   # fused kernel when on CUDA
        )

        # Cosine decay with linear warmup.
        total_steps: int = self.trainer.estimated_stepping_batches
        warmup_steps: int = sch_cfg.warmup_steps
        min_lr_ratio: float = sch_cfg.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train a transformer with the given Hydra config.

    Args:
        cfg: Composed configuration from ``conf/config.yaml`` and overrides.
    """
    L.seed_everything(cfg.seed, workers=True)

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Model ────────────────────────────────────────────────────────────
    model = LanguageModel(cfg.model, cfg.trainer)

    # ── Data ─────────────────────────────────────────────────────────────
    datamodule = LanguageModelDataModule(cfg.data)

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{step}-{val/loss:.4f}",
        ),
    ]

    # ── Logger ───────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="recurrent-residuals",
        name=f"{cfg.model.residual_mode}_d{cfg.model.d_model}_L{cfg.model.num_layers}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
