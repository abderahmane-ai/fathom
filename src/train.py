"""Training entry point for Recurrent Residual (RR) experiments.

Orchestrates the model, data pipeline, and training loop using Hydra and
PyTorch Lightning. Supports experiment switching via Hydra overrides.
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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.modules import TransformerDecoder
from src.data import LanguageModelDataModule

log = logging.getLogger(__name__)


class LanguageModel(L.LightningModule):
    """LightningModule wrapping TransformerDecoder for causal LM training.

    Handles cross-entropy loss, AdamW optimization with cosine LR scheduling,
    and performance logging (perplexity, gradient norms, etc.).
    """

    def __init__(self, model_cfg: DictConfig, trainer_cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(model_cfg, resolve=True))
        self.trainer_cfg = trainer_cfg
        self.model = TransformerDecoder(model_cfg)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Initialized model with %s M parameters", f"{n_params / 1e6:.1f}")

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes next-token cross-entropy loss on shifted inputs."""
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/ppl", torch.exp(loss.detach()), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/ppl", torch.exp(loss), on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures AdamW with weight decay filtering and cosine decay."""
        opt_cfg = self.trainer_cfg.optimizer
        sch_cfg = self.trainer_cfg.scheduler

        # Weight decay only on 2D+ tensors (kernels/weights), excluding biases and norms.
        decay_params = [p for n, p in self.model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters() if p.requires_grad and p.dim() < 2]
        
        param_groups = [
            {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            fused=torch.cuda.is_available(),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = sch_cfg.warmup_steps
        min_lr_ratio = sch_cfg.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """System entry point for training and evaluation."""
    L.seed_everything(cfg.seed, workers=True)
    log.info("Training Configuration:\n%s", OmegaConf.to_yaml(cfg))

    model = LanguageModel(cfg.model, cfg.trainer)
    datamodule = LanguageModelDataModule(cfg.data)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{step}-{val/loss:.4f}",
        ),
    ]

    wandb_logger = WandbLogger(
        project="recurrent-residuals",
        name=f"{cfg.model.residual_mode}_d{cfg.model.d_model}_L{cfg.model.num_layers}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

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

