"""Training entry point for Recurrent Residual (RR) experiments.

Orchestrates the model, data pipeline, and training loop using Hydra and
PyTorch Lightning. Supports experiment switching via Hydra overrides.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data import LanguageModelDataModule
from src.modules import TransformerDecoder

_VEGA_DECAY_VAR_REG_WEIGHT: float = 0.001

log = logging.getLogger(__name__)


def _tensor_stats(t: torch.Tensor, name: str) -> str:
    """Return a compact diagnostic string for a tensor."""
    tf = t.detach().float()
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={tf.min().item():.4g} max={tf.max().item():.4g} "
        f"mean={tf.mean().item():.4g} "
        f"nan={tf.isnan().any().item()} inf={tf.isinf().any().item()}"
    )


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
        # Cache VEGA cell reference before any compilation wrapping.
        self._vega_cell = getattr(self.model, "vega_cell", None)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Initialized model with %s M parameters", f"{n_params / 1e6:.1f}")

    def _compute_loss(self, batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Computes next-token cross-entropy loss on shifted or pre-split inputs."""
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            logits = self.model(input_ids)
            return F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1),
        )

    def _check_nan_loss(self, loss: torch.Tensor, step: int) -> None:
        """Dump per-parameter diagnostics to stdout when loss is NaN or Inf.

        Uses ``print()`` rather than the logger so messages appear in container logs.
        """
        if not (torch.isnan(loss) or torch.isinf(loss)):
            return
        tag = "NaN" if torch.isnan(loss) else "Inf"
        print(f"\n[DIAG step={step}] *** {tag} LOSS DETECTED (value={loss.item()}) ***")
        print(f"[DIAG step={step}] Scanning all named parameters for bad values...")
        bad_params: list[str] = []
        for name, param in self.named_parameters():
            if param is None:
                continue
            pf = param.detach().float()
            has_nan = pf.isnan().any().item()
            has_inf = pf.isinf().any().item()
            stats = (
                f"  param  {name}: "
                f"shape={tuple(param.shape)} dtype={param.dtype} "
                f"min={pf.min().item():.4g} max={pf.max().item():.4g} "
                f"mean={pf.mean().item():.4g} nan={has_nan} inf={has_inf}"
            )
            if has_nan or has_inf:
                bad_params.append(name)
                print(f"[DIAG step={step}] *** {stats}")
            else:
                print(f"[DIAG step={step}] {stats}")
        if bad_params:
            print(f"[DIAG step={step}] BAD parameters: {bad_params}")
        else:
            print(f"[DIAG step={step}] All parameters look clean — NaN may originate in activations/ops.")
        print(f"[DIAG step={step}] *** end of parameter dump ***\n")

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        ce_loss = self._compute_loss(batch)
        self._check_nan_loss(ce_loss, step=self.global_step)
        loss = ce_loss
        if self._vega_cell is not None:
            alpha = torch.sigmoid(self._vega_cell.decay)
            if alpha.numel() > 1:
                reg = _VEGA_DECAY_VAR_REG_WEIGHT * alpha.var(dim=-1).mean()
                loss = ce_loss - reg
                self.log("train/vega_reg", reg.detach(), on_step=True)
        self.log("train/loss", ce_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/ppl", torch.exp(ce_loss.detach().clamp(max=20.0)), on_step=True, on_epoch=False)
        self._log_needle_accuracy(batch, "train")
        return loss

    def validation_step(self, batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)
        self._log_needle_accuracy(batch, "val")

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/ppl", torch.exp(loss), on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log global gradient norm before each optimizer step.

        Args:
            optimizer: Optimizer about to step.

        Returns:
            None.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("grad/global_norm", total_norm, on_step=True)
        # Diagnostic: warn in stdout if any gradient is already NaN/Inf before clipping.
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            print(f"\n[DIAG step={self.global_step}] *** NaN/Inf GRADIENT NORM ({total_norm.item()}) ***")
            for name, param in self.named_parameters():
                if param.grad is None:
                    continue
                gf = param.grad.detach().float()
                if gf.isnan().any() or gf.isinf().any():
                    print(
                        f"[DIAG step={self.global_step}]   BAD grad  {name}: "
                        f"min={gf.min().item():.4g} max={gf.max().item():.4g} "
                        f"nan={gf.isnan().any().item()} inf={gf.isinf().any().item()}"
                    )
            print(f"[DIAG step={self.global_step}] *** end of gradient dump ***\n")

    def _log_needle_accuracy(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        prefix: str,
    ) -> None:
        """Log token accuracy on masked (needle) positions for DeepNeedle batches.

        No-op for standard LM batches (plain tensors).
        """
        if not isinstance(batch, (tuple, list)):
            return
        _, targets = batch
        mask = targets != -100
        if not mask.any():
            return
        with torch.no_grad():
            logits = self.model(batch[0])
            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == targets[mask]).float().sum()
            total = mask.float().sum()
        if total > 0:
            self.log(f"{prefix}/needle_acc", correct / total, on_step=True, on_epoch=True, prog_bar=True)

    # pyrefly: ignore [bad-override]
    def configure_optimizers(self) -> dict[str, Any]:
        """Configures AdamW with weight decay filtering and cosine decay."""
        opt_cfg = self.trainer_cfg.optimizer
        sch_cfg = self.trainer_cfg.scheduler

        # Weight decay only on 2D+ weights (kernels), excluding biases, gains, decays, and norms.
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_bias_or_gain_or_decay = any(keyword in name for keyword in ("bias", "decay", "gain", "scale", "m_init", "damp_weight"))
            if p.dim() < 2 or is_bias_or_gain_or_decay:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            fused=False,
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


def build_logger(cfg: DictConfig) -> Any:
    """Build the configured Lightning logger.

    Args:
        cfg: Root Hydra config.

    Returns:
        WandB logger when configured and available, otherwise CSV logger.
    """
    logger_cfg = cfg.trainer.logger
    run_name = f"{cfg.model.residual_mode}_d{cfg.model.d_model}_L{cfg.model.num_layers}"
    if logger_cfg.name == "wandb":
        try:
            from lightning.pytorch.loggers import WandbLogger

            return WandbLogger(
                project=logger_cfg.project,
                name=run_name,
                save_dir=logger_cfg.save_dir,
                offline=bool(logger_cfg.offline),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception:
            log.exception("WandB logger unavailable; falling back to CSVLogger.")

    return CSVLogger(save_dir=logger_cfg.save_dir, name=run_name)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """System entry point for training and evaluation."""
    L.seed_everything(cfg.seed, workers=True)
    log.info("Training Configuration:\n%s", OmegaConf.to_yaml(cfg))

    model = LanguageModel(cfg.model, cfg.trainer)
    precision = str(cfg.trainer.get("precision", ""))
    use_compile = bool(cfg.get("compile", False))
    if use_compile and "bf16" in precision:
        log.warning("Disabling torch.compile: unstable with bf16-mixed in this stack.")
        use_compile = False
    if use_compile:
        log.info("Compiling model using torch.compile...")
        model.model = torch.compile(model.model)
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

    logger = build_logger(cfg)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        logger=logger,
        # pyrefly: ignore [bad-argument-type]
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
