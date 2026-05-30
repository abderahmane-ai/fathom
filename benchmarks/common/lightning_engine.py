"""Lightning runner shared by Modal benchmark scripts."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import lightning as lightning
import torch
import torch.nn.functional as torch_functional
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from benchmarks.common.artifacts import (
    checkpoint_dir,
    commit_modal_volume,
    find_resume_checkpoint,
    log_dir,
    metrics_dir,
    resolve_val_check_interval,
    run_dir,
    write_status,
)
from benchmarks.common.metrics import ThroughputMeter, peak_cuda_memory_mb, write_json
from benchmarks.common.param_count import assert_model_under_cap, count_parameters
from src.data import LanguageModelDataModule
from src.modules import TransformerDecoder

log = logging.getLogger(__name__)


class BenchmarkModule(lightning.LightningModule):
    """Lightning module for language-model and synthetic benchmark batches.

    Args:
        model_cfg: Transformer model configuration.
        trainer_cfg: Optimizer and scheduler configuration.
    """

    def __init__(self, model_cfg: DictConfig, trainer_cfg: DictConfig) -> None:
        super().__init__()
        self.model = TransformerDecoder(model_cfg)
        self.trainer_cfg = trainer_cfg
        self.save_hyperparameters(OmegaConf.to_container(model_cfg, resolve=True))

    def _loss_from_batch(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-entropy loss for a benchmark batch.

        Args:
            batch: Either packed LM tokens or ``(tokens, targets)`` for needle tasks.

        Returns:
            Scalar cross-entropy loss.
        """
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            logits = self.model(input_ids)
            return torch_functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        return torch_functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

    def training_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Benchmark batch.
            _batch_idx: Unused Lightning batch index.

        Returns:
            Training loss.
        """
        loss = self._loss_from_batch(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", torch.exp(loss.detach()), on_step=True, on_epoch=False)
        self._log_rr_gates()
        return loss

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> None:
        """Run one validation step.

        Args:
            batch: Benchmark batch.
            _batch_idx: Unused Lightning batch index.

        Returns:
            None.
        """
        loss = self._loss_from_batch(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss.detach()), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Build AdamW with cosine warmup scheduling.

        Returns:
            Lightning optimizer and scheduler configuration.
        """
        opt_cfg = self.trainer_cfg.optimizer
        sch_cfg = self.trainer_cfg.scheduler

        # Partition parameters to avoid applying weight decay to biases,
        # normalization scales, decay rates, gains, damp scales, or initial states.
        decay_params = []
        no_decay_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_bias_or_gain_or_decay = any(
                keyword in name
                for keyword in ("bias", "decay", "gain", "scale", "m_init", "damp_weight")
            )
            if p.dim() < 2 or is_bias_or_gain_or_decay:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": float(opt_cfg.weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=float(opt_cfg.lr),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            fused=False,
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(sch_cfg.warmup_steps)
        min_lr_ratio = float(sch_cfg.min_lr_ratio)

        def lr_lambda(step: int) -> float:
            """Return the LR multiplier for one optimizer step."""
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

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log global gradient norm before the optimizer step.

        Args:
            optimizer: Optimizer about to step.

        Returns:
            None.
        """
        squared_norm = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
        self.log("grad/global_norm", squared_norm.sqrt(), on_step=True, prog_bar=False)

    def _log_rr_gates(self) -> None:
        """Log RR diagnostics when the model exposes them.

        Returns:
            None.
        """
        rr_cell = getattr(self.model, "rr_cell", None)
        if rr_cell is None or not hasattr(rr_cell, "last_update_gate"):
            return
        self.log("rr/read_gate_mean", rr_cell.last_read_gate.detach(), on_step=True)
        self.log("rr/update_gate_mean", rr_cell.last_update_gate.detach(), on_step=True)


class StatusCallback(Callback):
    """Write status and throughput snapshots during training.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.
        tokens_per_step: Tokens processed per optimizer step.
        every_n_steps: Status write interval.
    """

    def __init__(
        self,
        benchmark_name: str,
        residual_mode: str,
        run_id: str,
        tokens_per_step: int,
        every_n_steps: int,
    ) -> None:
        super().__init__()
        self.benchmark_name = benchmark_name
        self.residual_mode = residual_mode
        self.run_id = run_id
        self.every_n_steps = every_n_steps
        self.throughput = ThroughputMeter(tokens_per_step)

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Persist periodic run status.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Training step outputs.
            batch: Batch that just completed.
            batch_idx: Batch index.

        Returns:
            None.
        """
        step = trainer.global_step
        if step <= 0 or step % self.every_n_steps != 0:
            return
        tokens_per_second = self.throughput.tokens_per_second(step)
        pl_module.log("perf/tokens_per_second", tokens_per_second, on_step=True)
        write_status(
            self.benchmark_name,
            self.residual_mode,
            self.run_id,
            status="running",
            global_step=step,
            tokens_per_second=tokens_per_second,
            peak_cuda_memory_mb=peak_cuda_memory_mb(),
        )
        commit_modal_volume()


def _build_logger(cfg: DictConfig, benchmark_name: str, residual_mode: str, run_id: str) -> Any:
    """Create a WandB logger when available, otherwise CSV.

    Args:
        cfg: Benchmark config.
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Lightning logger instance.
    """
    logs = log_dir(benchmark_name, residual_mode, run_id)
    wandb_cfg = cfg.get("wandb", {})
    if bool(wandb_cfg.get("enabled", False)) and os.environ.get("WANDB_API_KEY"):
        try:
            from lightning.pytorch.loggers import WandbLogger

            return WandbLogger(
                project=str(wandb_cfg.get("project", "recurrent-residuals")),
                name=f"{benchmark_name}-{residual_mode}-{run_id}",
                save_dir=str(logs),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception:
            log.exception("Falling back to CSVLogger after WandB setup failed")

    return CSVLogger(save_dir=str(logs.parent), name=logs.name)


def build_datamodule(cfg: DictConfig) -> lightning.LightningDataModule:
    """Create the data module requested by a benchmark config.

    Args:
        cfg: Benchmark config.

    Returns:
        Lightning data module.
    """
    task = str(cfg.benchmark.get("task", "lm"))
    if task == "depth_needle":
        from src.data import DeepNeedleDataModule

        return DeepNeedleDataModule(
            seq_len=int(cfg.data.max_seq_len),
            vocab_size=int(cfg.data.vocab_size),
            start_token=int(cfg.data.start_token),
            blank_token=int(cfg.data.blank_token),
            output_token=int(cfg.data.output_token),
            batch_size=int(cfg.data.batch_size),
            n_eval=int(cfg.data.get("n_eval", 1000)),
            num_workers=int(cfg.data.num_workers),
        )
    return LanguageModelDataModule(cfg.data)


def run_benchmark(cfg: DictConfig, benchmark_name: str, residual_mode: str, run_id: str) -> None:
    """Run one benchmark job.

    Args:
        cfg: Benchmark config already specialized to one residual mode.
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        None.
    """
    max_params = int(cfg.benchmark.get("max_params", 60_000_000))
    params = assert_model_under_cap(cfg.model, max_params)
    write_status(
        benchmark_name,
        residual_mode,
        run_id,
        status="starting",
        parameter_count=params,
        max_params=max_params,
    )

    lightning.seed_everything(int(cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision(str(cfg.trainer.get("matmul_precision", "high")))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    datamodule = build_datamodule(cfg)
    module = BenchmarkModule(cfg.model, cfg.trainer)
    if bool(cfg.get("compile", False)):
        log.info("Compiling model using torch.compile...")
        module.model = torch.compile(module.model)
    batch_size = int(cfg.data.batch_size)
    seq_len = int(cfg.data.max_seq_len)
    tokens_per_step = batch_size * seq_len

    ckpts = checkpoint_dir(benchmark_name, residual_mode, run_id)
    ckpts.mkdir(parents=True, exist_ok=True)
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=str(ckpts),
            filename="{step:06d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=int(cfg.benchmark.get("save_top_k", 2)),
            save_last=True,
            every_n_train_steps=int(cfg.benchmark.get("checkpoint_every_n_steps", 500)),
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        StatusCallback(
            benchmark_name,
            residual_mode,
            run_id,
            tokens_per_step=tokens_per_step,
            every_n_steps=int(cfg.benchmark.get("status_every_n_steps", 100)),
        ),
    ]

    logger = _build_logger(cfg, benchmark_name, residual_mode, run_id)
    trainer = lightning.Trainer(
        accelerator=str(cfg.trainer.get("accelerator", "auto")),
        devices=cfg.trainer.get("devices", "auto"),
        strategy=cfg.trainer.get("strategy", "auto"),
        max_steps=int(cfg.benchmark.max_steps),
        max_epochs=int(cfg.trainer.get("max_epochs", -1)),
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.benchmark.get("log_every_n_steps", 50)),
        val_check_interval=resolve_val_check_interval(
            int(cfg.benchmark.get("estimated_train_batches", 1000)),
            cfg.benchmark.get("val_check_interval", 500),
        ),
        gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 1.0)),
        default_root_dir=str(run_dir(benchmark_name, residual_mode, run_id)),
    )

    resume_path = None
    if bool(cfg.benchmark.get("resume", True)):
        resume_path = find_resume_checkpoint(benchmark_name, residual_mode, run_id)

    started_at = time.perf_counter()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)
        elapsed = time.perf_counter() - started_at
        summary = {
            "benchmark_name": benchmark_name,
            "residual_mode": residual_mode,
            "run_id": run_id,
            "parameter_count": count_parameters(module.model),
            "elapsed_seconds": elapsed,
            "global_step": trainer.global_step,
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
        }
        write_json(metrics_dir(benchmark_name, residual_mode, run_id) / "summary.json", summary)
        status_fields = {
            k: v
            for k, v in summary.items()
            if k not in ("benchmark_name", "residual_mode", "run_id")
        }
        write_status(benchmark_name, residual_mode, run_id, status="completed", **status_fields)
        commit_modal_volume()
    except Exception as exc:
        write_status(
            benchmark_name,
            residual_mode,
            run_id,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        commit_modal_volume()
        raise
