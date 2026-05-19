"""Lightning benchmark runner with checkpointing, resume, and artifact sync."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from benchmarks.artifacts import (
    checkpoint_dir,
    commit_modal_volume,
    find_resume_checkpoint,
    log_dir,
    run_dir,
    write_status,
)
from benchmarks.needle_task import evaluate_needle
from src.data import LanguageModelDataModule
from src.modules import TransformerDecoder

log = logging.getLogger(__name__)


class VolumeSyncCallback(Callback):
    """Commit Modal volume after checkpoints and periodic steps."""

    def __init__(self, residual_mode: str, every_n_steps: int = 500) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.every_n_steps = every_n_steps

    def _sync(self, trainer: L.Trainer, reason: str) -> None:
        write_status(
            self.residual_mode,
            status="running",
            global_step=trainer.global_step,
            last_sync_reason=reason,
        )
        commit_modal_volume()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            self._sync(trainer, reason=f"step_{step}")


class BenchmarkCallback(Callback):
    """Needle eval, throughput, and RR gate diagnostics."""

    def __init__(self, residual_mode: str, needle_every_n_steps: int = 200) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.needle_every_n_steps = needle_every_n_steps
        self.start_time: Optional[float] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.start_time = time.time()
        write_status(
            self.residual_mode,
            status="running",
            global_step=0,
            max_steps=trainer.max_steps,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step <= 0 or step % self.needle_every_n_steps != 0:
            return

        module = pl_module.model
        assert isinstance(module, TransformerDecoder)

        max_seq_len = module.pos_embeddings.num_embeddings
        acc = evaluate_needle(module, pl_module.device, seq_len=max_seq_len)
        pl_module.log("needle/acc", acc, on_step=True, prog_bar=True)

        if self.start_time is not None:
            elapsed = max(time.time() - self.start_time, 1e-6)
            dm = trainer.datamodule
            batch_size = dm.cfg.batch_size if dm is not None else 1
            seq_len = dm.cfg.max_seq_len if dm is not None else max_seq_len
            throughput = (step * batch_size * seq_len) / elapsed
            pl_module.log("perf/throughput_tokens_per_sec", throughput, on_step=True)

        with torch.no_grad():
            sample_batch = batch[:1, :-1]
            logits = module(sample_batch)
            norm = torch.norm(logits, p=2, dim=-1).mean()
            pl_module.log("norm/final_logits", norm, on_step=True)

        if (
            module.residual_mode == "recurrent_residual"
            and module.rr_cell is not None
            and hasattr(module.rr_cell, "last_alpha")
        ):
            pl_module.log("gate/mean_alpha", module.rr_cell.last_alpha.item(), on_step=True)

        write_status(
            self.residual_mode,
            status="running",
            global_step=step,
            needle_acc=float(acc),
        )



class BenchmarkingLM(L.LightningModule):
    """Causal LM wrapper for benchmark runs."""

    def __init__(self, model_cfg: DictConfig, trainer_cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerDecoder(model_cfg)
        self.trainer_cfg = trainer_cfg

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if (
            self.model.residual_mode == "recurrent_residual"
            and self.model.rr_cell is not None
            and hasattr(self.model.rr_cell, "last_alpha")
        ):
            self.log("gate/mean_alpha", self.model.rr_cell.last_alpha.detach(), on_step=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        from torch.optim import AdamW

        return AdamW(
            self.parameters(),
            lr=self.trainer_cfg.optimizer.lr,
            weight_decay=self.trainer_cfg.optimizer.weight_decay,
        )


def run(cfg: DictConfig) -> None:
    """Run one benchmark job with checkpointing and optional resume.

    Args:
        cfg: Config with ``model``, ``trainer``, ``data``, ``benchmark`` sections.

    Environment:
        ``BENCHMARK_ARTIFACT_ROOT``: Root for logs/checkpoints/status.
        ``BENCHMARK_RESUME``: ``1`` to auto-resume from latest checkpoint.
        ``BENCHMARK_COMPILE``: ``1`` to enable ``torch.compile``.
        ``BENCHMARK_VOLUME_NAME``: Modal volume name for ``commit()``.
    """
    mode: str = cfg.model.residual_mode
    log.info("Starting benchmark residual_mode=%s", mode)

    write_status(
        mode,
        status="starting",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config_summary={
            "num_layers": cfg.model.num_layers,
            "d_model": cfg.model.d_model,
            "max_steps": cfg.benchmark.max_steps,
        },
    )
    commit_modal_volume()

    try:
        L.seed_everything(int(cfg.seed), workers=True)
        torch.set_float32_matmul_precision("high")

        pl_module = BenchmarkingLM(cfg.model, cfg.trainer)
        datamodule = LanguageModelDataModule(cfg.data)

        if os.environ.get("BENCHMARK_COMPILE", "0") == "1" and hasattr(torch, "compile"):
            try:
                pl_module.model = torch.compile(pl_module.model, mode="reduce-overhead")
                log.info("torch.compile enabled")
            except Exception:
                log.exception("torch.compile failed; using eager mode")

        ckpt_path = checkpoint_dir(mode)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_path),
            filename="{step:06d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=int(cfg.benchmark.checkpoint_every_n_steps),
            enable_version_counter=False,
        )

        callbacks: list[Callback] = [
            checkpoint_cb,
            BenchmarkCallback(mode, needle_every_n_steps=int(cfg.benchmark.needle_freq)),
            VolumeSyncCallback(mode, every_n_steps=int(cfg.benchmark.volume_commit_every_n_steps)),
            LearningRateMonitor(logging_interval="step"),
        ]

        csv_root = log_dir(mode)
        logger = CSVLogger(save_dir=str(csv_root.parent), name=csv_root.name)

        resume_path: Optional[str] = None
        if os.environ.get("BENCHMARK_RESUME", "1") == "1":
            resume_path = find_resume_checkpoint(mode)
            if resume_path:
                log.info("Resuming from checkpoint: %s", resume_path)
                write_status(mode, status="resuming", resume_checkpoint=resume_path)

        trainer = L.Trainer(
            max_steps=int(cfg.benchmark.max_steps),
            precision=cfg.trainer.precision,
            logger=logger,
            callbacks=callbacks,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=int(cfg.benchmark.log_every_n_steps),
            val_check_interval=int(cfg.benchmark.val_check_interval),
            gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 1.0)),
            default_root_dir=str(run_dir(mode)),
        )

        trainer.fit(pl_module, datamodule=datamodule, ckpt_path=resume_path)

        write_status(
            mode,
            status="completed",
            global_step=trainer.global_step,
            best_checkpoint=checkpoint_cb.best_model_path,
            last_checkpoint=checkpoint_cb.last_model_path,
        )
        commit_modal_volume()
        log.info("Benchmark completed: %s", mode)

    except Exception as exc:
        write_status(
            mode,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        commit_modal_volume()
        log.exception("Benchmark failed: %s", mode)
        raise
