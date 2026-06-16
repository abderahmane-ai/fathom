"""Lightning runner shared by Modal benchmark scripts."""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
from typing import Any

import lightning as lightning
import torch
import torch.nn.functional as torch_functional
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from benchmarks.common.activation_profile import ActivationMagnitudeTracker
from benchmarks.common.artifacts import (
    checkpoint_dir,
    commit_modal_volume,
    find_resume_checkpoint,
    log_dir,
    metrics_dir,
    resolve_val_check_interval,
    run_dir,
    write_run_metadata,
    write_status,
)
from benchmarks.common.grad_norms import PerLayerGradTracker
from benchmarks.common.jsonl_logger import JsonlLogger
from benchmarks.common.metrics import ThroughputMeter, peak_cuda_memory_mb, write_json
from benchmarks.common.param_count import assert_model_under_cap, count_parameters
from benchmarks.common.run_metadata import (
    capture_run_metadata,
    log_run_banner,
    log_run_finish,
)
from src.data import LanguageModelDataModule
from src.modules import TransformerDecoder

_VEGA_DECAY_VAR_REG_WEIGHT: float = 0.001

log = logging.getLogger(__name__)


class BenchmarkModule(lightning.LightningModule):
    """Lightning module for language-model and synthetic benchmark batches.

    Args:
        model_cfg: Transformer model configuration.
        trainer_cfg: Optimizer and scheduler configuration.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        trainer_cfg: DictConfig,
        benchmark_cfg: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = TransformerDecoder(model_cfg)
        self.trainer_cfg = trainer_cfg
        self.save_hyperparameters(OmegaConf.to_container(model_cfg, resolve=True))
        # Cache VEGA cell reference before any compilation wrapping.
        self._vega_cell = getattr(self.model, "vega_cell", None)
        # Per-layer diagnostics: created here, attached in on_fit_start so any
        # compile / FSDP wrapping that happens between __init__ and training
        # does not orphan the hooks.
        self._grad_tracker: PerLayerGradTracker | None = None
        self._act_tracker: ActivationMagnitudeTracker | None = None
        self._track_diagnostics: bool = bool((benchmark_cfg or {}).get("track_per_layer_metrics", False))

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
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        return torch_functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

    def _check_nan_loss(self, loss: torch.Tensor, step: int) -> None:
        """Dump per-parameter diagnostics to stdout when loss is NaN or Inf.

        Uses ``print()`` rather than the logger so messages appear in Modal
        container logs even if the logging handler is not configured.

        Args:
            loss: The scalar loss tensor to inspect.
            step: Current global step (for log prefix).

        Returns:
            None.
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
        if self._grad_tracker is not None:
            self._grad_tracker.begin_step()
        if self._act_tracker is not None:
            self._act_tracker.begin_step()
        ce_loss = self._loss_from_batch(batch)
        # Diagnostic: dump parameter stats to stdout on NaN/Inf so Modal logs capture it.
        self._check_nan_loss(ce_loss, step=self.global_step)
        loss = ce_loss
        if self._vega_cell is not None:
            alpha = torch.sigmoid(self._vega_cell.decay)
            if alpha.numel() > 1:
                reg = _VEGA_DECAY_VAR_REG_WEIGHT * alpha.var(dim=-1).mean()
                loss = ce_loss - reg
                self.log("train/vega_reg", reg.detach(), on_step=True)
        self.log("train/loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/ppl",
            torch.exp(ce_loss.detach().clamp(max=20.0)),
            on_step=True,
            on_epoch=False,
        )
        self._log_needle_accuracy(batch, "train")
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
        self.log(
            "val/ppl",
            torch.exp(loss.detach().clamp(max=20.0)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self._log_needle_accuracy(batch, "val")

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
            is_bias_or_gain_or_decay = any(keyword in name for keyword in ("bias", "decay", "gain", "scale", "m_init", "damp_weight"))
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
        self._log_per_layer_metrics()
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("grad/global_norm", total_norm, on_step=True, prog_bar=False)
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
        """Log token accuracy on masked (needle) positions when the batch is a
        (tokens, targets) tuple.

        Only logs when targets contain ``-100`` entries (the DeepNeedle ignore index),
        indicating a synthetic depth-needle task.  Standard LM batches are no-ops.
        """
        if not isinstance(batch, (tuple, list)):
            return
        _, targets = batch
        mask = targets != -100
        if not mask.any():
            return
        logits = self.model(batch[0])
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == targets[mask]).float().sum()
        total = mask.float().sum()
        if total > 0:
            self.log(f"{prefix}/needle_acc", correct / total, on_step=True, on_epoch=True, prog_bar=True)

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

    def on_fit_start(self) -> None:
        """Attach per-layer diagnostic trackers when enabled.

        Hooks are installed after the model has been compiled / FSDP-wrapped
        so they survive any later module replacement.
        """
        if not self._track_diagnostics:
            return
        self._grad_tracker = PerLayerGradTracker(self.model)
        self._grad_tracker.attach()
        self._act_tracker = ActivationMagnitudeTracker(self.model)
        self._act_tracker.attach()

    def on_fit_end(self) -> None:
        """Detach per-layer diagnostic trackers and release their hooks."""
        if self._grad_tracker is not None:
            self._grad_tracker.detach()
            self._grad_tracker = None
        if self._act_tracker is not None:
            self._act_tracker.detach()
            self._act_tracker = None

    def _log_per_layer_metrics(self) -> None:
        """Log aggregated per-layer gradient and activation metrics.

        Called from ``on_before_optimizer_step`` so that gradient norms are
        available but clipping has not yet modified them. Activation norms
        are also logged here, even though they were captured during the
        forward pass, for symmetry with the gradient logging cadence.
        """
        if self._grad_tracker is not None and self._grad_tracker.should_log():
            grad_metrics = self._grad_tracker.compute_metrics()
            for name, value in grad_metrics.items():
                self.log(f"grad_layer/{name}", value, on_step=True, prog_bar=False)
        if self._act_tracker is not None and self._act_tracker.should_log():
            act_metrics = self._act_tracker.compute_metrics()
            for name, value in act_metrics.items():
                self.log(f"act_layer/{name}", value, on_step=True, prog_bar=False)


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
    seed = int(cfg.get("seed", 42))
    metadata = capture_run_metadata(
        benchmark_name=benchmark_name,
        run_id=run_id,
        residual_mode=residual_mode,
        config=cfg,
        seed=seed,
        extra={"max_params": max_params, "parameter_count": params},
    )
    write_run_metadata(benchmark_name, residual_mode, run_id, metadata=metadata)
    log_run_banner(log, metadata)
    write_status(
        benchmark_name,
        residual_mode,
        run_id,
        status="starting",
        parameter_count=params,
        max_params=max_params,
    )

    jsonl = JsonlLogger(
        run_dir(benchmark_name, residual_mode, run_id) / "events.jsonl",
        run_id=run_id,
    )
    jsonl.emit("run_start", benchmark_name=benchmark_name, residual_mode=residual_mode, seed=seed, parameter_count=params)

    lightning.seed_everything(seed, workers=True)
    log.info("Seeded with %d", seed)
    torch.set_float32_matmul_precision(str(cfg.trainer.get("matmul_precision", "high")))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    datamodule = build_datamodule(cfg)
    module = BenchmarkModule(cfg.model, cfg.trainer, benchmark_cfg=cfg.benchmark)
    precision = str(cfg.trainer.get("precision", ""))
    use_compile = bool(cfg.get("compile", False))
    if use_compile and "bf16" in precision:
        log.warning("Disabling torch.compile: unstable with bf16-mixed in this stack.")
        use_compile = False
    if use_compile:
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
        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 1)),
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
        status_fields = {k: v for k, v in summary.items() if k not in ("benchmark_name", "residual_mode", "run_id")}
        write_status(benchmark_name, residual_mode, run_id, status="completed", **status_fields)
        write_run_metadata(
            benchmark_name,
            residual_mode,
            run_id,
            ended_at=metadata["started_at"],
            status="completed",
            elapsed_seconds=elapsed,
            global_step=trainer.global_step,
            peak_cuda_memory_mb=peak_cuda_memory_mb(),
        )
        jsonl.emit(
            "run_end",
            status="completed",
            elapsed_seconds=elapsed,
            global_step=trainer.global_step,
            peak_cuda_memory_mb=peak_cuda_memory_mb(),
        )
        log_run_finish(
            log,
            status="completed",
            elapsed_seconds=elapsed,
            global_step=trainer.global_step,
            peak_mem=f"{peak_cuda_memory_mb():.0f} MiB",
            params=f"{count_parameters(module.model):,}",
        )
        commit_modal_volume()
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        tb = traceback.format_exc()
        write_status(
            benchmark_name,
            residual_mode,
            run_id,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=tb,
        )
        write_run_metadata(
            benchmark_name,
            residual_mode,
            run_id,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=tb,
            elapsed_seconds=elapsed,
        )
        jsonl.emit(
            "run_end",
            status="failed",
            elapsed_seconds=elapsed,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        log_run_finish(log, status="failed", elapsed_seconds=elapsed, error=str(exc))
        commit_modal_volume()
        raise
    finally:
        jsonl.close()
