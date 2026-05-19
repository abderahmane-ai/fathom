import logging
import os
import time

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
from benchmarks.needle_task import evaluate_needle
from src.modules import TransformerDecoder
from src.data import LanguageModelDataModule

log = logging.getLogger(__name__)

class BenchmarkCallback(Callback):
    def __init__(self, needle_every_n_steps: int = 200):
        super().__init__()
        self.needle_every_n_steps = needle_every_n_steps
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.needle_every_n_steps == 0:
            max_seq_len = pl_module.model.pos_embeddings.num_embeddings
            acc = evaluate_needle(pl_module.model, pl_module.device, seq_len=max_seq_len)
            pl_module.log("needle/acc", acc, on_step=True, prog_bar=True)
            
            # Log throughput
            elapsed = time.time() - self.start_time
            throughput = (step * trainer.datamodule.cfg.batch_size * trainer.datamodule.cfg.max_seq_len) / elapsed
            pl_module.log("perf/throughput_tokens_per_sec", throughput, on_step=True)
            
            # Hidden-state norm stability
            # We need to hook into the model to get the last hidden state
            # For simplicity, we can do a forward pass here or use a hook
            # Let's just do a small batch forward pass
            with torch.no_grad():
                sample_batch = batch[:1, :-1]
                # We need a way to get hidden states. 
                # Let's assume TransformerDecoder can return them or we just use the final output norm
                logits = pl_module.model(sample_batch)
                norm = torch.norm(logits, p=2, dim=-1).mean()
                pl_module.log("norm/final_logits", norm, on_step=True)

            # Gate specialization (RR only; requires last forward write in RR cell)
            if (
                pl_module.model.residual_mode == "recurrent_residual"
                and hasattr(pl_module.model, "rr_cell")
                and hasattr(pl_module.model.rr_cell, "last_alpha")
            ):
                pl_module.log(
                    "gate/mean_alpha",
                    pl_module.model.rr_cell.last_alpha.item(),
                    on_step=True,
                )


class BenchmarkingLM(L.LightningModule):
    def __init__(self, model_cfg, trainer_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerDecoder(model_cfg)
        self.trainer_cfg = trainer_cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        from torch.optim import AdamW
        optimizer = AdamW(self.parameters(), lr=self.trainer_cfg.optimizer.lr, weight_decay=self.trainer_cfg.optimizer.weight_decay)
        return optimizer

def run(cfg) -> None:
    """Run a single benchmark training job.

    Args:
        cfg: Full benchmark config (model, trainer, data, benchmark sections).

    Raises:
        Exception: Re-raises any failure after logging a full traceback.
    """
    mode = cfg.model.residual_mode
    log.info("Starting benchmark for residual_mode=%s", mode)

    L.seed_everything(cfg.seed)

    # Set matmul precision for A100 Tensor Cores
    torch.set_float32_matmul_precision("high")

    model = BenchmarkingLM(cfg.model, cfg.trainer)
    datamodule = LanguageModelDataModule(cfg.data)

    if hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            log.info("torch.compile enabled (mode=reduce-overhead)")
        except Exception:
            log.exception("torch.compile failed; continuing with eager mode")

    log_root = os.environ.get("BENCHMARK_LOG_DIR", "logs")
    logger = CSVLogger(
        save_dir=log_root,
        name=f"{mode}_L{cfg.model.num_layers}",
    )
    log.info("CSV logs -> %s", os.path.join(log_root, logger.name))

    trainer = L.Trainer(
        max_steps=cfg.benchmark.max_steps,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=[BenchmarkCallback(needle_every_n_steps=cfg.benchmark.needle_freq)],
        accelerator="gpu",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    log.info("Benchmark finished for residual_mode=%s", mode)
