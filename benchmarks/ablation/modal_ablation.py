"""Modal entrypoint for the Ablation suite (Targeted Component Removal)."""

from __future__ import annotations

import logging
import os
import sys

import modal

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import config_for_mode, load_benchmark_config, make_run_id
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    write_spawn_manifest,
)

BENCHMARK_NAME = "ablation"
log = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/root/rr"})
    .uv_pip_install(
        "torch>=2.2.0",
        "lightning>=2.2.0",
        "transformers>=4.39.0",
        "datasets>=2.18.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "wandb>=0.17.0",
        "jaxtyping>=0.2.28",
        "beartype>=0.17.2",
        "pyarrow>=15.0.0",
    )
    .add_local_dir(str(repo_root()), remote_path=REMOTE_ROOT, ignore=modal_ignore_patterns())
)
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("rr-ablation")


def _prepare_remote() -> None:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ.setdefault("HF_HOME", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume})
def run_vega_no_var_reg(run_id: str) -> None:
    """VEGA without Variance Regularization on decay."""
    _prepare_remote()

    # Monkey-patch variance regularization to 0
    import benchmarks.common.lightning_engine as engine

    engine._VEGA_DECAY_VAR_REG_WEIGHT = 0.0

    cfg = load_benchmark_config(BENCHMARK_NAME)
    cfg = config_for_mode(cfg, "vega")

    engine.run_benchmark(cfg, BENCHMARK_NAME, "vega_no_var_reg", run_id)
    artifact_volume.commit()


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume})
def run_vega_no_multiscale(run_id: str) -> None:
    """VEGA initialized uniformly (no multi-scale log-linear init)."""
    _prepare_remote()
    from benchmarks.common.lightning_engine import run_benchmark

    cfg = load_benchmark_config(BENCHMARK_NAME)
    cfg = config_for_mode(cfg, "vega")

    # Override decay ranges so all heads start identically
    cfg.model.vega.fast_decay_range = [2.0, 2.0]
    cfg.model.vega.slow_decay_range = [2.0, 2.0]

    run_benchmark(cfg, BENCHMARK_NAME, "vega_no_multiscale", run_id)
    artifact_volume.commit()


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume})
def run_rr_no_depth_biases(run_id: str) -> None:
    """RR without layer-specific depth biases."""
    _prepare_remote()
    import lightning as L

    from benchmarks.common.lightning_engine import BenchmarkModule, run_benchmark

    # Create a callback to zero out the biases before training
    class ZeroDepthBiasesCallback(L.Callback):
        def on_fit_start(self, trainer: L.Trainer, pl_module: BenchmarkModule) -> None:
            # We locate the recurrent residual cell inside the model
            model = pl_module.model
            if hasattr(model, "rr_cell") and model.rr_cell is not None:
                for attr in [
                    "depth_read_bias",
                    "depth_forget_bias",
                    "depth_update_bias",
                    "depth_damp_bias",
                ]:
                    if hasattr(model.rr_cell, attr):
                        param = getattr(model.rr_cell, attr)
                        param.data.zero_()
                        param.requires_grad = False
                log.info("Zeroed out and froze RR depth biases.")

    cfg = load_benchmark_config(BENCHMARK_NAME)
    cfg = config_for_mode(cfg, "recurrent_residual")

    # To use the custom callback, we would normally pass it to Trainer.
    # Since run_benchmark handles Trainer instantiation, we'll monkey-patch pl_module
    # on_fit_start directly as a hack to preserve zero-intrusion to src.

    original_on_fit_start = BenchmarkModule.on_fit_start

    def modified_on_fit_start(self: BenchmarkModule) -> None:
        original_on_fit_start(self)
        model = self.model
        if hasattr(model, "rr_cell") and model.rr_cell is not None:
            for attr in [
                "depth_read_bias",
                "depth_forget_bias",
                "depth_update_bias",
                "depth_damp_bias",
            ]:
                if hasattr(model.rr_cell, attr):
                    param = getattr(model.rr_cell, attr)
                    param.data.zero_()
                    param.requires_grad = False
            log.info("Monkey-patched RR depth biases to zero.")

    BenchmarkModule.on_fit_start = modified_on_fit_start

    run_benchmark(cfg, BENCHMARK_NAME, "rr_no_depth_biases", run_id)
    artifact_volume.commit()


@app.local_entrypoint()
def main(wait: bool = False) -> None:
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "vega_no_var_reg": run_vega_no_var_reg.spawn(run_id),
        "vega_no_multiscale": run_vega_no_multiscale.spawn(run_id),
        "rr_no_depth_biases": run_rr_no_depth_biases.spawn(run_id),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
