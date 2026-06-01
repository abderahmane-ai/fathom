"""Modal entrypoint for IsoFLOP Depth vs. Width tradeoff benchmark."""

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

BENCHMARK_NAME = "iso_flop"
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
app = modal.App("rr-iso-flop")


def _prepare_remote() -> None:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ.setdefault("HF_HOME", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 10,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_wide_shallow(run_id: str) -> None:
    """Run Wide & Shallow Standard (6 Layers, d_model=1024)."""
    _prepare_remote()
    from benchmarks.common.lightning_engine import run_benchmark

    cfg = load_benchmark_config("scaling_efficiency")  # Base it on scaling efficiency config
    cfg = config_for_mode(cfg, "standard")

    # Override for Wide & Shallow
    cfg.model.num_layers = 6
    cfg.model.d_model = 1024
    cfg.model.ff_dim = 1024 * 4

    run_benchmark(cfg, BENCHMARK_NAME, "standard_wide", run_id)
    artifact_volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 10,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_narrow_deep(run_id: str, mode: str = "vega") -> None:
    """Run Narrow & Deep VEGA/RR (24 Layers, d_model=512)."""
    _prepare_remote()
    from benchmarks.common.lightning_engine import run_benchmark

    cfg = load_benchmark_config(BENCHMARK_NAME)
    cfg = config_for_mode(cfg, mode)

    # Override for Narrow & Deep
    cfg.model.num_layers = 24
    cfg.model.d_model = 512
    cfg.model.ff_dim = 512 * 4

    run_benchmark(cfg, BENCHMARK_NAME, f"{mode}_narrow", run_id)
    artifact_volume.commit()


@app.local_entrypoint()
def main(wait: bool = False) -> None:
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "wide_shallow_std": run_wide_shallow.spawn(run_id),
        "narrow_deep_vega": run_narrow_deep.spawn(run_id, mode="vega"),
        "narrow_deep_rr": run_narrow_deep.spawn(run_id, mode="recurrent_residual"),
        "narrow_deep_hc": run_narrow_deep.spawn(run_id, mode="hyper_connection"),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
