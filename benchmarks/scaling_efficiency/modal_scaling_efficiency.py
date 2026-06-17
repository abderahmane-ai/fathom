"""Modal entrypoint for the scaling efficiency benchmark."""

from __future__ import annotations

import logging
import os
import sys

import modal

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import (
    config_for_mode,
    load_benchmark_config,
    make_run_id,
    model_sweep,
)
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    print_run_summary,
    write_spawn_manifest,
)

BENCHMARK_NAME = "scaling_efficiency"
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
app = modal.App("rr-scaling-efficiency")


def _prepare_remote() -> None:
    """Prepare imports, working directory, and environment on Modal.

    Returns:
        None.
    """
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ.setdefault("HF_HOME", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache/datasets")
    os.environ.setdefault("RR_PACKED_CACHE_DIR", f"{ARTIFACT_MOUNT}/packed_cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run_mode(residual_mode: str, run_id: str, compile: bool = False) -> None:
    """Run all sweep points for one residual mode remotely.

    Args:
        residual_mode: Residual mode to benchmark.
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _prepare_remote()
    from benchmarks.common.lightning_engine import run_benchmark

    base_cfg = load_benchmark_config(BENCHMARK_NAME)
    for model_cfg in model_sweep(base_cfg):
        cfg = config_for_mode(base_cfg, residual_mode, model_cfg=model_cfg)
        cfg.compile = compile
        sweep_run_id = f"{run_id}-d{cfg.model.d_model}-l{cfg.model.num_layers}"
        run_benchmark(cfg, BENCHMARK_NAME, residual_mode, sweep_run_id)
    artifact_volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_standard(run_id: str, compile: bool = False) -> None:
    """Run standard residual scaling sweep.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("standard", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_recurrent_residual(run_id: str, compile: bool = False) -> None:
    """Run Recurrent Residual scaling sweep.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("recurrent_residual", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_vega(run_id: str, compile: bool = False) -> None:
    """Run VEGA scaling sweep.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("vega", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_block_attnres(run_id: str, compile: bool = False) -> None:
    """Run Block AttnRes scaling sweep.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("block_attnres", run_id, compile=compile)


@app.local_entrypoint()
def main(wait: bool = False, compile: bool = False) -> None:
    """Spawn scaling efficiency benchmark modes.

    Args:
        wait: Whether to wait for remote jobs.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "standard": run_standard.spawn(run_id, compile=compile),
        "recurrent_residual": run_recurrent_residual.spawn(run_id, compile=compile),
        "vega": run_vega.spawn(run_id, compile=compile),
        "block_attnres": run_block_attnres.spawn(run_id, compile=compile),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
        print_run_summary(log, BENCHMARK_NAME, run_id, list(handles.keys()))
