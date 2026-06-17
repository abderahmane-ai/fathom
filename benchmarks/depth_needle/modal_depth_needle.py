"""Modal entrypoint for the depth needle benchmark."""

from __future__ import annotations

import logging
import os
import sys

import modal

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import benchmark_modes, config_for_mode, load_benchmark_config, make_run_id
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    print_run_summary,
    write_spawn_manifest,
)

BENCHMARK_NAME = "depth_needle"
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
app = modal.App("rr-depth-needle")


def _prepare_remote() -> None:
    """Prepare imports, working directory, and environment on Modal.

    Returns:
        None.
    """
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run_mode(residual_mode: str, run_id: str, compile: bool = False) -> None:
    """Run a single residual mode remotely.

    Args:
        residual_mode: Residual mode to benchmark.
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _prepare_remote()
    from benchmarks.common.lightning_engine import run_benchmark

    cfg = config_for_mode(load_benchmark_config(BENCHMARK_NAME), residual_mode)
    cfg.compile = compile
    run_benchmark(cfg, BENCHMARK_NAME, residual_mode, run_id)
    artifact_volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_standard(run_id: str, compile: bool = False) -> None:
    """Run the standard residual depth needle benchmark.

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
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_recurrent_residual(run_id: str, compile: bool = False) -> None:
    """Run the Recurrent Residual depth needle benchmark.

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
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_vega(run_id: str, compile: bool = False) -> None:
    """Run the VEGA depth needle benchmark.

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
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_block_attnres(run_id: str, compile: bool = False) -> None:
    """Run the Block AttnRes depth needle benchmark.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("block_attnres", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_full_attnres(run_id: str, compile: bool = False) -> None:
    """Run the tiny Full AttnRes depth needle reference benchmark.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("full_attnres", run_id, compile=compile)


@app.local_entrypoint()
def main(wait: bool = False, include_full: bool = False, compile: bool = False) -> None:
    """Spawn depth needle benchmark modes.

    Args:
        wait: Whether to wait for remote jobs.
        include_full: Whether to include the Full AttnRes reference.
        compile: Whether to compile the models using torch.compile.

    Returns:
        None.
    """
    run_id = make_run_id(BENCHMARK_NAME)
    cfg = load_benchmark_config(BENCHMARK_NAME)
    mode_funcs = {
        "standard": run_standard,
        "recurrent_residual": run_recurrent_residual,
        "vega": run_vega,
        "block_attnres": run_block_attnres,
        "full_attnres": run_full_attnres,
    }
    modes = benchmark_modes(cfg)
    if include_full and "full_attnres" not in modes:
        modes = [*modes, "full_attnres"]
    handles = {
        mode: mode_funcs[mode].spawn(run_id, compile=compile)
        for mode in modes
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
        print_run_summary(log, BENCHMARK_NAME, run_id, list(handles.keys()))
