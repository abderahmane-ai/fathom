"""Modal entrypoint for the Depth Preservation Score (DPS) benchmark."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import modal
import torch
from omegaconf import DictConfig

from benchmarks.common.artifacts import (
    benchmark_dir,
    repo_root,
    run_dir,
    write_run_metadata,
    write_status,
)
from benchmarks.common.configs import config_for_mode, load_benchmark_config, make_run_id
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    default_retries,
    print_run_summary,
    write_spawn_manifest,
)
from benchmarks.common.run_metadata import (
    WallClock,
    capture_run_metadata,
    log_run_banner,
    log_run_finish,
)

BENCHMARK_NAME = "depth_preservation"
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
app = modal.App("rr-depth-preservation")


def _prepare_remote() -> None:
    """Prepare imports, working directory, and environment on Modal."""
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ.setdefault("HF_HOME", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache/datasets")
    os.environ.setdefault("RR_PACKED_CACHE_DIR", f"{ARTIFACT_MOUNT}/packed_cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run_dps_evaluation(cfg: DictConfig, residual_mode: str, run_id: str) -> dict:
    """Run the DPS evaluation logic.

    Returns:
        A dict of final results (dri, gpi, n_tokens, dps_scores, gps_scores).
    """
    from benchmarks.common.dps_extractor import DPSEvaluator
    from benchmarks.common.metrics import calculate_dri, calculate_gpi, compute_dps_closed_form
    from src.data import LanguageModelDataModule
    from src.modules import TransformerDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Initializing %s model for DPS probing on %s", residual_mode, device)

    # Initialize Model (Untrained baseline probing)
    model = TransformerDecoder(cfg.model).to(device)
    model.eval()

    # Initialize Data
    datamodule = LanguageModelDataModule(cfg.data)
    datamodule.setup(stage="fit")  # Prepare datasets
    dataloader = datamodule.val_dataloader()

    L = cfg.model.num_layers
    target_layers = list(range(1, L))  # Layers 1 to L-1
    n_target_tokens = int(cfg.benchmark.get("n_tokens", 100000))
    lambda_val = float(cfg.benchmark.get("lambda_val", 1.0))
    final_norm_name = str(cfg.benchmark.get("final_norm_name", "norm"))

    dps_scores = []
    gps_scores = []

    with torch.no_grad():
        for k in target_layers:
            log.info("Evaluating DPS/GPS for layer %d / %d", k, L - 1)
            evaluator = DPSEvaluator(model, layer_idx=k, final_norm_name=final_norm_name)

            tokens_processed = 0
            for batch in dataloader:
                if tokens_processed >= n_target_tokens:
                    break

                # Causal next-token shift: inputs are tokens 0 to S-2, targets are tokens 1 to S-1
                input_ids = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)

                # Forward pass
                _ = model(input_ids)

                # Accumulate statistics
                evaluator.process_batch(targets=targets)

                tokens_processed += input_ids.numel()

            res = evaluator.get_results()
            evaluator.remove_hooks()

            # 1. Compute DPS (Depth Preservation Score)
            dps = compute_dps_closed_form(
                xtx=res["xtx"],
                xty=res["xty"],
                yty=res["yty"],
                target_variance=res["target_variance"],
                lambda_val=lambda_val,
            )
            dps_scores.append(dps)

            # 2. Compute GPS (Gradient Preservation Score)
            gps = 0.0
            if "xty_gps" in res:
                gps = compute_dps_closed_form(
                    xtx=res["xtx"],
                    xty=res["xty_gps"],
                    yty=res["yty_gps"],
                    target_variance=res["target_variance_gps"],
                    lambda_val=lambda_val,
                )
            gps_scores.append(gps)

            log.info(
                "Layer %d DPS: %.4f | GPS: %.4f (Dissim: %.4f)",
                k, dps, gps, res["mean_dissim"],
            )

    dri = calculate_dri(dps_scores)
    gpi = calculate_gpi(gps_scores)
    log.info("Final DRI for %s: %.4f | GPI: %.4f", residual_mode, dri, gpi)

    return {
        "residual_mode": residual_mode,
        "run_id": run_id,
        "dps_scores": dps_scores,
        "gps_scores": gps_scores,
        "dri": dri,
        "gpi": gpi,
        "n_tokens": n_target_tokens,
    }


def _run_mode(residual_mode: str, run_id: str, compile: bool = False) -> None:
    """Setup and run a single mode remotely.

    Args:
        residual_mode: Residual mode to benchmark.
        run_id: Shared run id.
        compile: Whether to compile the model.
    """
    import traceback

    from benchmarks.common.param_count import count_parameters

    _prepare_remote()
    cfg = config_for_mode(load_benchmark_config(BENCHMARK_NAME), residual_mode)
    cfg.compile = compile

    metadata = capture_run_metadata(
        benchmark_name=BENCHMARK_NAME,
        run_id=run_id,
        residual_mode=residual_mode,
        config=cfg,
    )
    write_run_metadata(BENCHMARK_NAME, residual_mode, run_id, metadata=metadata)
    log_run_banner(log, metadata)
    write_status(BENCHMARK_NAME, residual_mode, run_id, status="starting")

    clock = WallClock()
    try:
        results = _run_dps_evaluation(cfg, residual_mode, run_id)
        elapsed = clock.elapsed()

        # Save results to artifact volume (legacy path retained for backward compat)
        output_dir = Path(ARTIFACT_MOUNT) / "results" / BENCHMARK_NAME / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{residual_mode}_dps.json", "w") as f:
            json.dump(results, f, indent=2)

        # Also write to canonical artifacts/ tree so status + find_all_runs can find it
        canonical_dir = run_dir(BENCHMARK_NAME, residual_mode, run_id)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        with open(canonical_dir / "dps.json", "w") as f:
            json.dump(results, f, indent=2)

        write_status(
            BENCHMARK_NAME,
            residual_mode,
            run_id,
            status="completed",
            dri=results["dri"],
            gpi=results["gpi"],
            n_tokens=results["n_tokens"],
            n_layers=len(results["dps_scores"]),
            elapsed_seconds=elapsed,
        )
        log_run_finish(
            log,
            status="completed",
            elapsed_seconds=elapsed,
            dri=f"{results['dri']:.4f}",
            gpi=f"{results['gpi']:.4f}",
            n_tokens=results["n_tokens"],
        )
        artifact_volume.commit()
    except Exception as exc:
        elapsed = clock.elapsed()
        tb = traceback.format_exc()
        write_status(
            BENCHMARK_NAME,
            residual_mode,
            run_id,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=tb,
            elapsed_seconds=elapsed,
        )
        log_run_finish(log, status="failed", elapsed_seconds=elapsed, error=str(exc))
        artifact_volume.commit()
        raise


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=default_retries(),
)
def run_standard(run_id: str, compile: bool = False) -> None:
    """Run the standard residual depth preservation benchmark.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.
    """
    _run_mode("standard", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=default_retries(),
)
def run_recurrent_residual(run_id: str, compile: bool = False) -> None:
    """Run the Recurrent Residual depth preservation benchmark.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.
    """
    _run_mode("recurrent_residual", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=default_retries(),
)
def run_vega(run_id: str, compile: bool = False) -> None:
    """Run the VEGA depth preservation benchmark.

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
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=default_retries(),
)
def run_block_attnres(run_id: str, compile: bool = False) -> None:
    _run_mode("block_attnres", run_id, compile=compile)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=default_retries(),
)
def run_hyper_connection(run_id: str, compile: bool = False) -> None:
    """Run the mHC-Lite hyper-connection depth preservation benchmark.

    Args:
        run_id: Shared run id.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_mode("hyper_connection", run_id, compile=compile)


@app.local_entrypoint()
def main(wait: bool = False, compile: bool = False) -> None:
    """Spawn all DPS benchmark modes.

    Args:
        wait: Whether to wait for remote jobs.
        compile: Whether to compile the models using torch.compile.

    Returns:
        None.
    """
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "standard": run_standard.spawn(run_id, compile=compile),
        "recurrent_residual": run_recurrent_residual.spawn(run_id, compile=compile),
        "vega": run_vega.spawn(run_id, compile=compile),
        "block_attnres": run_block_attnres.spawn(run_id, compile=compile),
        "hyper_connection": run_hyper_connection.spawn(run_id, compile=compile),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
        print_run_summary(log, BENCHMARK_NAME, run_id, list(handles.keys()))
