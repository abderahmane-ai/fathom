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

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import config_for_mode, load_benchmark_config, make_run_id
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    write_spawn_manifest,
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


def _run_dps_evaluation(cfg: DictConfig, residual_mode: str, run_id: str) -> None:
    """Run the DPS evaluation logic."""
    from benchmarks.common.dps_extractor import DPSEvaluator
    from benchmarks.common.metrics import calculate_dri, calculate_gpi, compute_dps_closed_form
    from src.data import LanguageModelDataModule
    from src.modules import TransformerDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Initializing {residual_mode} model for DPS probing on {device}")

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
            log.info(f"Evaluating DPS/GPS for layer {k} / {L - 1}")
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

            log.info(f"Layer {k} DPS: {dps:.4f} | GPS: {gps:.4f} (Dissim: {res['mean_dissim']:.4f})")

    dri = calculate_dri(dps_scores)
    gpi = calculate_gpi(gps_scores)
    log.info(f"Final DRI for {residual_mode}: {dri:.4f} | GPI: {gpi:.4f}")

    # Save results to artifact volume
    output_dir = Path(ARTIFACT_MOUNT) / "results" / BENCHMARK_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "residual_mode": residual_mode,
        "run_id": run_id,
        "dps_scores": dps_scores,
        "gps_scores": gps_scores,
        "dri": dri,
        "gpi": gpi,
        "n_tokens": n_target_tokens,
    }

    with open(output_dir / f"{residual_mode}_dps.json", "w") as f:
        json.dump(results, f, indent=2)


def _run_mode(residual_mode: str, run_id: str) -> None:
    """Setup and run a single mode remotely."""
    _prepare_remote()
    cfg = config_for_mode(load_benchmark_config(BENCHMARK_NAME), residual_mode)
    _run_dps_evaluation(cfg, residual_mode, run_id)
    artifact_volume.commit()


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_standard(run_id: str) -> None:
    _run_mode("standard", run_id)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_recurrent_residual(run_id: str) -> None:
    _run_mode("recurrent_residual", run_id)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_block_attnres(run_id: str) -> None:
    _run_mode("block_attnres", run_id)


@app.local_entrypoint()
def main(wait: bool = False) -> None:
    """Spawn all DPS benchmark modes."""
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "standard": run_standard.spawn(run_id),
        "recurrent_residual": run_recurrent_residual.spawn(run_id),
        "block_attnres": run_block_attnres.spawn(run_id),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info(f"Waiting for {mode}")
            handle.get()
