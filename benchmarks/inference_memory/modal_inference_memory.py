"""Modal entrypoint for Inference Memory and Latency Profiling benchmark."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path

import modal
import torch

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import config_for_mode, load_benchmark_config, make_run_id
from benchmarks.common.inference_latency import profile_forward
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    modal_ignore_patterns,
    write_spawn_manifest,
)

BENCHMARK_NAME = "inference_memory"
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
app = modal.App("rr-inference-memory")


def _prepare_remote() -> None:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_memory_profile(run_id: str, compile: bool = False) -> None:
    """Profile peak activation memory and forward-pass latency across modes and depths."""
    _prepare_remote()
    from src.modules.transformer import TransformerDecoder

    if not torch.cuda.is_available():
        log.warning("CUDA is not available. Script will output zeros.")
        return

    modes = ["standard", "recurrent_residual", "vega", "block_attnres", "hyper_connection"]
    depths = [12, 24, 48, 96]
    seq_len = 100
    vocab_size = 1024

    results: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    base_cfg = load_benchmark_config(BENCHMARK_NAME)

    for mode in modes:
        for L in depths:
            log.info(f"Profiling {mode} at {L} layers...")
            cfg = config_for_mode(base_cfg, mode)
            cfg.model.num_layers = L

            model = TransformerDecoder(cfg.model)
            if compile:
                model = torch.compile(model)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            latency = profile_forward(
                model,
                batch_size=1,
                seq_len=seq_len,
                vocab_size=vocab_size,
                n_warmup=2,
                n_runs=5,
                device="cuda",
            )
            results[mode].append(
                {
                    "layers": L,
                    "peak_vram_mb": latency.peak_vram_mb,
                    "mean_latency_ms": latency.mean_ms,
                    "p50_latency_ms": latency.p50_ms,
                    "p99_latency_ms": latency.p99_ms,
                    "tokens_per_second": latency.tokens_per_second,
                }
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()

    out_dir = Path(ARTIFACT_MOUNT) / "inference_memory" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "profile_results.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Profiling complete. Results saved to {out_file}")
    artifact_volume.commit()


@app.local_entrypoint()
def main(wait: bool = False, compile: bool = False) -> None:
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "profile": run_memory_profile.spawn(run_id, compile=compile),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
