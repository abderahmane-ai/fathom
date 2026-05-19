"""Modal entrypoint: parallel RR vs AttnRes on 2x A100 with persistent artifacts."""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

log = logging.getLogger(__name__)

VOLUME_NAME = "rr-benchmark-artifacts"
ARTIFACT_MOUNT = "/artifacts"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "lightning>=2.2.0",
        "transformers>=4.39.0",
        "datasets>=2.18.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
    )
    .add_local_dir(".", remote_path="/root/rr")
)

artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("rr-benchmarks")


def _build_cfg(residual_mode: str):
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "seed": 42,
        "model": {
            "residual_mode": residual_mode,
            "d_model": 512,
            "n_heads": 8,
            "ff_dim": 2048,
            "num_layers": 6,
            "max_seq_len": 128,
            "vocab_size": 50257,
            "dropout": 0.1,
            "recurrent_residual": {
                "gate_r_bias": -3.0,
                "gate_alpha_bias": -2.0,
                "eps": 1e-5,
            },
            "attnres_block": {"block_size": 2},
        },
        "trainer": {
            "precision": "bf16-mixed",
            "gradient_clip_val": 1.0,
            "optimizer": {"lr": 1e-3, "weight_decay": 0.1},
        },
        "data": {
            "dataset_name": "roneneldan/TinyStories",
            "dataset_config": "default",
            "train_split": "train[:1%]",
            "val_split": "validation[:1%]",
            "test_split": "validation[:1%]",
            "tokenizer_name": "gpt2",
            "max_seq_len": 128,
            "batch_size": 128,
            "num_workers": 4,
        },
        "benchmark": {
            "max_steps": 4000,
            "needle_freq": 200,
            "checkpoint_every_n_steps": 500,
            "volume_commit_every_n_steps": 500,
            "val_check_interval": 500,
            "log_every_n_steps": 50,
        },
    })


@app.function(
    image=image,
    gpu="A100",
    timeout=36000,
    volumes={ARTIFACT_MOUNT: artifact_volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=30.0),
)
def run_benchmark(residual_mode: str = "recurrent_residual"):
    """Train one mode on one A100; artifacts under /artifacts/<mode>/."""
    import os
    import sys

    os.chdir("/root/rr")
    sys.path.insert(0, "/root/rr")

    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ["BENCHMARK_RESUME"] = "1"
    os.environ["BENCHMARK_COMPILE"] = "0"
    os.environ["HF_HOME"] = f"{ARTIFACT_MOUNT}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{ARTIFACT_MOUNT}/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = f"{ARTIFACT_MOUNT}/hf_cache/datasets"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from benchmarks.artifacts import commit_modal_volume, write_status
    from benchmarks.benchmark_engine import run

    write_status(
        residual_mode,
        status="booting",
        modal_function="run_benchmark",
    )
    commit_modal_volume()

    cfg = _build_cfg(residual_mode)
    log.info("Starting benchmark job: %s", residual_mode)
    try:
        run(cfg)
        artifact_volume.commit()
        log.info("Benchmark succeeded: %s", residual_mode)
    except Exception:
        log.error("Benchmark failed: %s\n%s", residual_mode, traceback.format_exc())
        try:
            artifact_volume.commit()
        except Exception:
            log.exception("Failed to commit volume after error")
        raise


def _write_spawn_manifest(handles: dict[str, object]) -> Path:
    """Write spawn handles locally so detached runs remain traceable."""
    manifest_dir = Path("benchmarks")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "last_spawn.json"
    payload = {
        "spawned_at": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME_NAME,
        "artifact_mount": ARTIFACT_MOUNT,
        "jobs": {
            mode: {
                "object_id": getattr(handle, "object_id", str(handle)),
            }
            for mode, handle in handles.items()
        },
        "status_paths": {
            mode: f"{ARTIFACT_MOUNT}/{mode}/status.json"
            for mode in handles
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@app.local_entrypoint()
def main(wait: bool = False) -> None:
    """Spawn RR + AttnRes on two A100s (detached by default)."""
    print("Spawning recurrent_residual on A100...")
    h_rr = run_benchmark.spawn(residual_mode="recurrent_residual")
    print("Spawning attnres_block on A100...")
    h_attn = run_benchmark.spawn(residual_mode="attnres_block")

    manifest_path = _write_spawn_manifest({
        "recurrent_residual": h_rr,
        "attnres_block": h_attn,
    })

    print(f"Spawned 2 jobs (1x A100 each). Manifest: {manifest_path.resolve()}")
    print(f"Artifacts volume: {VOLUME_NAME} -> {ARTIFACT_MOUNT}/<mode>/")
    print("  status.json, checkpoints/, csv_logs/")
    print("Monitor: Modal dashboard OR `modal volume ls rr-benchmark-artifacts`")

    if wait:
        print("Waiting for both jobs...")
        errors: list[str] = []
        for name, handle in [("recurrent_residual", h_rr), ("attnres_block", h_attn)]:
            try:
                handle.get()
                print(f"  OK: {name}")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"  FAILED: {name}: {exc}")
        if errors:
            raise RuntimeError("Benchmark failures:\n" + "\n".join(errors))
        print("Both benchmarks completed.")
    else:
        print("Detached. Inspect benchmarks/last_spawn.json for Modal object IDs.")
