import logging
import traceback

import modal

log = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "lightning",
        "transformers",
        "datasets",
        "hydra-core",
        "omegaconf",
        "einops",
        "wandb",
    )
    .add_local_dir(".", remote_path="/root/rr")
)

log_volume = modal.Volume.from_name("rr-benchmark-logs", create_if_missing=True)

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
        },
    })


@app.function(
    image=image,
    gpu="A100",
    timeout=36000,
    volumes={"/logs": log_volume},
)
def run_benchmark(residual_mode: str = "recurrent_residual"):
    """Train one residual mode on a single A100. Logs to /logs volume + Modal stdout."""
    import os
    import sys

    os.chdir("/root/rr")
    sys.path.insert(0, "/root/rr")
    os.environ["BENCHMARK_LOG_DIR"] = "/logs"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from benchmarks.benchmark_engine import run

    cfg = _build_cfg(residual_mode)
    log.info("Starting Modal benchmark job: %s", residual_mode)
    try:
        run(cfg)
        log_volume.commit()
        log.info("Benchmark succeeded: %s", residual_mode)
    except Exception:
        log.error("Benchmark failed: %s\n%s", residual_mode, traceback.format_exc())
        raise


@app.local_entrypoint()
def main(wait: bool = False):
    """Spawn RR + AttnRes on two A100s in parallel. Use --wait to block until done."""
    print("Spawning recurrent_residual on A100...")
    h_rr = run_benchmark.spawn(residual_mode="recurrent_residual")
    print("Spawning attnres_block on A100...")
    h_attn = run_benchmark.spawn(residual_mode="attnres_block")
    print("Two jobs spawned (1 GPU each). CSV logs: Modal volume rr-benchmark-logs mounted at /logs")
    if wait:
        print("Waiting for both jobs...")
        h_rr.get()
        h_attn.get()
        print("Both benchmarks completed.")
    else:
        print("Detached. Re-run with: modal run benchmarks/modal_entrypoint.py --wait")
