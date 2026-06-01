"""Modal entrypoint for Inference Memory Profiling benchmark."""

from __future__ import annotations

import logging
import os
import sys
import json
import gc
from pathlib import Path

import modal
import torch

from benchmarks.common.artifacts import repo_root
from benchmarks.common.configs import config_for_mode, load_benchmark_config, make_run_id
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
    gpu="A10G", # Use a smaller GPU since it's just inference testing
    timeout=3600,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_memory_profile(run_id: str) -> None:
    """Run inference memory profiling."""
    _prepare_remote()
    from src.modules.transformer import TransformerDecoder

    if not torch.cuda.is_available():
        log.warning("CUDA is not available. Script will output zeros.")
        return

    modes = ["standard", "recurrent_residual", "vega", "block_attnres"]
    depths = [12, 24, 48, 96]
    seq_len = 100
    
    results = {mode: [] for mode in modes}
    base_cfg = load_benchmark_config(BENCHMARK_NAME)

    for mode in modes:
        for L in depths:
            log.info(f"Profiling {mode} at {L} layers...")
            
            cfg = config_for_mode(base_cfg, mode)
            cfg.model.num_layers = L
            
            model = TransformerDecoder(cfg.model).cuda().eval()
            x = torch.randint(0, 1000, (1, seq_len)).cuda()
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(x)
                
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            results[mode].append({"layers": L, "peak_vram_mb": peak_memory})
            
            del model
            del x
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
def main(wait: bool = False) -> None:
    run_id = make_run_id(BENCHMARK_NAME)
    handles = {
        "profile": run_memory_profile.spawn(run_id),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, run_id)
    print(f"Spawned {BENCHMARK_NAME} jobs with run_id={run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
