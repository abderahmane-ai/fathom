"""Modal entrypoint for Natural Text Needle In A Haystack (NIAH)."""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

import modal
import torch

from benchmarks.common.artifacts import (
    repo_root,
    run_dir,
    write_run_metadata,
    write_status,
)
from benchmarks.common.configs import config_for_mode, load_benchmark_config
from benchmarks.common.modal_utils import (
    ARTIFACT_MOUNT,
    REMOTE_ROOT,
    VOLUME_NAME,
    default_retries,
    modal_ignore_patterns,
    print_run_summary,
    write_spawn_manifest,
)
from benchmarks.common.run_metadata import (
    WallClock,
    capture_run_metadata,
    log_run_banner,
    log_run_finish,
)

BENCHMARK_NAME = "natural_niah"
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
app = modal.App("rr-natural-niah")

# Long natural text context (Paul Graham essay excerpt style)
HAYSTACK_TEXT = (
    "In the early days of Y Combinator, we noticed a distinct pattern among the most "
    "successful startups. They weren't necessarily the ones with the most brilliant "
    "initial ideas, nor were they the teams with the most impressive academic pedigrees. "
    "Instead, the defining characteristic of a breakout success was an obsessive focus "
    "on building something people actually wanted, coupled with an extraordinarily tight "
    "feedback loop. The founders who thrived were those who could ship a minimal viable "
    "product, talk to their users, and iterate on a daily or sometimes even hourly basis. "
    "This agility allowed them to navigate the unpredictable terrain of "
    "finding product-market fit. "
    "We used to tell them that the biggest risk wasn't launching something imperfect, "
    "but rather spending months building something in isolation only to discover that "
    "nobody cared about it. The best founders were relentless. They viewed every bug report, "
    "every feature request, and every churned user as a vital piece of intelligence. "
    "Over time, this process of continuous refinement compounded, leading to products that "
    "felt almost magical in their utility. It is a simple formula, but executing it requires "
    "a level of discipline and humility that is surprisingly rare. It demands that you "
    "subordinate your ego to the reality of the market, acknowledging that your initial "
    "assumptions are likely wrong. When we reflect on the companies that made it big, "
    "from Airbnb to Stripe, this philosophy was always at the core of their operations. "
    "They understood that startups are not about executing a master plan, but about "
    "rapidly discovering the right plan through trial and error. "
)


def _prepare_remote() -> None:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    os.environ["BENCHMARK_ARTIFACT_ROOT"] = ARTIFACT_MOUNT
    os.environ["BENCHMARK_VOLUME_NAME"] = VOLUME_NAME
    os.environ.setdefault("HF_HOME", f"{ARTIFACT_MOUNT}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{ARTIFACT_MOUNT}/hf_cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@torch.no_grad()
def generate(model: torch.nn.Module, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Simple autoregressive generation loop."""
    model.eval()
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        # Slice to context window if needed, but here we assume it fits
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated


def _run_niah_eval(mode: str, lm_run_id: str, compile: bool = False) -> None:
    import traceback

    _prepare_remote()
    cfg = config_for_mode(load_benchmark_config(BENCHMARK_NAME), mode)
    metadata = capture_run_metadata(
        benchmark_name=BENCHMARK_NAME,
        run_id=lm_run_id,
        residual_mode=mode,
        config=cfg,
        extra={"compile": compile, "lm_run_id": lm_run_id},
    )
    write_run_metadata(BENCHMARK_NAME, mode, lm_run_id, metadata=metadata)
    log_run_banner(log, metadata)
    write_status(BENCHMARK_NAME, mode, lm_run_id, status="starting")

    clock = WallClock()
    try:
        from transformers import AutoTokenizer

        from src.modules.transformer import TransformerDecoder

        ckpt_dir = Path(ARTIFACT_MOUNT) / "lm_quality" / mode / lm_run_id / "checkpoints"
        checkpoints = glob.glob(str(ckpt_dir / "*.ckpt"))

        if not checkpoints:
            log.warning("No checkpoints found for %s at %s. Skipping NIAH eval.", mode, ckpt_dir)
            write_status(
                BENCHMARK_NAME,
                mode,
                lm_run_id,
                status="skipped",
                reason="no checkpoints",
                elapsed_seconds=clock.elapsed(),
            )
            artifact_volume.commit()
            return

        checkpoint_path = checkpoints[0]  # Just take the first/last available
        log.info("Loading checkpoint %s for %s", checkpoint_path, mode)

        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name, use_fast=True)

        model = TransformerDecoder(cfg.model)
        # Lightning checkpoints wrap the model state_dict in `state_dict`, often prefixed with `model.`
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda().eval()
        if compile:
            model = torch.compile(model)

        # Prepare Context: inject passkey at 20% mark
        words = HAYSTACK_TEXT.split()
        inject_idx = len(words) // 5
        words.insert(inject_idx, "The secret passkey to the vault is 84729.")

        context_text = " ".join(words)
        prompt_text = context_text + " The secret passkey to the vault is"

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()

        log.info("Running generation for %s (input length: %d)...", mode, input_ids.shape[1])
        output_ids = generate(model, input_ids, max_new_tokens=5)

        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1] :])
        success = "84729" in generated_text

        log.info("[%s] Generated completion: %s", mode, generated_text)
        log.info("[%s] Passkey retrieved: %s", mode, success)

        results = {
            "mode": mode,
            "success": success,
            "generated_text": generated_text,
            "input_length": input_ids.shape[1],
        }

        out_dir = Path(ARTIFACT_MOUNT) / "natural_niah" / mode / lm_run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "niah_result.json", "w") as f:
            json.dump(results, f, indent=2)

        # Also write to canonical artifacts/ tree
        canonical_dir = run_dir(BENCHMARK_NAME, mode, lm_run_id)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        with open(canonical_dir / "niah_result.json", "w") as f:
            json.dump(results, f, indent=2)

        elapsed = clock.elapsed()
        write_status(
            BENCHMARK_NAME,
            mode,
            lm_run_id,
            status="completed",
            success=success,
            input_length=int(input_ids.shape[1]),
            elapsed_seconds=elapsed,
        )
        log_run_finish(
            log,
            status="completed",
            elapsed_seconds=elapsed,
            passkey_retrieved=success,
            input_length=int(input_ids.shape[1]),
        )
        artifact_volume.commit()
    except Exception as exc:
        elapsed = clock.elapsed()
        tb = traceback.format_exc()
        write_status(
            BENCHMARK_NAME,
            mode,
            lm_run_id,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=tb,
            elapsed_seconds=elapsed,
        )
        log_run_finish(log, status="failed", elapsed_seconds=elapsed, error=str(exc))
        artifact_volume.commit()
        raise


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume}, retries=default_retries())
def run_standard_niah(lm_run_id: str, compile: bool = False) -> None:
    _run_niah_eval("standard", lm_run_id, compile=compile)


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume}, retries=default_retries())
def run_recurrent_residual_niah(lm_run_id: str, compile: bool = False) -> None:
    _run_niah_eval("recurrent_residual", lm_run_id, compile=compile)


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume}, retries=default_retries())
def run_vega_niah(lm_run_id: str, compile: bool = False) -> None:
    _run_niah_eval("vega", lm_run_id, compile=compile)


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume}, retries=default_retries())
def run_block_attnres_niah(lm_run_id: str, compile: bool = False) -> None:
    _run_niah_eval("block_attnres", lm_run_id, compile=compile)


@app.function(image=image, gpu="A100", timeout=3600, volumes={ARTIFACT_MOUNT: artifact_volume}, retries=default_retries())
def run_hyper_connection_niah(lm_run_id: str, compile: bool = False) -> None:
    """Run the mHC-Lite hyper-connection NIAH evaluation.

    Args:
        lm_run_id: The LM quality run id to evaluate.
        compile: Whether to compile the model.

    Returns:
        None.
    """
    _run_niah_eval("hyper_connection", lm_run_id, compile=compile)


@app.local_entrypoint()
def main(lm_run_id: str, wait: bool = False, compile: bool = False) -> None:
    """Evaluate Natural Text NIAH on existing LM Quality checkpoints.

    Args:
        lm_run_id: The LM quality run id to evaluate.
        wait: Whether to wait for remote jobs.
        compile: Whether to compile the models using torch.compile.

    Returns:
        None.
    """
    handles = {
        "standard": run_standard_niah.spawn(lm_run_id, compile=compile),
        "recurrent_residual": run_recurrent_residual_niah.spawn(lm_run_id, compile=compile),
        "vega": run_vega_niah.spawn(lm_run_id, compile=compile),
        "block_attnres": run_block_attnres_niah.spawn(lm_run_id, compile=compile),
        "hyper_connection": run_hyper_connection_niah.spawn(lm_run_id, compile=compile),
    }
    manifest = write_spawn_manifest(BENCHMARK_NAME, handles, lm_run_id)
    print(f"Spawned {BENCHMARK_NAME} eval jobs with lm_run_id={lm_run_id}")
    print(f"Manifest: {manifest}")
    if wait:
        for mode, handle in handles.items():
            log.info("Waiting for %s", mode)
            handle.get()
        print_run_summary(log, BENCHMARK_NAME, lm_run_id, list(handles.keys()))
