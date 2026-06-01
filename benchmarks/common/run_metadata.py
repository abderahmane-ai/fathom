"""Run-level metadata capture for reproducibility.

Every modal benchmark run should call `capture_run_metadata()` once at start
and write the returned dict to `<run_dir>/run.json`.  This gives a single
artifact that answers: "what code, what config, what hardware, what seed, and
what time produced this result?".
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def get_git_commit(repo_root: Path | str | None = None, short: bool = True) -> str | None:
    """Return the current git commit hash, or None if not in a git repo / git missing.

    Args:
        repo_root: directory to run `git rev-parse` from.  Defaults to cwd.
        short: if True, return 12-character short hash; else full 40-char hash.
    """
    try:
        if short:
            cmd = ["git", "rev-parse", "--short=12", "HEAD"]
        else:
            cmd = ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def get_git_status(repo_root: Path | str | None = None) -> str | None:
    """Return `git status --short` output, or None on failure.

    Empty string means a clean tree; non-empty means there are uncommitted
    changes that may affect reproducibility.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def get_gpu_info() -> dict[str, Any]:
    """Return GPU name, total memory, and CUDA version, or empty fields if no GPU.

    Returns a dict with keys: name, total_memory_mb, cuda_version, driver_version.
    """
    info: dict[str, Any] = {
        "name": None,
        "total_memory_mb": None,
        "cuda_version": None,
        "driver_version": None,
    }
    if not torch.cuda.is_available():
        return info
    try:
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["total_memory_mb"] = round(props.total_memory / (1024 * 1024))
        info["cuda_version"] = torch.version.cuda
    except (RuntimeError, AttributeError):
        pass
    return info


def get_environment_info() -> dict[str, Any]:
    """Return platform + Python + torch versions for the run record."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_version": torch.version.cuda,
        "pid": os.getpid(),
    }


def capture_run_metadata(
    *,
    benchmark_name: str,
    run_id: str,
    residual_mode: str,
    config: Any | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a metadata dict for a single run.

    Args:
        benchmark_name: e.g. "lm_quality".
        run_id: e.g. "lm_quality-20260601T120000Z".
        residual_mode: e.g. "hyper_connection".
        config: an OmegaConf DictConfig (or any object with `.to_container()` /
            `__str__`); will be serialized to a plain dict for JSON.
        seed: random seed for this run (optional, recommended).
        extra: any additional fields the caller wants in the metadata.
        repo_root: where to look for git; defaults to cwd.

    Returns:
        A JSON-serializable dict.  Always contains: benchmark_name, run_id,
        residual_mode, started_at (ISO UTC), git_commit, git_status, gpu,
        environment.  Optionally contains: seed, config, plus any extras.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    config_dict: dict[str, Any] | None = None
    if config is not None:
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(config, (DictConfig, ListConfig)):
                config_dict = OmegaConf.to_container(config, resolve=True)
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {"repr": str(config)}
        except ImportError:
            config_dict = config if isinstance(config, dict) else {"repr": str(config)}
    metadata: dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "run_id": run_id,
        "residual_mode": residual_mode,
        "started_at": started_at,
        "git_commit": get_git_commit(repo_root),
        "git_status": get_git_status(repo_root),
        "gpu": get_gpu_info(),
        "environment": get_environment_info(),
        "seed": seed,
        "config": config_dict,
    }
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


def log_run_banner(log: Any, metadata: dict[str, Any]) -> None:
    """Emit a one-shot banner summarising the run, structured for readability.

    The banner has 6 lines, each a separate log.info call so timestamps and
    interleaved output stay clean.  All fields come from the metadata dict
    produced by `capture_run_metadata`.
    """
    log.info("=" * 78)
    log.info("RUN START: %s / %s / %s", metadata["benchmark_name"], metadata["residual_mode"], metadata["run_id"])
    log.info("=" * 78)
    git = metadata.get("git_commit") or "UNKNOWN"
    git_status = metadata.get("git_status")
    if git_status:
        log.info("git:    %s (dirty: %d files changed)", git, len(git_status.splitlines()))
    else:
        log.info("git:    %s (clean)", git)
    log.info("seed:   %s", metadata.get("seed"))
    gpu = metadata.get("gpu") or {}
    if gpu.get("name"):
        log.info("gpu:    %s (%d MiB total, CUDA %s)", gpu["name"], gpu.get("total_memory_mb", -1), gpu.get("cuda_version"))
    else:
        log.info("gpu:    none (CPU-only)")
    env = metadata.get("environment") or {}
    log.info(
        "python: %s | torch %s | host %s | pid %d",
        env.get("python_version", "?"),
        env.get("torch_version", "?"),
        env.get("hostname", "?"),
        env.get("pid", -1),
    )
    log.info("=" * 78)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS for log output."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def log_run_finish(log: Any, *, status: str, elapsed_seconds: float, **fields: Any) -> None:
    """Emit a one-shot end-of-run banner with the final status and any extra fields."""
    log.info("=" * 78)
    log.info("RUN END:   status=%s | elapsed=%s", status, format_duration(elapsed_seconds))
    for key, value in fields.items():
        log.info("  %-15s %s", key + ":", value)
    log.info("=" * 78)


class WallClock:
    """Tiny monotonic wall-clock helper so callers don't have to import time."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
