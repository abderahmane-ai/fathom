"""Modal helper functions shared by benchmark entrypoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from benchmarks.common.artifacts import benchmark_dir, find_all_runs, repo_root

REMOTE_ROOT = "/root/rr"
ARTIFACT_MOUNT = "/artifacts"
VOLUME_NAME = "rr-benchmark-artifacts"


def default_retries(max_retries: int = 1) -> modal.Retries:
    """Return a standard Modal Retries policy.

    Args:
        max_retries: Maximum retry count.  Default 1 = one automatic retry on
            transient failure (Modal container start, OOM, etc.).

    Returns:
        ``modal.Retries`` with exponential backoff (initial 30s, coefficient 2.0).
    """
    return modal.Retries(max_retries=max_retries, backoff_coefficient=2.0, initial_delay=30.0)


def modal_ignore_patterns() -> list[str]:
    """Return local paths excluded from Modal uploads.

    Returns:
        Ignore patterns accepted by ``Image.add_local_dir``.
    """
    return [
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "**/__pycache__",
        "benchmarks/artifacts",
        "logs",
        "lightning_logs",
    ]


def write_spawn_manifest(
    benchmark_name: str,
    handles: dict[str, Any],
    run_id: str,
) -> Path:
    """Write local Modal spawn metadata.

    Args:
        benchmark_name: Benchmark folder name.
        handles: Mapping from residual mode to Modal call handle.
        run_id: Run identifier.

    Returns:
        Path to the manifest.
    """
    directory = repo_root() / "benchmarks" / benchmark_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "last_spawn.json"
    payload = {
        "benchmark_name": benchmark_name,
        "run_id": run_id,
        "spawned_at": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME_NAME,
        "artifact_mount": ARTIFACT_MOUNT,
        "jobs": {
            mode: {"object_id": getattr(handle, "object_id", str(handle))}
            for mode, handle in handles.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_run_summary(
    log: logging.Logger,
    benchmark_name: str,
    run_id: str,
    modes: list[str],
    root: Path | str | None = None,
) -> None:
    """Print a one-shot summary table of every (mode, status) pair for a run.

    Looks up ``status.json`` files under
    ``<root>/<benchmark_name>/<mode>/<run_id>/status.json`` (or the
    ``<root>/<mode>/<run_id>/`` layout) and prints a 5-column table:
    mode, status, step, elapsed, peak_mem.  Called from each modal
    ``local_entrypoint`` after all handles are awaited.

    Args:
        log: Logger to emit the summary through.
        benchmark_name: Benchmark folder name.
        run_id: Run identifier.
        modes: Modes to include in the summary.
        root: Search root; defaults to ``benchmark_dir(benchmark_name)``.

    Returns:
        None.
    """
    base = Path(root) if root is not None else benchmark_dir(benchmark_name)
    log.info("=" * 78)
    log.info("RUN SUMMARY: %s | run_id=%s", benchmark_name, run_id)
    log.info("-" * 78)
    log.info("%-22s %-10s %10s %12s %12s", "mode", "status", "step", "elapsed_s", "peak_mem_mb")
    log.info("-" * 78)
    n_completed = 0
    n_failed = 0
    for mode in modes:
        # Try canonical layout first
        for candidate in (
            base / mode / run_id / "status.json",
            base / benchmark_name / mode / run_id / "status.json",
        ):
            if candidate.is_file():
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    payload = {}
                status = payload.get("status", "?")
                step = payload.get("global_step", "-")
                elapsed = payload.get("elapsed_seconds", "-")
                peak = payload.get("peak_cuda_memory_mb", "-")
                if status == "completed":
                    n_completed += 1
                elif status == "failed":
                    n_failed += 1
                log.info(
                    "%-22s %-10s %10s %12s %12s",
                    mode, status, step, _fmt_num(elapsed), _fmt_num(peak),
                )
                break
        else:
            log.info("%-22s %-10s %10s %12s %12s", mode, "not_found", "-", "-", "-")
    log.info("-" * 78)
    log.info("Total: %d completed, %d failed (out of %d)", n_completed, n_failed, len(modes))
    log.info("=" * 78)


def _fmt_num(value: Any) -> str:
    """Format a number for the summary table, or '-' for None/empty."""
    if value is None or value == "-":
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
