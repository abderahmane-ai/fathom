"""Filesystem helpers for benchmark artifacts and status files."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def repo_root() -> Path:
    """Return the repository root.

    Returns:
        Absolute path to the project root.

    Preconditions:
        This module lives in ``benchmarks/common``.
    """
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    """Return the root directory for benchmark artifacts.

    Returns:
        Path from ``BENCHMARK_ARTIFACT_ROOT`` or ``benchmarks/artifacts``.
    """
    configured = os.environ.get("BENCHMARK_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root() / "benchmarks" / "artifacts"


def benchmark_dir(benchmark_name: str) -> Path:
    """Return the artifact directory for one benchmark.

    Args:
        benchmark_name: Name such as ``lm_quality``.

    Returns:
        Benchmark-level artifact directory.
    """
    return artifact_root() / benchmark_name


def run_dir(benchmark_name: str, residual_mode: str, run_id: str = "default") -> Path:
    """Return the artifact directory for a benchmark run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Directory containing checkpoints, logs, metrics, and status.
    """
    return benchmark_dir(benchmark_name) / residual_mode / run_id


def checkpoint_dir(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
) -> Path:
    """Return the checkpoint directory for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Directory for Lightning checkpoints.
    """
    return run_dir(benchmark_name, residual_mode, run_id) / "checkpoints"


def log_dir(benchmark_name: str, residual_mode: str, run_id: str = "default") -> Path:
    """Return the logger directory for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Directory for CSV and WandB logs.
    """
    return run_dir(benchmark_name, residual_mode, run_id) / "logs"


def metrics_dir(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
) -> Path:
    """Return the metrics directory for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Directory for JSON metric summaries.
    """
    return run_dir(benchmark_name, residual_mode, run_id) / "metrics"


def status_path(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
) -> Path:
    """Return the status JSON path for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Path to ``status.json``.
    """
    return run_dir(benchmark_name, residual_mode, run_id) / "status.json"


def _utc_now() -> str:
    """Return a UTC timestamp.

    Returns:
        ISO-8601 timestamp in UTC.
    """
    return datetime.now(timezone.utc).isoformat()


def write_status(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
    **fields: Any,
) -> None:
    """Merge fields into a benchmark status JSON file.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.
        **fields: JSON-serializable values to merge.

    Returns:
        None.
    """
    path = status_path(benchmark_name, residual_mode, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Resetting corrupt status file at %s", path)

    payload.setdefault("benchmark_name", benchmark_name)
    payload.setdefault("residual_mode", residual_mode)
    payload.setdefault("run_id", run_id)
    payload["updated_at"] = _utc_now()
    payload.update(fields)

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def run_metadata_path(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
) -> Path:
    """Return the run-metadata JSON path for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Path to ``run.json`` (full run metadata, written once at start).
    """
    return run_dir(benchmark_name, residual_mode, run_id) / "run.json"


def write_run_metadata(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
    metadata: dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Write (or merge into) the run-metadata JSON file.

    The file is ``run.json`` at the top level of the run directory.  It is
    written once at run start with the full metadata dict, but subsequent
    calls (e.g. on completion) can merge ``ended_at`` and other fields.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.
        metadata: Full metadata dict from
            ``benchmarks.common.run_metadata.capture_run_metadata``.
        **fields: Additional JSON-serializable fields to merge.
    """
    path = run_metadata_path(benchmark_name, residual_mode, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Resetting corrupt run-metadata file at %s", path)

    if metadata is not None:
        payload.update(metadata)
    payload.setdefault("benchmark_name", benchmark_name)
    payload.setdefault("residual_mode", residual_mode)
    payload.setdefault("run_id", run_id)
    payload["updated_at"] = _utc_now()
    payload.update(fields)

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def find_all_runs(benchmark_name: str, root: Path | str | None = None) -> list[dict[str, Any]]:
    """Find all run records under a benchmark directory.

    Walks ``<root>/<benchmark_name>/<residual_mode>/<run_id>/status.json``
    (or ``<root>/<residual_mode>/<run_id>/status.json`` when ``root`` already
    points at the benchmark directory).  Returns one dict per file, with the
    parsed JSON content augmented by ``_path`` (the status.json path) and
    ``_run_dir`` (the run directory).

    Args:
        benchmark_name: Name of the benchmark (used as a filter when root
            points above the benchmark dir).
        root: Search root; defaults to ``artifact_root()``.

    Returns:
        List of run records, each with status.json content + ``_path`` and ``_run_dir``.
    """
    base = Path(root) if root is not None else benchmark_dir(benchmark_name)
    if not base.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Try 3-component pattern first: <root>/<benchmark>/<mode>/<run_id>/status.json
    for status_file in sorted(base.glob(f"{benchmark_name}/*/*/status.json")):
        if str(status_file) in seen:
            continue
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = str(status_file)
        payload["_run_dir"] = str(status_file.parent)
        runs.append(payload)
        seen.add(str(status_file))
    # Fallback 2-component pattern: <root>/<mode>/<run_id>/status.json (caller passed benchmark dir)
    for status_file in sorted(base.glob("*/*/status.json")):
        if str(status_file) in seen:
            continue
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = str(status_file)
        payload["_run_dir"] = str(status_file.parent)
        runs.append(payload)
        seen.add(str(status_file))
    return runs


def find_resume_checkpoint(
    benchmark_name: str,
    residual_mode: str,
    run_id: str = "default",
) -> str | None:
    """Find the newest checkpoint for a run.

    Args:
        benchmark_name: Benchmark folder name.
        residual_mode: Residual mode under evaluation.
        run_id: Stable run identifier.

    Returns:
        Absolute checkpoint path, or ``None`` when no checkpoint exists.
    """
    directory = checkpoint_dir(benchmark_name, residual_mode, run_id)
    if not directory.is_dir():
        return None

    last = directory / "last.ckpt"
    if last.is_file():
        return str(last.resolve())

    candidates = sorted(directory.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    return str(candidates[-1].resolve())


def resolve_val_check_interval(num_train_batches: int, requested: int | float) -> int | float:
    """Clamp Lightning validation cadence to a valid value.

    Args:
        num_train_batches: Estimated batches in the epoch.
        requested: Integer step interval or fractional epoch interval.

    Returns:
        ``requested`` when valid, otherwise ``1.0`` for once per epoch.

    Preconditions:
        ``num_train_batches`` is positive.
    """
    if isinstance(requested, int) and requested > num_train_batches:
        return 1.0
    return requested


def commit_modal_volume(volume_name: str | None = None) -> None:
    """Commit a mounted Modal volume when running remotely.

    Args:
        volume_name: Modal volume name. Defaults to ``BENCHMARK_VOLUME_NAME``.

    Returns:
        None.
    """
    name = volume_name or os.environ.get("BENCHMARK_VOLUME_NAME")
    if not name:
        return
    try:
        import modal

        modal.Volume.from_name(name).commit()
    except Exception:
        log.exception("Modal volume commit failed for %s", name)
