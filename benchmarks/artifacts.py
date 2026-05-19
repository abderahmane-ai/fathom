"""Persistent benchmark artifacts: status files, checkpoint discovery, Modal volume sync."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def artifact_root() -> Path:
    """Root directory for all benchmark artifacts (logs, checkpoints, status)."""
    return Path(os.environ.get("BENCHMARK_ARTIFACT_ROOT", "artifacts"))


def run_dir(residual_mode: str) -> Path:
    """Per-mode artifact directory."""
    return artifact_root() / residual_mode


def status_path(residual_mode: str) -> Path:
    """Path to the JSON status file for a benchmark run."""
    return run_dir(residual_mode) / "status.json"


def checkpoint_dir(residual_mode: str) -> Path:
    """Directory where Lightning ModelCheckpoint files are stored."""
    return run_dir(residual_mode) / "checkpoints"


def log_dir(residual_mode: str) -> Path:
    """Directory for CSVLogger output."""
    return run_dir(residual_mode) / "csv_logs"


def write_status(residual_mode: str, **fields: Any) -> None:
    """Merge fields into the run status JSON and flush to disk."""
    path = status_path(residual_mode)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt status file %s; resetting", path)

    payload.setdefault("residual_mode", residual_mode)
    payload["updated_at"] = _utc_now()
    payload.update(fields)

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_resume_checkpoint(residual_mode: str) -> Optional[str]:
    """Find the best checkpoint to resume from, if any."""
    ckpt_dir = checkpoint_dir(residual_mode)
    if not ckpt_dir.is_dir():
        return None

    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return str(last.resolve())

    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0].resolve())
    return None


def commit_modal_volume() -> None:
    """Persist mounted Modal volume writes (no-op outside Modal)."""
    vol_name = os.environ.get("BENCHMARK_VOLUME_NAME")
    if not vol_name:
        return
    try:
        import modal

        modal.Volume.from_name(vol_name).commit()
        log.info("Committed Modal volume %s", vol_name)
    except Exception:
        log.exception("Modal volume commit failed for %s", vol_name)
