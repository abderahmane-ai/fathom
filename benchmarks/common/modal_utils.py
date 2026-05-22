"""Modal helper functions shared by benchmark entrypoints."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.artifacts import repo_root

REMOTE_ROOT = "/root/rr"
ARTIFACT_MOUNT = "/artifacts"
VOLUME_NAME = "rr-benchmark-artifacts"


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

