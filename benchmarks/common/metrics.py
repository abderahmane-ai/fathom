"""Metric helpers for benchmark runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


class ThroughputMeter:
    """Track elapsed time and token throughput.

    Args:
        tokens_per_step: Number of tokens processed per optimizer step.
    """

    def __init__(self, tokens_per_step: int) -> None:
        self.tokens_per_step = tokens_per_step
        self.started_at = time.perf_counter()

    def tokens_per_second(self, step: int) -> float:
        """Compute throughput.

        Args:
            step: Completed optimizer steps.

        Returns:
            Tokens per second since construction.
        """
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return float(step * self.tokens_per_step / elapsed)


def peak_cuda_memory_mb() -> float:
    """Return peak CUDA memory allocated in MiB.

    Returns:
        Peak allocated memory, or ``0.0`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def safe_float(value: Any) -> float:
    """Convert scalar-like values to ``float``.

    Args:
        value: Python scalar or scalar tensor.

    Returns:
        Float representation.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload atomically.

    Args:
        path: Destination path.
        payload: JSON-serializable dictionary.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

