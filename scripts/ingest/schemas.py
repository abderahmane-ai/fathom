"""Result schemas for the 5 benchmark artifact formats.

Each schema is a dataclass with a ``from_json(path)`` classmethod and a
``to_row()`` method that produces a single flat dict suitable for writing
as a row of a CSV.  All schemas are forgiving: they accept None / missing
fields and produce row dicts with None where data is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DPSResult:
    """Schema for ``depth_preservation/<run_id>/<mode>_dps.json``.

    Attributes:
        residual_mode: e.g. "vega".
        run_id: e.g. "depth_preservation-20260601T120000Z".
        dps_scores: Per-layer DPS values, length = num_layers - 1.
        gps_scores: Per-layer GPS values, same length as dps_scores.
        dri: Dilution Resistance Index (mean of first-half DPS).
        gpi: Gradient Preservation Index (mean of first-half GPS).
        n_tokens: Tokens processed during probing.
        n_layers: Number of layers probed (= len(dps_scores) + 1).
        benchmark_name: Always "depth_preservation".
    """

    residual_mode: str
    run_id: str
    dps_scores: list[float] = field(default_factory=list)
    gps_scores: list[float] = field(default_factory=list)
    dri: float | None = None
    gpi: float | None = None
    n_tokens: int | None = None
    n_layers: int | None = None
    benchmark_name: str = "depth_preservation"

    @classmethod
    def from_json(cls, path: Path | str) -> DPSResult:
        """Load a DPS result from a JSON file, accepting both legacy and canonical paths."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dps_scores = payload.get("dps_scores", []) or []
        return cls(
            residual_mode=payload.get("residual_mode", "?"),
            run_id=payload.get("run_id", "?"),
            dps_scores=[float(x) for x in dps_scores],
            gps_scores=[float(x) for x in (payload.get("gps_scores", []) or [])],
            dri=_to_float(payload.get("dri")),
            gpi=_to_float(payload.get("gpi")),
            n_tokens=_to_int(payload.get("n_tokens")),
            n_layers=len(dps_scores) + 1 if dps_scores else None,
            benchmark_name="depth_preservation",
        )

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single row of CSV-ready data."""
        return {
            "benchmark_name": self.benchmark_name,
            "residual_mode": self.residual_mode,
            "run_id": self.run_id,
            "dri": self.dri,
            "gpi": self.gpi,
            "n_tokens": self.n_tokens,
            "n_layers": self.n_layers,
            "dps_mean": _mean(self.dps_scores),
            "gps_mean": _mean(self.gps_scores),
            "dps_min": _min(self.dps_scores),
            "dps_max": _max(self.dps_scores),
        }


@dataclass
class LatencyProfile:
    """Schema for ``inference_memory/<run_id>/profile_results.json``.

    A profile is a dict mode -> list of (layers, peak_vram_mb,
    mean_latency_ms, p50_latency_ms, p99_latency_ms, tokens_per_second).
    Each (mode, layers) pair becomes one row in the CSV.

    Attributes:
        benchmark_name: Always "inference_memory".
        residual_mode: e.g. "standard".
        run_id: From the parent directory.
        layers: Depth profiled.
        peak_vram_mb: Peak VRAM during forward pass.
        mean_latency_ms: Mean forward latency.
        p50_latency_ms: 50th percentile latency.
        p99_latency_ms: 99th percentile latency.
        tokens_per_second: Throughput at this depth.
    """

    benchmark_name: str = "inference_memory"
    residual_mode: str = "?"
    run_id: str = "?"
    layers: int = 0
    peak_vram_mb: float | None = None
    mean_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    tokens_per_second: float | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> list[LatencyProfile]:
        """Load a full profile (one entry per (mode, layers) pair)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        run_id = Path(path).parent.name
        results: list[LatencyProfile] = []
        for mode, rows in payload.items():
            for row in rows or []:
                results.append(
                    cls(
                        residual_mode=mode,
                        run_id=run_id,
                        layers=_to_int(row.get("layers")) or 0,
                        peak_vram_mb=_to_float(row.get("peak_vram_mb")),
                        mean_latency_ms=_to_float(row.get("mean_latency_ms")),
                        p50_latency_ms=_to_float(row.get("p50_latency_ms")),
                        p99_latency_ms=_to_float(row.get("p99_latency_ms")),
                        tokens_per_second=_to_float(row.get("tokens_per_second")),
                    )
                )
        return results

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NIAHResult:
    """Schema for ``natural_niah/<mode>/<lm_run_id>/niah_result.json``.

    Attributes:
        benchmark_name: Always "natural_niah".
        residual_mode: e.g. "vega".
        run_id: The lm_run_id that produced the checkpoint.
        success: Whether the passkey was retrieved.
        generated_text: The model's completion.
        input_length: Number of input tokens in the prompt.
    """

    benchmark_name: str = "natural_niah"
    residual_mode: str = "?"
    run_id: str = "?"
    success: bool | None = None
    generated_text: str | None = None
    input_length: int | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> NIAHResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            residual_mode=payload.get("mode", "?"),
            run_id=Path(path).parent.name,
            success=bool(payload["success"]) if "success" in payload else None,
            generated_text=payload.get("generated_text"),
            input_length=_to_int(payload.get("input_length")),
        )

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LMRunSummary:
    """Schema for ``<bench>/<mode>/<run_id>/metrics/summary.json`` (any training benchmark).

    Used for ``lm_quality``, ``ablation``, ``depth_needle``, ``iso_flop``,
    ``scaling_efficiency``.

    Attributes:
        benchmark_name: e.g. "lm_quality".
        residual_mode: e.g. "vega".
        run_id: e.g. "lm_quality-20260601T120000Z".
        parameter_count: Total parameter count.
        elapsed_seconds: Wall-clock training time.
        global_step: Number of optimizer steps.
        peak_cuda_memory_mb: Peak GPU memory.
    """

    benchmark_name: str = "?"
    residual_mode: str = "?"
    run_id: str = "?"
    parameter_count: int | None = None
    elapsed_seconds: float | None = None
    global_step: int | None = None
    peak_cuda_memory_mb: float | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> LMRunSummary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            benchmark_name=payload.get("benchmark_name", "?"),
            residual_mode=payload.get("residual_mode", "?"),
            run_id=payload.get("run_id", "?"),
            parameter_count=_to_int(payload.get("parameter_count")),
            elapsed_seconds=_to_float(payload.get("elapsed_seconds")),
            global_step=_to_int(payload.get("global_step")),
            peak_cuda_memory_mb=_to_float(payload.get("peak_cuda_memory_mb")),
        )

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LMRunStep:
    """Schema for a single step in ``metrics.csv`` (Lightning's per-step metrics).

    Used to produce step-level rows for loss curves and gradient-norm plots.

    Attributes:
        benchmark_name: Parent benchmark.
        residual_mode: Parent mode.
        run_id: Parent run id.
        step: Lightning global step.
        epoch: Lightning epoch.
        train_loss: Training loss in nats (None if not present).
        val_loss: Validation loss in nats.
        grad_global_norm: L2 norm of all gradients.
        tokens_per_second: Throughput.
        learning_rate: Current LR.
    """

    benchmark_name: str = "?"
    residual_mode: str = "?"
    run_id: str = "?"
    step: int = 0
    epoch: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    grad_global_norm: float | None = None
    tokens_per_second: float | None = None
    learning_rate: float | None = None

    @classmethod
    def from_metrics_csv(cls, path: Path | str, *, benchmark_name: str, residual_mode: str, run_id: str) -> list[LMRunStep]:
        """Read Lightning's metrics.csv and produce one LMRunStep per row.

        Handles the column names Lightning uses: ``step``, ``epoch``, ``train/loss``,
        ``val/loss``, ``grad/global_norm``, ``perf/tokens_per_second``,
        ``lr-AdamW`` (or similar).
        """
        import csv

        results: list[LMRunStep] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(
                    cls(
                        benchmark_name=benchmark_name,
                        residual_mode=residual_mode,
                        run_id=run_id,
                        step=_to_int(row.get("step")) or 0,
                        epoch=_to_float(row.get("epoch")),
                        train_loss=_to_float(row.get("train/loss")),
                        val_loss=_to_float(row.get("val/loss")),
                        grad_global_norm=_to_float(row.get("grad/global_norm")),
                        tokens_per_second=_to_float(row.get("perf/tokens_per_second")),
                        learning_rate=_find_lr(row),
                    )
                )
        return results

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return min(values)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)


def _find_lr(row: dict[str, str]) -> float | None:
    """Find the learning rate column in a Lightning metrics.csv row.

    Lightning's LearningRateMonitor uses ``lr-AdamW`` or ``lr-Adam`` depending
    on the optimizer, or ``lr`` if there's a single group.
    """
    for key in row:
        if key.startswith("lr-"):
            return _to_float(row[key])
    return _to_float(row.get("lr"))
