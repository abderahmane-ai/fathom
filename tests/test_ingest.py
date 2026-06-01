"""Tests for the ingest schemas and collect CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ingest.collect import (
    collect_all_runs,
    collect_dps,
    collect_latency,
    collect_lm_summaries,
    collect_niah,
)
from scripts.ingest.schemas import (
    DPSResult,
    LatencyProfile,
    LMRunStep,
    LMRunSummary,
    NIAHResult,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestDPSResult:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "dps.json"
        _write_json(
            path,
            {
                "residual_mode": "hyper_connection",
                "run_id": "r1",
                "dps_scores": [0.95, 0.92, 0.88],
                "gps_scores": [0.91, 0.85, 0.82],
                "dri": 0.92,
                "gpi": 0.86,
                "n_tokens": 100000,
            },
        )
        result = DPSResult.from_json(path)
        assert result.residual_mode == "hyper_connection"
        assert result.dri == 0.92
        assert result.n_layers == 4  # 3 scores + 1
        row = result.to_row()
        assert row["dps_mean"] == pytest.approx((0.95 + 0.92 + 0.88) / 3)

    def test_to_row_has_all_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "dps.json"
        _write_json(
            path,
            {
                "residual_mode": "standard",
                "run_id": "r1",
                "dps_scores": [0.99, 0.98],
                "gps_scores": [0.97, 0.95],
                "dri": 0.985,
                "gpi": 0.96,
                "n_tokens": 50000,
            },
        )
        result = DPSResult.from_json(path)
        row = result.to_row()
        for key in ("benchmark_name", "residual_mode", "run_id", "dri", "gpi", "n_tokens", "n_layers", "dps_mean", "gps_mean", "dps_min", "dps_max"):
            assert key in row

    def test_missing_fields_are_none(self, tmp_path: Path) -> None:
        path = tmp_path / "dps.json"
        _write_json(path, {"residual_mode": "x", "run_id": "y"})
        result = DPSResult.from_json(path)
        assert result.dri is None
        assert result.dps_scores == []


class TestLatencyProfile:
    def test_from_json_yields_one_per_layer(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.json"
        _write_json(
            path,
            {
                "standard": [
                    {"layers": 12, "peak_vram_mb": 100.0, "mean_latency_ms": 5.0, "p50_latency_ms": 5.0, "p99_latency_ms": 6.0, "tokens_per_second": 1000.0},
                    {"layers": 24, "peak_vram_mb": 200.0, "mean_latency_ms": 10.0, "p50_latency_ms": 10.0, "p99_latency_ms": 12.0, "tokens_per_second": 800.0},
                ],
                "hyper_connection": [
                    {"layers": 12, "peak_vram_mb": 110.0, "mean_latency_ms": 6.0, "p50_latency_ms": 6.0, "p99_latency_ms": 7.0, "tokens_per_second": 950.0},
                ],
            },
        )
        profiles = LatencyProfile.from_json(path)
        assert len(profiles) == 3
        standard_12 = next(p for p in profiles if p.residual_mode == "standard" and p.layers == 12)
        assert standard_12.peak_vram_mb == 100.0
        assert standard_12.tokens_per_second == 1000.0


class TestNIAHResult:
    def test_from_json(self, tmp_path: Path) -> None:
        path = tmp_path / "niah.json"
        _write_json(
            path,
            {
                "mode": "hyper_connection",
                "success": True,
                "generated_text": " 84729",
                "input_length": 350,
            },
        )
        result = NIAHResult.from_json(path)
        assert result.residual_mode == "hyper_connection"
        assert result.success is True
        assert result.input_length == 350
        assert result.run_id == path.parent.name


class TestLMRunSummary:
    def test_from_json(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        _write_json(
            path,
            {
                "benchmark_name": "lm_quality",
                "residual_mode": "standard",
                "run_id": "r1",
                "parameter_count": 1000000,
                "elapsed_seconds": 123.4,
                "global_step": 1000,
                "peak_cuda_memory_mb": 4096.0,
            },
        )
        summary = LMRunSummary.from_json(path)
        assert summary.benchmark_name == "lm_quality"
        assert summary.parameter_count == 1000000


class TestLMRunStep:
    def test_from_metrics_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        path.write_text(
            "step,epoch,train/loss,val/loss,grad/global_norm,perf/tokens_per_second,lr-AdamW\n100,0,4.2,4.3,1.5,5000.0,1e-3\n200,0,3.8,4.0,1.4,5100.0,9.5e-4\n"
        )
        steps = LMRunStep.from_metrics_csv(path, benchmark_name="lm_quality", residual_mode="standard", run_id="r1")
        assert len(steps) == 2
        assert steps[0].step == 100
        assert steps[0].train_loss == 4.2
        assert steps[0].learning_rate == 1e-3
        assert steps[1].grad_global_norm == 1.4

    def test_handles_missing_lr_column(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        path.write_text("step,train/loss\n100,4.2\n")
        steps = LMRunStep.from_metrics_csv(path, benchmark_name="lm_quality", residual_mode="standard", run_id="r1")
        assert steps[0].learning_rate is None


class TestCollectFunctions:
    def _make_dps_tree(self, root: Path) -> None:
        """Create <root>/depth_preservation/<mode>/<run_id>/dps.json + status.json."""
        for mode, dri in [("standard", 0.99), ("hyper_connection", 0.95)]:
            run_id = f"{mode}-r1"
            _write_json(
                root / "depth_preservation" / mode / run_id / "dps.json",
                {
                    "residual_mode": mode,
                    "run_id": run_id,
                    "dps_scores": [dri, dri - 0.01, dri - 0.02],
                    "gps_scores": [dri - 0.05, dri - 0.06, dri - 0.07],
                    "dri": dri,
                    "gpi": dri - 0.06,
                    "n_tokens": 100000,
                },
            )
            _write_json(
                root / "depth_preservation" / mode / run_id / "status.json",
                {
                    "benchmark_name": "depth_preservation",
                    "residual_mode": mode,
                    "run_id": run_id,
                    "status": "completed",
                },
            )

    def test_collect_dps(self, tmp_path: Path) -> None:
        self._make_dps_tree(tmp_path)
        results = collect_dps(tmp_path)
        assert len(results) == 2
        modes = {r.residual_mode for r in results}
        assert modes == {"standard", "hyper_connection"}

    def test_collect_dps_legacy_layout(self, tmp_path: Path) -> None:
        """Legacy: <root>/results/<benchmark>/<run_id>/<mode>_dps.json."""
        _write_json(
            tmp_path / "results" / "depth_preservation" / "r1" / "standard_dps.json",
            {
                "residual_mode": "standard",
                "run_id": "r1",
                "dps_scores": [0.99, 0.98],
                "gps_scores": [0.97, 0.95],
                "dri": 0.985,
                "gpi": 0.96,
                "n_tokens": 50000,
            },
        )
        results = collect_dps(tmp_path)
        assert len(results) == 1
        assert results[0].dri == 0.985

    def test_collect_all_runs(self, tmp_path: Path) -> None:
        self._make_dps_tree(tmp_path)
        rows = collect_all_runs(tmp_path, "depth_preservation")
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "completed"

    def test_collect_niah(self, tmp_path: Path) -> None:
        for mode, success in [("standard", True), ("hyper_connection", False)]:
            _write_json(
                tmp_path / "natural_niah" / mode / "r1" / "niah_result.json",
                {
                    "mode": mode,
                    "success": success,
                    "generated_text": "abc",
                    "input_length": 100,
                },
            )
        results = collect_niah(tmp_path)
        assert len(results) == 2
        standard = next(r for r in results if r.residual_mode == "standard")
        assert standard.success is True

    def test_collect_latency(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / "inference_memory" / "r1" / "profile_results.json",
            {
                "standard": [
                    {"layers": 12, "peak_vram_mb": 100.0, "mean_latency_ms": 5.0, "p50_latency_ms": 5.0, "p99_latency_ms": 6.0, "tokens_per_second": 1000.0}
                ],  # noqa: E501
                "hyper_connection": [
                    {"layers": 12, "peak_vram_mb": 110.0, "mean_latency_ms": 6.0, "p50_latency_ms": 6.0, "p99_latency_ms": 7.0, "tokens_per_second": 950.0}
                ],  # noqa: E501
            },
        )
        profiles = collect_latency(tmp_path)
        assert len(profiles) == 2

    def test_collect_lm_summaries(self, tmp_path: Path) -> None:
        for mode in ["standard", "hyper_connection"]:
            _write_json(
                tmp_path / "lm_quality" / mode / "r1" / "metrics" / "summary.json",
                {
                    "benchmark_name": "lm_quality",
                    "residual_mode": mode,
                    "run_id": "r1",
                    "parameter_count": 1000000,
                    "elapsed_seconds": 100.0,
                    "global_step": 1000,
                    "peak_cuda_memory_mb": 4096.0,
                },
            )
        summaries = collect_lm_summaries(tmp_path, "lm_quality")
        assert len(summaries) == 2


class TestCollectCLI:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Build a small fake tree, run the CLI, verify output CSVs exist."""
        # Create dps data
        _write_json(
            tmp_path / "depth_preservation" / "standard" / "r1" / "dps.json",
            {
                "residual_mode": "standard",
                "run_id": "r1",
                "dps_scores": [0.99, 0.98, 0.97],
                "gps_scores": [0.95, 0.94, 0.93],
                "dri": 0.98,
                "gpi": 0.94,
                "n_tokens": 100000,
            },
        )
        _write_json(
            tmp_path / "depth_preservation" / "standard" / "r1" / "status.json",
            {"benchmark_name": "depth_preservation", "residual_mode": "standard", "run_id": "r1", "status": "completed"},
        )
        # Create lm_quality data
        _write_json(
            tmp_path / "lm_quality" / "standard" / "r1" / "metrics" / "summary.json",
            {
                "benchmark_name": "lm_quality",
                "residual_mode": "standard",
                "run_id": "r1",
                "parameter_count": 1000,
                "elapsed_seconds": 10.0,
                "global_step": 100,
                "peak_cuda_memory_mb": 256.0,
            },
        )

        out_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "-m", "scripts.ingest.collect", "--root", str(tmp_path), "--out", str(out_dir), "--benchmarks", "depth_preservation,lm_quality"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out_dir / "depth_preservation.csv").exists()
        assert (out_dir / "lm_quality_summary.csv").exists()
        assert (out_dir / "all_runs.csv").exists()
