"""Smoke tests for the plot scripts: build synthetic data, generate plots, check files exist."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_dps_curves(tmp_path: Path) -> None:
    """Build 2 fake dps.json files, run the plotter, check PNGs are produced."""
    artifact_root = tmp_path / "artifacts"
    for mode, dri in [("standard", 0.99), ("hyper_connection", 0.95)]:
        _write_json(artifact_root / "depth_preservation" / mode / "r1" / "dps.json", {
            "residual_mode": mode,
            "run_id": f"{mode}-r1",
            "dps_scores": [dri, dri - 0.01, dri - 0.02],
            "gps_scores": [dri - 0.05, dri - 0.06, dri - 0.07],
            "dri": dri,
            "gpi": dri - 0.06,
            "n_tokens": 100000,
        })

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.plots.dps_curves",
         "--artifact-root", str(artifact_root),
         "--out", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "dps_vs_layer.png").exists()
    assert (out_dir / "dps_vs_layer.pdf").exists()
    assert (out_dir / "gps_vs_layer.png").exists()


def test_scaling_pareto(tmp_path: Path) -> None:
    """Build a small summary CSV, run the plotter, check PNG produced."""
    csv_path = tmp_path / "summary.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,parameter_count,elapsed_seconds,global_step,peak_cuda_memory_mb",
        [
            "scaling_efficiency,standard,r1,1000000,10.0,100,256.0",
            "scaling_efficiency,standard,r2,2000000,20.0,100,512.0",
            "scaling_efficiency,hyper_connection,r1,1050000,11.0,100,256.0",
            "scaling_efficiency,hyper_connection,r2,2100000,22.0,100,512.0",
        ],
    )
    steps_path = tmp_path / "steps.csv"
    _write_csv(
        steps_path,
        "run_id,step,val_loss",
        ["r1,100,4.2", "r2,100,3.8", "r1,100,4.0", "r2,100,3.6"],
    )

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.plots.scaling_pareto",
         "--csv", str(csv_path), "--steps-csv", str(steps_path),
         "--out", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "pareto.png").exists()
    assert (out_dir / "pareto.pdf").exists()


def test_inference_memory(tmp_path: Path) -> None:
    csv_path = tmp_path / "inference.csv"
    rows = []
    for mode in ["standard", "hyper_connection"]:
        for L, vram, lat in [(12, 100, 5), (24, 200, 10), (48, 400, 20), (96, 800, 40)]:
            rows.append(f"{mode},{L},{vram},{lat}")
    _write_csv(
        csv_path,
        "residual_mode,layers,peak_vram_mb,p99_latency_ms,tokens_per_second",
        rows,
    )

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.plots.inference_memory",
         "--csv", str(csv_path), "--out", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "vram_vs_depth.png").exists()
    assert (out_dir / "latency_vs_depth.png").exists()
    assert (out_dir / "throughput_vs_depth.png").exists()


def test_iso_flop(tmp_path: Path) -> None:
    csv_path = tmp_path / "iso.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,parameter_count,elapsed_seconds,global_step,peak_cuda_memory_mb",
        [
            "iso_flop,wide_shallow_std,r1,1000000,10.0,100,256.0",
            "iso_flop,narrow_deep_vega,r1,1000000,12.0,100,256.0",
            "iso_flop,narrow_deep_rr,r1,1000000,11.0,100,256.0",
        ],
    )
    steps_path = tmp_path / "steps.csv"
    _write_csv(
        steps_path,
        "run_id,step,val_loss",
        ["r1,100,4.2", "r1,100,3.9", "r1,100,4.0"],
    )

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.plots.iso_flop",
         "--csv", str(csv_path), "--steps-csv", str(steps_path),
         "--out", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "iso_flop.png").exists()
