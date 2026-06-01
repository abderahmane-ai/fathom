"""Smoke tests for the markdown table scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_dps_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "dps.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,dri,gpi,n_tokens,n_layers,dps_mean,gps_mean,dps_min,dps_max",
        [
            "depth_preservation,standard,r1,0.99,0.95,100000,12,0.97,0.93,0.85,0.99",
            "depth_preservation,hyper_connection,r1,0.95,0.90,100000,12,0.93,0.88,0.80,0.95",
        ],
    )
    out = tmp_path / "SUMMARY.md"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tables.dps_table", "--csv", str(csv_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    text = out.read_text()
    assert "Depth Preservation Summary" in text
    assert "standard" in text
    assert "hyper_connection" in text
    assert "0.99" in text  # standard DRI
    assert "**0.99**" in text or "**0.9900**" in text  # winner bolded


def test_lm_quality_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,parameter_count,elapsed_seconds,global_step,peak_cuda_memory_mb",
        ["lm_quality,standard,r1,1000000,10.0,100,256.0", "lm_quality,hyper_connection,r1,1050000,11.0,100,256.0"],
    )
    steps_path = tmp_path / "steps.csv"
    _write_csv(
        steps_path,
        "benchmark_name,residual_mode,run_id,step,val_loss",
        ["lm_quality,standard,r1,100,4.2", "lm_quality,hyper_connection,r1,100,3.9"],
    )
    out = tmp_path / "SUMMARY.md"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tables.lm_quality_table", "--csv", str(csv_path), "--steps-csv", str(steps_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    text = out.read_text()
    assert "LM Quality Summary" in text
    assert "3.9" in text  # val_loss of hyper_connection
    assert "3.9000" in text or "**3.9**" in text or "**3.9000**" in text


def test_ablation_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,parameter_count,elapsed_seconds,global_step,peak_cuda_memory_mb",
        [
            "ablation,vega,r1,1000000,10.0,100,256.0",
            "ablation,vega_no_var_reg,r1,1000000,10.0,100,256.0",
            "ablation,rr_no_depth_biases,r1,1000000,10.0,100,256.0",
        ],
    )
    steps_path = tmp_path / "steps.csv"
    _write_csv(
        steps_path,
        "benchmark_name,residual_mode,run_id,step,val_loss",
        ["ablation,vega,r1,100,4.2", "ablation,vega_no_var_reg,r1,100,4.5", "ablation,rr_no_depth_biases,r1,100,4.1"],
    )
    out = tmp_path / "SUMMARY.md"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tables.ablation_table", "--csv", str(csv_path), "--steps-csv", str(steps_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    text = out.read_text()
    assert "Ablation Summary" in text
    assert "vega_no_var_reg" in text
    assert "Δ val_loss" in text


def test_niah_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "niah.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,success,generated_text,input_length",
        [
            "natural_niah,standard,r1,True, 84729,350",
            "natural_niah,hyper_connection,r1,False, 12345,350",
        ],
    )
    out = tmp_path / "SUMMARY.md"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tables.niah_table", "--csv", str(csv_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    text = out.read_text()
    assert "Needle" in text
    assert "✓" in text
    assert "✗" in text


def test_iso_flop_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "iso.csv"
    _write_csv(
        csv_path,
        "benchmark_name,residual_mode,run_id,parameter_count,elapsed_seconds,global_step,peak_cuda_memory_mb",
        [
            "iso_flop,wide_shallow_std,r1,6000000,10.0,100,256.0",
            "iso_flop,narrow_deep_vega,r1,6000000,12.0,100,256.0",
        ],
    )
    steps_path = tmp_path / "steps.csv"
    _write_csv(
        steps_path,
        "benchmark_name,residual_mode,run_id,step,val_loss",
        ["iso_flop,wide_shallow_std,r1,100,4.2", "iso_flop,narrow_deep_vega,r1,100,3.8"],
    )
    out = tmp_path / "SUMMARY.md"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tables.iso_flop_table", "--csv", str(csv_path), "--steps-csv", str(steps_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    text = out.read_text()
    assert "Iso-FLOP Summary" in text
    assert "wide-shallow" in text
    assert "narrow-deep" in text
