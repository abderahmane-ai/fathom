"""Tests for benchmark configs, paths, docs, and Modal entrypoints."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from benchmarks.common.artifacts import (
    artifact_root,
    checkpoint_dir,
    find_resume_checkpoint,
    resolve_val_check_interval,
    status_path,
    write_status,
)
from benchmarks.common.configs import benchmark_modes, load_benchmark_config
from benchmarks.common.param_count import assert_model_under_cap

BENCHMARKS = ("lm_quality", "depth_needle", "scaling_efficiency")


def test_write_status_roundtrip(tmp_path, monkeypatch):
    """Status files should live under benchmark/mode/run_id."""
    monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
    write_status("lm_quality", "recurrent_residual", "run-a", status="running", global_step=10)
    data = json.loads(status_path("lm_quality", "recurrent_residual", "run-a").read_text())
    assert data["status"] == "running"
    assert data["global_step"] == 10
    assert data["benchmark_name"] == "lm_quality"
    assert data["residual_mode"] == "recurrent_residual"


def test_find_resume_prefers_last_ckpt(tmp_path, monkeypatch):
    """Checkpoint discovery should prefer Lightning's last checkpoint."""
    monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
    ckpt_dir = checkpoint_dir("lm_quality", "block_attnres", "run-a")
    ckpt_dir.mkdir(parents=True)
    older = ckpt_dir / "step=000100.ckpt"
    older.write_text("old", encoding="utf-8")
    last = ckpt_dir / "last.ckpt"
    last.write_text("last", encoding="utf-8")
    assert find_resume_checkpoint("lm_quality", "block_attnres", "run-a") == str(last.resolve())


def test_val_check_interval_caps_when_too_large():
    """Integer validation intervals larger than the epoch should become once-per-epoch."""
    assert resolve_val_check_interval(287, 500) == 1.0


def test_val_check_interval_keeps_small_int():
    """Valid integer validation intervals should pass through unchanged."""
    assert resolve_val_check_interval(287, 100) == 100


def test_val_check_interval_keeps_fraction():
    """Fractional validation intervals should pass through unchanged."""
    assert resolve_val_check_interval(287, 0.25) == 0.25


@pytest.mark.parametrize("benchmark_name", BENCHMARKS)
def test_benchmark_configs_load(benchmark_name):
    """Each benchmark folder must provide a loadable config."""
    cfg = load_benchmark_config(benchmark_name)
    assert cfg.benchmark.max_params == 60_000_000
    assert "standard" in benchmark_modes(cfg)
    assert len(benchmark_modes(cfg)) >= 2  # at least two modes for comparison


@pytest.mark.parametrize("benchmark_name", BENCHMARKS)
def test_benchmark_readme_documents_contract(benchmark_name):
    """Each benchmark README should state purpose, modes, metrics, run command, and paths."""
    path = Path("benchmarks") / benchmark_name / "README.md"
    text = path.read_text(encoding="utf-8")
    for required in ("Purpose", "Modes", "Metrics", "Run", "Artifacts"):
        assert required in text
    assert "modal run" in text
    assert "status.json" in text


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.lm_quality.modal_lm_quality",
        "benchmarks.depth_needle.modal_depth_needle",
        "benchmarks.scaling_efficiency.modal_scaling_efficiency",
    ],
)
def test_modal_scripts_import(module_name):
    """Modal entrypoint modules should import without launching remote work."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "app")


def test_parameter_cap_rejects_oversized_model():
    """The parameter guard should fail before a benchmark exceeds the cap."""
    cfg = load_benchmark_config("lm_quality").model
    with pytest.raises(ValueError, match="exceeding cap"):
        assert_model_under_cap(cfg, max_params=1)


def test_artifact_root_uses_environment(tmp_path, monkeypatch):
    """Artifact root should honor BENCHMARK_ARTIFACT_ROOT."""
    monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
    assert artifact_root() == tmp_path.resolve()
