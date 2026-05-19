"""Tests for benchmark artifact helpers."""
from __future__ import annotations

import json

from benchmarks.artifacts import (
    artifact_root,
    find_resume_checkpoint,
    status_path,
    write_status,
)


def test_write_status_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
    write_status("recurrent_residual", status="running", global_step=10)
    data = json.loads(status_path("recurrent_residual").read_text())
    assert data["status"] == "running"
    assert data["global_step"] == 10
    assert data["residual_mode"] == "recurrent_residual"


def test_find_resume_prefers_last_ckpt(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
    ckpt_dir = artifact_root() / "attnres_block" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    older = ckpt_dir / "step=000100.ckpt"
    older.write_text("old")
    last = ckpt_dir / "last.ckpt"
    last.write_text("last")
    assert find_resume_checkpoint("attnres_block") == str(last.resolve())
