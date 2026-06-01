"""Tests for benchmarks.common.artifacts: write_run_metadata and find_all_runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common.artifacts import (
    benchmark_dir,
    find_all_runs,
    run_dir,
    run_metadata_path,
    write_run_metadata,
    write_status,
)


class TestWriteRunMetadata:
    def test_creates_file_with_all_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        metadata = {
            "benchmark_name": "lm_quality",
            "run_id": "r1",
            "residual_mode": "hyper_connection",
            "started_at": "2026-01-01T00:00:00+00:00",
            "git_commit": "abc1234",
            "config": {"d_model": 768},
        }
        write_run_metadata("lm_quality", "hyper_connection", "r1", metadata=metadata)
        path = run_metadata_path("lm_quality", "hyper_connection", "r1")
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["git_commit"] == "abc1234"
        assert payload["config"]["d_model"] == 768
        assert payload["benchmark_name"] == "lm_quality"

    def test_merges_additional_fields(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        write_run_metadata("lm_quality", "standard", "r1", metadata={"foo": 1})
        write_run_metadata("lm_quality", "standard", "r1", ended_at="2026-01-01T01:00:00+00:00")
        payload = json.loads(run_metadata_path("lm_quality", "standard", "r1").read_text())
        assert payload["foo"] == 1
        assert payload["ended_at"] == "2026-01-01T01:00:00+00:00"

    def test_overwrites_corrupt_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        path = run_metadata_path("lm_quality", "standard", "r1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json")
        write_run_metadata("lm_quality", "standard", "r1", metadata={"recovered": True})
        payload = json.loads(path.read_text())
        assert payload["recovered"] is True


class TestFindAllRuns:
    def test_empty_directory(self, tmp_path: Path) -> None:
        assert find_all_runs("lm_quality", root=tmp_path) == []

    def test_finds_runs_across_modes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        write_status("lm_quality", "standard", "r1", status="completed")
        write_status("lm_quality", "hyper_connection", "r2", status="failed", error="OOM")
        runs = find_all_runs("lm_quality", root=tmp_path)
        assert len(runs) == 2
        modes = {r["residual_mode"] for r in runs}
        assert modes == {"standard", "hyper_connection"}

    def test_run_record_has_path_metadata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        write_status("lm_quality", "standard", "r1", status="completed")
        runs = find_all_runs("lm_quality", root=tmp_path)
        assert len(runs) == 1
        assert runs[0]["_path"].endswith("status.json")
        assert runs[0]["_run_dir"].endswith("r1")


class TestRunMetadataPath:
    def test_path_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_ARTIFACT_ROOT", str(tmp_path))
        path = run_metadata_path("lm_quality", "hyper_connection", "r1")
        assert path == tmp_path / "lm_quality" / "hyper_connection" / "r1" / "run.json"
