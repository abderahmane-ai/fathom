"""Tests for benchmarks.common.run_metadata and benchmarks.common.jsonl_logger."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.common.jsonl_logger import JsonlLogger
from benchmarks.common.run_metadata import (
    WallClock,
    capture_run_metadata,
    format_duration,
    get_environment_info,
    get_git_commit,
    get_git_status,
    get_gpu_info,
    log_run_banner,
    log_run_finish,
)


class TestGetGitCommit:
    def test_returns_short_hash(self) -> None:
        """When in a git repo, returns a 12-char short hash."""
        result = get_git_commit(short=True)
        assert result is not None
        assert len(result) >= 7  # git short hash is 7-12 chars

    def test_returns_full_hash(self) -> None:
        """When short=False, returns 40-char full hash."""
        result = get_git_commit(short=False)
        assert result is not None
        assert len(result) == 40

    def test_handles_missing_git(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When git is not found, returns None instead of crashing."""
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
        assert get_git_commit(repo_root=tmp_path) is None


class TestGetGitStatus:
    def test_returns_string_or_none(self) -> None:
        """Returns the status --short output, or None on failure."""
        result = get_git_status()
        # In a real git repo, this is a string (possibly empty).  Either is OK.
        assert result is None or isinstance(result, str)

    def test_handles_missing_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
        assert get_git_status() is None


class TestGetGpuInfo:
    def test_returns_dict_with_expected_keys(self) -> None:
        info = get_gpu_info()
        assert set(info.keys()) == {"name", "total_memory_mb", "cuda_version", "driver_version"}
        # All values are None or strings/ints
        for value in info.values():
            assert value is None or isinstance(value, (str, int))


class TestGetEnvironmentInfo:
    def test_has_required_keys(self) -> None:
        env = get_environment_info()
        for key in ("hostname", "platform", "python_version", "torch_version", "pid"):
            assert key in env
        assert isinstance(env["pid"], int)


class TestCaptureRunMetadata:
    def test_basic_fields_present(self) -> None:
        m = capture_run_metadata(
            benchmark_name="lm_quality",
            run_id="lm-20260101T000000Z",
            residual_mode="hyper_connection",
        )
        for key in ("benchmark_name", "run_id", "residual_mode", "started_at", "git_commit", "environment", "gpu"):
            assert key in m
        assert m["benchmark_name"] == "lm_quality"
        assert m["residual_mode"] == "hyper_connection"

    def test_serializable_to_json(self) -> None:
        """The full dict must be JSON-serializable (this is what we write to disk)."""
        m = capture_run_metadata(
            benchmark_name="lm_quality",
            run_id="r1",
            residual_mode="standard",
            seed=42,
        )
        json.dumps(m, default=str)  # must not raise

    def test_with_omegaconfig(self) -> None:
        """If config has .to_container(), that path is taken."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {"d_model": 768}, "data": {"batch_size": 32}})
        m = capture_run_metadata(
            benchmark_name="lm_quality",
            run_id="r2",
            residual_mode="standard",
            config=cfg,
        )
        assert m["config"]["model"]["d_model"] == 768
        assert m["config"]["data"]["batch_size"] == 32

    def test_with_plain_dict(self) -> None:
        m = capture_run_metadata(
            benchmark_name="x",
            run_id="r",
            residual_mode="y",
            config={"key": "value"},
        )
        assert m["config"] == {"key": "value"}

    def test_extras_are_merged(self) -> None:
        m = capture_run_metadata(
            benchmark_name="x",
            run_id="r",
            residual_mode="y",
            extra={"foo": 1, "bar": "baz"},
        )
        assert m["extra"] == {"foo": 1, "bar": "baz"}


class TestLogRunBanner:
    def test_emits_eight_lines(self, caplog: pytest.LogCaptureFixture) -> None:
        """The banner is exactly 8 log.info calls: 3 '=' lines + 4 content + 1 closing '='."""
        import logging

        caplog.set_level(logging.INFO)
        log = logging.getLogger("test_banner")
        m = capture_run_metadata(benchmark_name="b", run_id="r", residual_mode="m", seed=42)
        log_run_banner(log, m)
        assert sum(1 for rec in caplog.records if rec.levelno == logging.INFO) == 8


class TestLogRunFinish:
    def test_emits_lines_with_extras(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        caplog.set_level(logging.INFO)
        log = logging.getLogger("test_finish")
        log_run_finish(log, status="completed", elapsed_seconds=3725.0, peak_mem=4096)
        msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("status=completed" in m for m in msgs)
        assert any("peak_mem" in m and "4096" in m for m in msgs)


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (1, "1s"),
            (59, "59s"),
            (60, "1m00s"),
            (61, "1m01s"),
            (3600, "1h00m00s"),
            (3725, "1h02m05s"),
        ],
    )
    def test_formats_correctly(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected


class TestWallClock:
    def test_elapsed_is_positive(self) -> None:
        import time

        clock = WallClock()
        time.sleep(0.01)
        assert clock.elapsed() > 0


class TestJsonlLogger:
    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "events.jsonl"
        logger = JsonlLogger(path, run_id="r1")
        logger.emit("start")
        logger.close()
        assert path.exists()

    def test_writes_one_json_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlLogger(path, run_id="r1") as log:
            log.emit("step", step=1, loss=4.2)
            log.emit("step", step=2, loss=3.8)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "ts" in record
            assert "event" in record
            assert record["run_id"] == "r1"
        assert json.loads(lines[0])["loss"] == 4.2
        assert json.loads(lines[1])["loss"] == 3.8

    def test_handles_non_serializable(self, tmp_path: Path) -> None:
        """Non-JSON values are coerced to repr, not raising."""
        path = tmp_path / "events.jsonl"
        with JsonlLogger(path) as log:
            log.emit("weird", value=object())
        record = json.loads(path.read_text().strip())
        assert "value" in record
        assert "object at 0x" in record["value"]

    def test_n_events_increments(self, tmp_path: Path) -> None:
        log = JsonlLogger(tmp_path / "e.jsonl")
        assert log.n_events == 0
        log.emit("a")
        log.emit("b")
        log.emit("c")
        assert log.n_events == 3
        log.close()

    def test_run_id_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "e.jsonl"
        with JsonlLogger(path) as log:
            log.emit("start")
        record = json.loads(path.read_text().strip())
        assert "run_id" not in record
