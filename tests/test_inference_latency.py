"""Unit tests for benchmarks/common/inference_latency.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from benchmarks.common.inference_latency import LatencyResult, profile_forward


class _TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))


class TestLatencyResult:
    def test_to_dict_round_trip(self):
        result = LatencyResult(
            mean_ms=10.0,
            p50_ms=9.5,
            p99_ms=15.0,
            min_ms=8.0,
            max_ms=20.0,
            stddev_ms=2.5,
            tokens_per_second=1000.0,
            peak_vram_mb=512.0,
        )
        d = result.to_dict()
        assert d["mean_ms"] == 10.0
        assert d["tokens_per_second"] == 1000.0
        assert d["peak_vram_mb"] == 512.0

    def test_dataclass_field_count(self):
        result = LatencyResult(
            mean_ms=1.0,
            p50_ms=1.0,
            p99_ms=1.0,
            min_ms=1.0,
            max_ms=1.0,
            stddev_ms=0.0,
            tokens_per_second=1.0,
            peak_vram_mb=0.0,
        )
        assert len(result.to_dict()) == 8


class TestProfileForward:
    def test_runs_on_cpu(self):
        """Smoke test on CPU so the test suite has no GPU dependency."""
        model = _TinyModel()
        result = profile_forward(
            model,
            batch_size=2,
            seq_len=8,
            vocab_size=64,
            n_warmup=1,
            n_runs=3,
            device="cpu",
        )
        assert result.mean_ms > 0
        assert result.p50_ms > 0
        assert result.min_ms <= result.mean_ms <= result.max_ms
        assert result.tokens_per_second > 0
        assert result.peak_vram_mb == 0.0
        assert result.stddev_ms >= 0

    def test_n_runs_clamped(self):
        model = _TinyModel()
        result = profile_forward(
            model,
            batch_size=1,
            seq_len=4,
            vocab_size=16,
            n_warmup=0,
            n_runs=0,
            device="cpu",
        )
        assert result.mean_ms > 0

    def test_percentile_ordering(self):
        """Mean must lie between min and max; p50 must be in the data range."""
        model = _TinyModel()
        result = profile_forward(
            model,
            batch_size=1,
            seq_len=4,
            vocab_size=16,
            n_warmup=1,
            n_runs=5,
            device="cpu",
        )
        assert result.min_ms <= result.p50_ms <= result.max_ms
        assert result.min_ms <= result.p99_ms <= result.max_ms

    def test_tokens_per_second_matches_mean(self):
        """tokens_per_second = (batch * seq) / (mean_ms / 1000)."""
        model = _TinyModel()
        batch = 4
        seq = 8
        result = profile_forward(
            model,
            batch_size=batch,
            seq_len=seq,
            vocab_size=16,
            n_warmup=1,
            n_runs=3,
            device="cpu",
        )
        expected = (batch * seq) / (result.mean_ms / 1000.0)
        assert result.tokens_per_second == pytest.approx(expected, rel=1e-9)
