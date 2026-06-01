"""Unit tests for benchmarks/common/activation_profile.py."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from benchmarks.common.activation_profile import ActivationMagnitudeTracker


@pytest.fixture
def tiny_model(d_model):
    class _Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.proj = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x) + x

    return nn.ModuleList([_Block(d_model) for _ in range(4)])


class _WrapperModel(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestActivationMagnitudeTracker:
    def test_records_one_norm_per_layer(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model)
        tracker.attach()
        try:
            tracker.begin_step()
            x = torch.randn(2, 4, d_model)
            model(x)
            assert len(tracker.norms()) == len(tiny_model)
        finally:
            tracker.detach()

    def test_compute_metrics_growth_ratio(self, d_model, tiny_model):
        """A model that doubles the activation norm each layer should report the
        ratio between the largest and smallest layer norms."""
        model = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        wrapper = _WrapperModel(model)
        tracker = ActivationMagnitudeTracker(wrapper)
        tracker.attach()
        try:
            tracker.begin_step()
            x = torch.ones(1, 1, d_model)
            for layer in model:
                x = layer(x) * 2.0
            metrics = tracker.compute_metrics()
            assert metrics["act_growth_ratio"] > 1.0
        finally:
            tracker.detach()

    def test_compute_metrics_empty(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model)
        m = tracker.compute_metrics()
        assert m["act_max_norm"] == 0.0
        assert m["act_growth_ratio"] == 1.0

    def test_growth_ratio_handles_zero_min(self, tiny_model):
        """If one layer's norm is exactly zero, growth ratio should not divide by zero."""
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model)
        tracker.attach()
        try:
            tracker.begin_step()
            x = torch.zeros(1, 1, 64)
            model(x)
            metrics = tracker.compute_metrics()
            assert math.isfinite(metrics["act_growth_ratio"]) or metrics["act_growth_ratio"] == 0.0
        finally:
            tracker.detach()

    def test_begin_step_clears_buffer(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model)
        tracker.attach()
        try:
            tracker.begin_step()
            model(torch.randn(2, 4, d_model))
            assert len(tracker.norms()) == 4
            tracker.begin_step()
            assert tracker.norms() == []
        finally:
            tracker.detach()

    def test_detach_is_idempotent(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model)
        tracker.attach()
        tracker.detach()
        tracker.detach()
        assert tracker._handles == []

    def test_should_log_cadence(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model, every_n_steps=2)
        tracker.begin_step()
        assert tracker.should_log() is False
        tracker.begin_step()
        assert tracker.should_log() is True

    def test_every_n_steps_clamped_to_one(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = ActivationMagnitudeTracker(model, every_n_steps=0)
        assert tracker.every_n_steps == 1
