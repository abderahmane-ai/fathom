"""Unit tests for benchmarks/common/grad_norms.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from benchmarks.common.grad_norms import PerLayerGradTracker, gini_coefficient


@pytest.fixture
def tiny_model(d_model):
    """Minimal module exposing ``.layers`` as a ModuleList of Linear stacks."""

    class _Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.proj = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x) + x

    return nn.ModuleList([_Block(d_model) for _ in range(4)])


class TestGiniCoefficient:
    def test_empty_returns_zero(self):
        assert gini_coefficient([]) == 0.0

    def test_uniform_returns_zero(self):
        assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_all_zero_returns_zero(self):
        assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0

    def test_single_element_returns_zero(self):
        assert gini_coefficient([0.42]) == 0.0

    def test_known_skewed_value(self):
        """With one dominant value and the rest zero, gini approaches 1.0."""
        g = gini_coefficient([100.0, 0.0, 0.0, 0.0])
        assert g == pytest.approx(0.75, abs=1e-9)

    def test_known_linear_progression(self):
        """Sorted [0, 1, 2, 3] yields a known, hand-checkable gini.

        Hand-computed: ((2*1-5)*0 + (2*2-5)*1 + (2*3-5)*2 + (2*4-5)*3) / (4*6)
        = (-3*0 + -1*1 + 1*2 + 3*3) / 24 = 10 / 24.
        """
        g = gini_coefficient([0.0, 1.0, 2.0, 3.0])
        assert g == pytest.approx(10.0 / 24.0, abs=1e-9)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gini_coefficient([-1.0, 1.0])

    def test_invariant_to_scale(self):
        """Gini is homogeneous of degree zero: scaling all values leaves gini unchanged."""
        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 30.0, 40.0]
        assert gini_coefficient(a) == pytest.approx(gini_coefficient(b), abs=1e-12)

    def test_invariant_to_reordering(self):
        """Gini is symmetric: input order does not matter."""
        a = [3.0, 1.0, 4.0, 1.0, 5.0]
        b = [1.0, 5.0, 1.0, 3.0, 4.0]
        assert gini_coefficient(a) == pytest.approx(gini_coefficient(b), abs=1e-12)

    def test_bounded_zero_one(self):
        rng = torch.Generator().manual_seed(0)
        for _ in range(20):
            v = torch.rand(50, generator=rng).tolist()
            g = gini_coefficient(v)
            assert 0.0 <= g <= 1.0


class _WrapperModel(nn.Module):
    """Wraps a ModuleList of layers in a module with a ``.layers`` attribute."""

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestPerLayerGradTracker:
    def test_records_one_norm_per_layer(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        try:
            x = torch.randn(2, 4, d_model, requires_grad=True)
            out = model(x)
            out.sum().backward()
            assert len(tracker.norms()) == len(tiny_model)
        finally:
            tracker.detach()

    def test_begin_step_clears_buffer(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        try:
            tracker.begin_step()
            x = torch.randn(2, 4, d_model, requires_grad=True)
            model(x).sum().backward()
            assert len(tracker.norms()) == 4
            tracker.begin_step()
            assert tracker.norms() == []
        finally:
            tracker.detach()

    def test_compute_metrics_basic(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        try:
            x = torch.randn(2, 4, d_model, requires_grad=True)
            model(x).sum().backward()
            m = tracker.compute_metrics()
            assert set(m.keys()) == {
                "grad_gini",
                "grad_max_layer",
                "grad_max_norm",
                "grad_min_norm",
                "grad_mean_norm",
            }
            assert 0.0 <= m["grad_gini"] <= 1.0
            assert 0 <= m["grad_max_layer"] < 4
            assert m["grad_max_norm"] >= m["grad_min_norm"]
        finally:
            tracker.detach()

    def test_compute_metrics_empty(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        m = tracker.compute_metrics()
        assert m["grad_gini"] == 0.0
        assert m["grad_max_norm"] == 0.0

    def test_should_log_cadence(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model, every_n_steps=3)
        tracker.begin_step()
        assert tracker.should_log() is False
        tracker.begin_step()
        assert tracker.should_log() is False
        tracker.begin_step()
        assert tracker.should_log() is True
        tracker.begin_step()
        assert tracker.should_log() is False

    def test_detach_is_idempotent(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        tracker.detach()
        tracker.detach()
        assert tracker._handles == []

    def test_detach_stops_recording(self, d_model, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        tracker.detach()
        x = torch.randn(2, 4, d_model, requires_grad=True)
        model(x).sum().backward()
        assert tracker.norms() == []

    def test_every_n_steps_clamped_to_one(self, tiny_model):
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model, every_n_steps=0)
        assert tracker.every_n_steps == 1
        tracker2 = PerLayerGradTracker(model, every_n_steps=-5)
        assert tracker2.every_n_steps == 1

    def test_norms_returns_copy(self, d_model, tiny_model):
        """Mutating the returned list must not corrupt the tracker's buffer."""
        model = _WrapperModel(tiny_model)
        tracker = PerLayerGradTracker(model)
        tracker.attach()
        try:
            x = torch.randn(2, 4, d_model, requires_grad=True)
            model(x).sum().backward()
            snapshot = tracker.norms()
            snapshot.append(999.0)
            assert len(tracker.norms()) == 4
        finally:
            tracker.detach()
