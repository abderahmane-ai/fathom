"""Per-layer activation magnitude profiling for depth-growth analysis.

Captures the L2 norm of the residual stream ``h_l`` at the output of every
``TransformerLayer`` during the forward pass.  Used to demonstrate the O(L)
hidden-state growth of standard Pre-LN residuals and how alternative
mechanisms (RR, VEGA, AttnRes, mHC) bound it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class ActivationMagnitudeTracker:
    """Forward hook to record per-layer output L2 norms.

    Compatible with all residual modes that flow through the standard
    ``forward()`` path of ``TransformerLayer``. Block-based modes that use
    ``forward_attnres`` / ``forward_full_attnres`` are not supported because
    the residual state is the ``partial_block`` accumulator, not the layer's
    positional return value.

    Args:
        model: The transformer model. Must expose ``.layers`` as a ModuleList.
        every_n_steps: Logging cadence. 1 = every training step.
    """

    def __init__(self, model: nn.Module, every_n_steps: int = 1) -> None:
        self.model = model
        self.every_n_steps = max(1, int(every_n_steps))
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_norms: list[float | None] = []
        self._step: int = 0

    def attach(self) -> None:
        """Register a forward hook on every layer."""
        self._current_norms = [None] * len(self.model.layers)
        self._handles = [
            layer.register_forward_hook(self._make_hook(idx))
            for idx, layer in enumerate(self.model.layers)
        ]

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, idx: int) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if isinstance(output, tuple):
                if len(output) == 0:
                    return
                h = output[0]
            else:
                h = output
            if not isinstance(h, torch.Tensor):
                return
            self._current_norms[idx] = float(h.detach().norm().item())

        return hook

    def begin_step(self) -> None:
        """Clear per-step buffers and advance the step counter."""
        self._current_norms = [None] * len(self.model.layers)
        self._step += 1

    def should_log(self) -> bool:
        """Return True if the current step's norms should be reported."""
        return self._step % self.every_n_steps == 0

    def norms(self) -> list[float]:
        """Return a copy of the per-layer activation norms for the current step."""
        if all(val is None for val in self._current_norms):
            return []
        return [float(val) if val is not None else 0.0 for val in self._current_norms]

    def compute_metrics(self) -> dict[str, float]:
        """Aggregate the current step's per-layer norms into scalar metrics.

        Returns:
            Dictionary with keys: ``act_max_layer`` (index of the layer with
            the largest norm), ``act_max_norm``, ``act_min_norm``,
            ``act_mean_norm``, and ``act_growth_ratio`` = ``max/min``.
            All values are 0 when no norms were recorded.
        """
        norms = self.norms()
        if not norms:
            return {
                "act_max_layer": 0,
                "act_max_norm": 0.0,
                "act_min_norm": 0.0,
                "act_mean_norm": 0.0,
                "act_growth_ratio": 1.0,
            }
        max_idx = 0
        max_val = norms[0]
        for i, v in enumerate(norms):
            if v > max_val:
                max_val = v
                max_idx = i
        min_val = min(norms)
        return {
            "act_max_layer": int(max_idx),
            "act_max_norm": float(max_val),
            "act_min_norm": float(min_val),
            "act_mean_norm": float(sum(norms) / len(norms)),
            "act_growth_ratio": float(max_val / min_val) if min_val > 0 else 0.0,
        }
