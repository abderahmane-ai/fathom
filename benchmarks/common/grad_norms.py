"""Per-layer gradient norm tracking for PreNorm dilution analysis.

Records the L2 norm of the gradient flowing into each ``TransformerLayer`` at
every training step and computes a Gini coefficient measuring how unevenly
gradient mass is distributed across depth.

Math:
    gini = (sum_i (2i - n - 1) * sorted_norms[i]) / (n * sum(sorted_norms))

    gini = 0 -> all layers receive the same gradient magnitude
    gini -> 1 -> one layer receives all the gradient mass

Pre-LN transformers typically show high gini (one or two late layers dominate
the gradient signal) because of the O(L) growth of hidden-state magnitude.
Alternative residual mechanisms aim to flatten this profile.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a non-negative sequence.

    Args:
        values: Non-negative values, e.g. gradient norms per layer.

    Returns:
        Gini coefficient in [0, 1]. Returns 0.0 for an empty sequence or one
        whose values sum to zero.
    """
    if not values:
        return 0.0
    arr = torch.tensor(values, dtype=torch.float64)
    if (arr < 0).any():
        raise ValueError("Gini coefficient requires non-negative values.")
    total = arr.sum().item()
    if total == 0.0:
        return 0.0
    arr, _ = arr.sort()
    n = arr.numel()
    index = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2.0 * index - n - 1.0) * arr).sum().item() / (n * total))


class PerLayerGradTracker:
    """Hooks a ``TransformerDecoder`` to record per-layer input gradient norms.

    Compatible with residual modes whose layer forward takes a single tensor
    as its first argument (``standard``, ``recurrent_residual``, ``vega``,
    ``vega``). Block-based modes (``block_attnres``,
    ``full_attnres``) pass a Python list as first argument and are silently
    skipped: their per-layer gradient is not directly comparable.

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
        """Register a backward hook on every layer."""
        self._current_norms = [None] * len(self.model.layers)
        self._handles = [layer.register_full_backward_hook(self._make_hook(idx)) for idx, layer in enumerate(self.model.layers)]

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, idx: int) -> Callable[[nn.Module, tuple[Any, ...], tuple[Any, ...]], None]:
        def hook(
            _module: nn.Module,
            grad_input: tuple[Any, ...],
            _grad_output: tuple[Any, ...],
        ) -> None:
            if not grad_input or grad_input[0] is None:
                return
            if not isinstance(grad_input[0], torch.Tensor):
                return
            self._current_norms[idx] = float(grad_input[0].detach().norm().item())

        return hook

    def begin_step(self) -> None:
        """Clear per-step buffers and advance the step counter."""
        self._current_norms = [None] * len(self.model.layers)
        self._step += 1

    def should_log(self) -> bool:
        """Return True if the current step's norms should be reported."""
        return self._step % self.every_n_steps == 0

    def norms(self) -> list[float]:
        """Return a copy of the per-layer gradient norms for the current step."""
        if all(val is None for val in self._current_norms):
            return []
        return [float(val) if val is not None else 0.0 for val in self._current_norms]

    def compute_metrics(self) -> dict[str, float]:
        """Aggregate the current step's per-layer norms into scalar metrics.

        Returns:
            Dictionary with keys: ``grad_gini`` (Gini coefficient of norms),
            ``grad_max_layer`` (index of the layer with the largest gradient),
            ``grad_max_norm``, ``grad_min_norm``, ``grad_mean_norm``.
            All values are 0 when no norms were recorded.
        """
        norms = self.norms()
        if not norms:
            return {
                "grad_gini": 0.0,
                "grad_max_layer": 0,
                "grad_max_norm": 0.0,
                "grad_min_norm": 0.0,
                "grad_mean_norm": 0.0,
            }
        max_idx = 0
        max_val = norms[0]
        for i, v in enumerate(norms):
            if v > max_val:
                max_val = v
                max_idx = i
        return {
            "grad_gini": gini_coefficient(norms),
            "grad_max_layer": int(max_idx),
            "grad_max_norm": float(max_val),
            "grad_min_norm": float(min(norms)),
            "grad_mean_norm": float(sum(norms) / len(norms)),
        }
