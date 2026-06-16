"""Metric helpers for benchmark runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


class ThroughputMeter:
    """Track elapsed time and token throughput.

    Args:
        tokens_per_step: Number of tokens processed per optimizer step.
    """

    def __init__(self, tokens_per_step: int) -> None:
        self.tokens_per_step = tokens_per_step
        self.started_at = time.perf_counter()

    def tokens_per_second(self, step: int) -> float:
        """Compute throughput.

        Args:
            step: Completed optimizer steps.

        Returns:
            Tokens per second since construction.
        """
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return float(step * self.tokens_per_step / elapsed)


def peak_cuda_memory_mb() -> float:
    """Return peak CUDA memory allocated in MiB.

    Returns:
        Peak allocated memory, or ``0.0`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload atomically.

    Args:
        path: Destination path.
        payload: JSON-serializable dictionary.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def compute_dps_closed_form(
    xtx: torch.Tensor,
    xty: torch.Tensor,
    yty: torch.Tensor,
    target_variance: torch.Tensor,
    lambda_val: float = 1.0,
) -> float:
    """Compute Depth Preservation Score (DPS) via closed-form Ridge Regression.

    Solves the ridge problem on pre-accumulated covariance matrices and returns R².

    Args:
        xtx: The accumulated covariance matrix [X, 1]^T [X, 1] of shape (d+1, d+1).
        xty: The accumulated cross-covariance matrix [X, 1]^T Y of shape (d+1, d).
        yty: The accumulated sum of squares of Y (Tr(Y^T Y)).
        target_variance: The accumulated sum of squared differences of the target
            from its mean, sum( ||a_k^(i) - mean(a_k)||^2 ).
        lambda_val: Ridge regularization strength.

    Returns:
        The Depth Preservation Score (DPS) as a float (R-squared).
    """
    d = xtx.size(0) - 1

    # Add ridge penalty to the diagonal (excluding the bias term at the end)
    reg_matrix = torch.eye(d + 1, device=xtx.device, dtype=xtx.dtype)
    reg_matrix[-1, -1] = 0.0  # Do not penalize the bias

    # Solve for W_tilde = [W; b]
    # W_tilde = (X^T X + lambda I)^{-1} X^T Y
    w_tilde = torch.linalg.solve(xtx + lambda_val * reg_matrix, xty)

    # Calculate Residual Sum of Squares (RSS) using the identity:
    # RSS = Tr(Y^T Y) - 2 * Tr(W_tilde^T X^T Y) + Tr(W_tilde^T X^T X W_tilde)
    rss_term1 = yty
    rss_term2 = -2.0 * torch.trace(w_tilde.T @ xty)
    rss_term3 = torch.trace(w_tilde.T @ xtx @ w_tilde)
    rss = rss_term1 + rss_term2 + rss_term3

    # Calculate R-squared
    dps = 1.0 - (rss / target_variance)

    return float(dps.item())


def calculate_dri(dps_scores: list[float]) -> float:
    """Calculate the Dilution Resistance Index (DRI).

    The DRI is the average of the DPS scores over the first half of the network.

    Args:
        dps_scores: List of DPS scores, ordered by layer depth (1 to L-1).

    Returns:
        The DRI scalar value.
    """
    if not dps_scores:
        return 0.0

    # L is total layers. dps_scores contains L-1 elements (layer 1 to L-1).
    # So total layers L = len(dps_scores) + 1.
    l_total = len(dps_scores) + 1
    half_l = l_total // 2

    if half_l == 0:
        return 0.0

    # Average over the first half (layers 1 to floor(L/2))
    # Indices in dps_scores are 0-based, so index 0 is layer 1.
    # We want indices 0 up to half_l - 1.
    first_half_scores = dps_scores[:half_l]
    dri = sum(first_half_scores) / len(first_half_scores)

    return dri


def calculate_gpi(gps_scores: list[float]) -> float:
    """Calculate the Gradient Preservation Index (GPI).

    The GPI is the average of the GPS scores over the first half of the network.

    Args:
        gps_scores: List of GPS scores, ordered by layer depth (1 to L-1).

    Returns:
        The GPI scalar value.
    """
    if not gps_scores:
        return 0.0

    # L is total layers. gps_scores contains L-1 elements (layer 1 to L-1).
    l_total = len(gps_scores) + 1
    half_l = l_total // 2

    if half_l == 0:
        return 0.0

    # Average over the first half (layers 1 to floor(L/2))
    # Indices in gps_scores are 0-based, so index 0 is layer 1.
    # We want indices 0 up to half_l - 1.
    first_half_scores = gps_scores[:half_l]
    gpi = sum(first_half_scores) / len(first_half_scores)

    return gpi
