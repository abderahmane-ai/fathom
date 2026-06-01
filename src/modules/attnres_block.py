"""Attention Residual modules — depth-wise aggregation over block history.

Reference: "Attention Residuals", Kimi Team / Moonshot AI, arXiv:2603.15031.

Both classes replace the standard residual addition with a softmax-weighted
sum over previous hidden states.  ``BlockAttnRes`` is the practical target;
``FullAttnRes`` stores every previous sublayer state and is kept as a
small-model diagnostic reference.

Math (BlockAttnRes):
    values  = stack([*blocks, partial_block])        # (N+1, B, S, d)
    keys    = RMSNorm(values)                        # normalized for bounded logits
    logits  = einsum("d, nbsd -> nbs", pseudo_query, keys) / sqrt(d)
    weights = softmax(logits, dim=0)                 # over depth axis
    output  = einsum("nbs, nbsd -> bsd", weights, values)

At init pseudo_query=0 → uniform weights → output equals the mean of all inputs.

Design-ladder role (see METHODOLOGY.md §1.1):
    This module is **Rung 4** of the project's design ladder — the upper bound
    on the "history aggregation" axis.  Rung 1 (RR) and Rung 2 (VEGA) are
    progressively cheaper approximations of exactly the operation this module
    performs: a content-conditioned retrieval over the history of depth states.
    The empirical question this project is designed to answer is how much of
    AttnRes's quality is recoverable at O(L) cost, and the block / full
    variants here are the reference points against which the cheaper
    alternatives are measured.

Init contract (verified by tests/test_design_ladder.py::test_attnres_uniform_mean_at_init):
    pseudo_query = 0 → uniform softmax → h_mid = mean([*blocks, partial]).
    For the first layer (blocks = [embedding], partial = embedding) this is
    well-defined; for deeper layers it converges to a uniform mean of the
    history.  This is the *weakest* zero-start of the alternatives — the
    cell's read-out is fully active at step 0, just with content-blind weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockAttnRes(nn.Module):
    """Gated aggregation over a sequence of completed architectural blocks.

    Follows the official Attention Residuals paper (Kimi / Moonshot AI,
    arXiv:2603.15031) §2.1: per-layer learned pseudo-query `w_l`, parameter-
    free RMSNorm of the keys, softmax over the depth axis.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for the internal RMS normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Zero-init pseudo_query → uniform softmax → mean residual at init
        # (paper: "all pseudo-query vectors must be initialized to zero").
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Parameter-free RMSNorm of the keys, matching the paper's protocol.

        The paper uses an RMSNorm with no learnable scale on the keys; this
        is the same function applied to the sublayer input, just without a
        learned gain.
        """
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(dtype)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate block history with a learned pseudo-query.

        Args:
            blocks: Completed block states including the embedding source.
                    Must be non-empty.
            partial_block: Current in-block residual state.

        Returns:
            Weighted aggregation of shape (B, S, d_model).
        """
        if not blocks:
            raise ValueError("History list 'blocks' must not be empty.")

        values: torch.Tensor = torch.stack([*blocks, partial_block], dim=0)
        keys: torch.Tensor = self._rms_norm(values)
        logits: torch.Tensor = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys) / (
            self.d_model**0.5
        )
        weights: torch.Tensor = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)


class FullAttnRes(nn.Module):
    """Full Attention Residual over every stored sublayer state.

    Stores every hidden state since the embedding and aggregates them all.
    Memory cost is O(2L * d) which limits practical use to small models.
    Kept as a diagnostic reference for the block-based version.

    Args:
        d_model: Hidden dimension.
        eps: Epsilon for the internal RMS normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Zero-init pseudo_query → uniform softmax over all stored states at init.
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Parameter-free RMSNorm, matching the paper's protocol (no learnable scale)."""
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(dtype)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate all previous depth states.

        Args:
            states: Hidden states in depth order.  Must be non-empty.

        Returns:
            Weighted aggregation of shape (B, S, d_model).
        """
        if not states:
            raise ValueError("State history must not be empty.")
        values = torch.stack(states, dim=0)
        keys = self._rms_norm(values)
        logits = torch.einsum("d, n b s d -> n b s", self.pseudo_query, keys) / (self.d_model**0.5)
        weights = logits.softmax(dim=0)
        return torch.einsum("n b s, n b s d -> b s d", weights, values)
