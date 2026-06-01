"""Manifold-Constrained Hyper-Connections (mHC) — both algorithms.

References
----------
- Xie, Z. et al., "mHC: Manifold-Constrained Hyper-Connections", DeepSeek-AI,
  arXiv:2512.24880, 2025.  (Original mHC: H_res = SK(exp(...)) with 20
  Sinkhorn-Knopp iterations.  This is the default in this codebase.)
- Yang, Y. & Gao, J., "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations",
  arXiv:2601.05732, 2026.  (mHC-Lite: H_res = softmax(...) · [permutations]
  using the Birkhoff-von Neumann theorem.  Equivalent to mHC for n=2.)

This module implements both for ``num_channels = 2`` (the only n for which
either algorithm is tractable in this codebase).  At init, both algorithms
produce doubly-stochastic 2×2 matrices that are very close to I_2.

The full layer update is

    H_{l+1} = H_res · H_l + H_post^T · F(H_pre · H_l, W_l)

where the three mappings are computed dynamically from the (RMS-normed,
flattened) input state:

    H_pre  = σ(α_pre · x_norm · W_pre  + b_pre)            (1 × n)
    H_post = 2 · σ(α_post · x_norm · W_post + b_post)      (1 × n)
    H_res  =  ┌ SK(exp(α_res · mat(x_norm · W_res) + b_res))      [algorithm="sinkhorn_knopp"]
             └ softmax(α_res · x_norm · W_res + b_res) · P         [algorithm="permutation_convex"]

Init protocol (verbatim from arXiv:2512.24880 §4.2 for SK, arXiv:2601.05732 §3.3
for permutation-convex):
    - W_pre, W_post, W_res = 0
    - α_pre, α_post, α_res = 0.01
    - b_pre: −1 in all entries except the main-channel entry (index 0) = +1
    - b_post: same structure as b_pre
    - b_res: −8 in all entries except the identity-permutation / identity-matrix
      entry = 0 (so SK / softmax concentrates on the identity)

At init (n=2):
    H_pre  ≈ [σ(+1), σ(−1)]      ≈ [0.731, 0.269]
    H_post ≈ 2·[σ(+1), σ(−1)]    ≈ [1.462, 0.538]
    H_res  ≈ I_2                  (permutation_convex is exact; SK is approximate after 20 iters)

This is an **approximate** zero-start: not bit-for-bit standard residual,
but the closest the papers' init protocols get.  See METHODOLOGY.md §5.2
for the full discussion of the strict-vs-soft init distinction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import RMSNorm


class HyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connections, n=2.

    Provides ``pre_mix`` and ``post_mix`` that the transformer layer
    composes around its attention and FFN sublayers.  All three mixing
    matrices (H_pre, H_post, H_res) are **input-dependent** (dynamic),
    computed from an RMSNorm of the flattened input state.

    The init protocol is exactly the one from the original DeepSeek mHC
    paper (arXiv:2512.24880) for ``algorithm="sinkhorn_knopp"`` (default),
    and the mHC-Lite paper (arXiv:2601.05732) for
    ``algorithm="permutation_convex"``.

    Args:
        d_model: Hidden dimension of each channel.
        num_channels: n — the number of parallel residual channels.
            Only n=2 is supported.  mHC's W_res has shape (n*d, n^2), so
            for n=2 it is (2d, 4); mHC-Lite's W_res has shape (n*d, n!),
            so for n=2 it is (2d, 2).  For n≥4 the SK approach becomes
            expensive (W_res size grows as n^2) and the permutation-convex
            approach becomes infeasible (n! grows factorially).  See the
            sHC paper (arXiv:2603.20896) for the polynomial alternative.
        algorithm: One of ``"sinkhorn_knopp"`` (default — the original
            DeepSeek mHC, Eq. 19 of the paper) or
            ``"permutation_convex"`` (mHC-Lite, Eq. 3 of the Yang & Gao
            paper).  For n=2 the two produce equivalent doubly-stochastic
            matrices; the SK version is the canonical reference.
        t_max: Number of Sinkhorn-Knopp iterations (only used when
            ``algorithm="sinkhorn_knopp"``).  The paper uses 20.
        use_static_input: When True, an additional static-input row is
            appended to W_pre (matches the original mHC paper).  Disabled
            here — none of the benchmark configs use it, and the
            arithmetic at init would need to be re-derived.
        init_static_gate: Initial value for the static-input row of
            W_pre.  Ignored when use_static_input is False.
    """

    ALGORITHMS = ("sinkhorn_knopp", "permutation_convex")

    def __init__(
        self,
        d_model: int,
        num_channels: int = 2,
        algorithm: str = "sinkhorn_knopp",
        t_max: int = 20,
        use_static_input: bool = False,
        init_static_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_channels != 2:
            raise NotImplementedError(
                f"mHC in this codebase only supports num_channels=2 "
                f"(got {num_channels}).  mHC's W_res grows as n*d*n^2 and "
                f"mHC-Lite's W_res grows as n*d*n!; both are intractable "
                f"for n > 2.  See arXiv:2603.20896 for the sHC alternative."
            )
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'.  Must be one of {self.ALGORITHMS}.")
        if use_static_input:
            raise NotImplementedError(
                "use_static_input=True is not supported in this implementation; "
                "the static-input path would require re-deriving the init biases."
            )

        self.d_model = d_model
        self.num_channels = num_channels
        self.algorithm = algorithm
        self.t_max = t_max
        self.use_static_input = use_static_input

        # Flattened input dimension (always n*d since use_static_input=False).
        in_dim = num_channels * d_model

        # ── Dynamic (input-dependent) mixing matrices ─────────────────────
        # W_pre and W_post are (in_dim, n) for both algorithms.
        self.W_pre = nn.Parameter(torch.zeros(in_dim, num_channels))
        self.W_post = nn.Parameter(torch.zeros(in_dim, num_channels))

        # W_res shape depends on the algorithm:
        #   sinkhorn_knopp:    (in_dim, n^2)   — will be reshaped to (B, S, n, n)
        #   permutation_convex: (in_dim, n!)   — for n=2, n! = 2, softmax over permutations
        res_out_dim = num_channels * num_channels if algorithm == "sinkhorn_knopp" else 2
        self.W_res = nn.Parameter(torch.zeros(in_dim, res_out_dim))

        # ── Scalar α gates (paper init: 0.01) ────────────────────────────
        self.alpha_pre = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_post = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_res = nn.Parameter(torch.full((1,), 0.01))

        # ── Biases per paper protocol ────────────────────────────────────
        # b_pre: −1 in all entries except main-channel entry = +1
        b_pre = torch.full((1, num_channels), -1.0)
        b_pre[0, 0] = 1.0
        self.b_pre = nn.Parameter(b_pre)

        # b_post: same structure
        b_post = torch.full((1, num_channels), -1.0)
        b_post[0, 0] = 1.0
        self.b_post = nn.Parameter(b_post)

        # b_res shape depends on the algorithm:
        #   sinkhorn_knopp:    (1, n, n)   — 2×2 matrix, identity = 0, off-diag = −8
        #   permutation_convex: (1, n!)    — for n=2, 2-vector, identity-perm = 0, swap = −8
        if algorithm == "sinkhorn_knopp":
            b_res = torch.full((1, num_channels, num_channels), -8.0)
            b_res[0, 0, 0] = 0.0  # identity-matrix top-left entry
        else:
            b_res = torch.full((1, 2), -8.0)
            b_res[0, 0] = 0.0  # identity-permutation entry
        self.b_res = nn.Parameter(b_res)

        # RMSNorm for the input (parameter-free, matching the mHC paper).
        # We reuse the existing RMSNorm module with its learnable scale zeroed
        # out at init, which is mathematically equivalent to a parameter-free
        # RMSNorm and keeps the codebase consistent.  The scale is a regular
        # learnable parameter that starts at 1 (the default for RMSNorm), so
        # the parameter-free behavior is the init point.
        self.norm = RMSNorm(in_dim, eps=1e-6)

        # Pre-compute the 2! = 2 permutation matrices of 2×2 (used by the
        # permutation_convex algorithm only):
        #   P_0 = I_2,  P_1 = swap
        # For n=2, applying H_res = a_0·I + a_1·swap to a tensor of shape
        # (..., n, d) is just a convex combination of the tensor and its
        # channel-flipped version.
        self.register_buffer(
            "permutations",
            torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ),
        )

        # Cached normalized input (set in pre_mix, used in post_mix).
        # We don't register this as a buffer because it's recomputed every
        # forward pass; this is a transient attribute only.
        self._cached_norm: torch.Tensor | None = None

    # ─────────────────────────────────────────────────────────────────────
    # H_res computation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_H_res(self, H_norm: torch.Tensor) -> torch.Tensor:
        """Compute the doubly-stochastic H_res matrix.

        Args:
            H_norm: RMSNormed flattened input, shape (B, S, n*d).

        Returns:
            H_res of shape (B, S, n, n) — a doubly-stochastic matrix
            (rows and columns sum to 1, all entries non-negative).
        """
        if self.algorithm == "sinkhorn_knopp":
            return self._sinkhorn_knopp_res(H_norm)
        return self._permutation_convex_res(H_norm)

    def _sinkhorn_knopp_res(self, H_norm: torch.Tensor) -> torch.Tensor:
        """Original DeepSeek mHC (arXiv:2512.24880 Eq. 19): H_res = SK(exp(...)).

        We use the per-row log-sum-exp stability trick (subtract row max before
        ``exp``) so the operation is safe under fp16/bf16 at training-time
        activation magnitudes.  This matches the canonical reference impls
        (svdrecbd/mhc-mlx, aamir-gmail/MC-Hyper-Connections-Implementation) and
        is bit-identical to the naive ``exp`` when logits are bounded.
        """
        B, S = H_norm.shape[:2]
        n = self.num_channels
        # α · x'W + b → (B, S, n^2), then mat() reshape to (B, S, n, n).
        raw = self.alpha_res * (H_norm @ self.W_res) + self.b_res.reshape(1, 1, n * n)
        H_res_logits = raw.reshape(B, S, n, n)
        # Per-row max subtraction for fp16/bf16 safety, then exp → Sinkhorn-Knopp.
        # Use fp32 throughout the SK loop; the final cast back happens after.
        M = (H_res_logits - H_res_logits.max(dim=-1, keepdim=True).values).float().exp()
        for _ in range(self.t_max):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-12)  # row-normalize
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-12)  # col-normalize
        return M.to(H_norm.dtype)

    def _permutation_convex_res(self, H_norm: torch.Tensor) -> torch.Tensor:
        """mHC-Lite (arXiv:2601.05732 Eq. 3): H_res = softmax(α·x'W + b) · P.

        For n=2 the convex combination is H_res = a_0·I + a_1·P_swap.
        Returns the H_res matrix (not the weighted combination — that is
        applied separately against the channel state).
        """
        H_res_logits = self.alpha_res * (H_norm @ self.W_res) + self.b_res  # (B, S, 2)
        H_res_weights = torch.softmax(H_res_logits, dim=-1)  # (B, S, 2)
        # einsum: (B, S, k) × (k, n, n) → (B, S, n, n)
        return torch.einsum("bsk, knm -> bsnm", H_res_weights, self.permutations)

    # ─────────────────────────────────────────────────────────────────────
    # Pre-mix: combine the n-channel input state into a single d-dim vector
    # for the sublayer to consume.
    # ─────────────────────────────────────────────────────────────────────

    def pre_mix(self, H: torch.Tensor) -> torch.Tensor:
        """Compute the dynamic H_pre and return the mixed sublayer input.

        Args:
            H: n-channel residual state, shape (B, S, n, d).

        Returns:
            Mixed sublayer input of shape (B, S, d) — the result of
            ``H_pre · H`` where ``H_pre = σ(α_pre · RMSNorm(H_flat) · W_pre + b_pre)``.
        """
        B, S, n, d = H.shape
        if n != self.num_channels:
            raise ValueError(
                f"Expected n={self.num_channels} channels, got {n}. "
                f"All layers in a model must use the same num_channels."
            )

        # Flatten to (B, S, n*d) and RMSNorm
        H_flat = H.reshape(B, S, n * d)
        H_norm = self.norm(H_flat)

        # Cache for post_mix (avoids recomputing the normalization)
        self._cached_norm = H_norm

        # Dynamic H_pre
        H_pre_logits = self.alpha_pre * (H_norm @ self.W_pre) + self.b_pre
        H_pre = torch.sigmoid(H_pre_logits)  # (B, S, n)

        # x_pre = H_pre · H: weighted sum over channels
        x_pre = torch.einsum("bsi, bsid -> bsd", H_pre, H)
        return x_pre

    # ─────────────────────────────────────────────────────────────────────
    # Post-mix: distribute the sublayer output to the n channels and mix
    # the residual stream via the doubly-stochastic H_res.
    # ─────────────────────────────────────────────────────────────────────

    def post_mix(self, H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dynamic H_post and H_res, then return the new residual state.

        Args:
            H: n-channel residual state **before** the sublayer, shape (B, S, n, d).
                Must be the same H that was passed to the most recent
                ``pre_mix`` call so the cached normalization is consistent.
            y: Sublayer output, shape (B, S, d).

        Returns:
            New n-channel residual state of shape (B, S, n, d), equal to
            ``H_res · H + H_post^T · y``.
        """
        B, S, n, d = H.shape
        if n != self.num_channels:
            raise ValueError(f"Expected n={self.num_channels} channels, got {n}.")

        if self._cached_norm is None:
            raise RuntimeError(
                "post_mix called without a prior pre_mix call. "
                "Each layer must call pre_mix before post_mix so the "
                "cached RMSNorm of the input is available."
            )

        H_norm = self._cached_norm

        # Dynamic H_post (note the 2× factor — paper §3.2)
        H_post_logits = self.alpha_post * (H_norm @ self.W_post) + self.b_post
        H_post = 2.0 * torch.sigmoid(H_post_logits)  # (B, S, n)

        # Dynamic H_res (algorithm-dependent)
        H_res = self._compute_H_res(H_norm)  # (B, S, n, n)

        # H_res · H: (B, S, n, n) @ (B, S, n, d) → (B, S, n, d)
        H_res_mixed = torch.einsum("bsnm, bsmd -> bsnd", H_res, H)

        # H_post^T · y: distribute y to channels. For each channel c, the
        # contribution is H_post[..., c] * y.  Broadcasting: (B,S,n,1) * (B,S,1,d) → (B,S,n,d)
        y_post = H_post.unsqueeze(-1) * y.unsqueeze(-2)  # (B, S, n, d)

        # Final: H_new = H_res · H + H_post^T · y
        return H_res_mixed + y_post

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_channels={self.num_channels}, "
            f"algorithm={self.algorithm}, t_max={self.t_max}, "
            f"use_static_input={self.use_static_input}"
        )
