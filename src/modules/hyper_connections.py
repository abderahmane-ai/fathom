"""mHC-Lite — Manifold-Constrained Hyper-Connections, lite variant.

References
----------
- Xie, Z. et al., "mHC: Manifold-Constrained Hyper-Connections", DeepSeek-AI,
  arXiv:2512.24880, 2025.  (Original mHC with Sinkhorn-Knopp projection.)
- Yang, Y. & Gao, J., "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations",
  arXiv:2601.05732, 2026.  (mHC-Lite: convex combination of permutation
  matrices — exact doubly stochasticity, no iterative projection.)

This module implements the **mHC-Lite** variant for ``num_channels = 2``
(the only n for which the factorial n! = 2 permutation matrices are
trivially small).  The two permutations of a 2×2 matrix are the
identity and the swap, so the residual mixing matrix H_res is exactly
the convex combination

    H_res = a_0 · I + a_1 · P_swap

with ``(a_0, a_1) = softmax(logits)`` — automatically doubly stochastic
by construction (Birkhoff-von Neumann theorem).

The full layer update is

    x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l)

where the three mappings are computed dynamically from the (RMS-normed,
flattened) input state:

    H_pre  = σ(α_pre · x_norm · W_pre  + b_pre)            (1 × n)
    H_post = 2 · σ(α_post · x_norm · W_post + b_post)      (1 × n)
    H_res  = softmax(α_res · x_norm · W_res + b_res) · P   (n × n)

Init protocol (verbatim from arXiv:2601.05732 §3.3):
    - W_pre, W_post, W_res = 0
    - α_pre, α_post, α_res = 0.01
    - b_pre: −1 in all entries except the main-channel entry (index 0) = +1
    - b_post: same structure as b_pre
    - b_res: −8 in all entries except the identity-permutation entry = 0
      (so softmax concentrates on the identity permutation)

At init (n=2):
    H_pre  ≈ [σ(+1), σ(−1)]      ≈ [0.731, 0.269]
    H_post ≈ 2·[σ(+1), σ(−1)]    ≈ [1.462, 0.538]
    H_res  ≈ I_2                  (softmax([0, −8]) ≈ [1, 0])

This is an **approximate** zero-start: not bit-for-bit standard
residual, but the closest the paper's init gets.  See METHODOLOGY.md
§5.2 for the full discussion of the strict-vs-soft init distinction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import RMSNorm


class HyperConnection(nn.Module):
    """mHC-Lite (num_channels = 2) following arXiv:2601.05732 §3.

    Provides ``pre_mix`` and ``post_mix`` that the transformer layer
    composes around its attention and FFN sublayers.  All three mixing
    matrices (H_pre, H_post, H_res) are **input-dependent** (dynamic),
    computed from an RMSNorm of the flattened input state.

    The init protocol is exactly the one from the paper — see the
    module-level docstring for the precise bias / scale values.

    Args:
        d_model: Hidden dimension of each channel.
        num_channels: n — the number of parallel residual channels.
            Only n=2 is supported; the mHC-Lite parameterization requires
            the factorial n! permutation matrices, which grows
            factorially.  For n=2 this is 2 matrices; for n=4 it would
            be 24, which is the regime the sHC paper warns about
            (arXiv:2603.20896).
        use_static_input: When True, an additional static-input row is
            appended to W_pre (matches the original mHC paper).  Disabled
            here — none of the benchmark configs use it, and the
            arithmetic at init would need to be re-derived.  Kept as a
            constructor arg for forward compatibility.
        init_static_gate: Initial value for the static-input row of
            W_pre.  Ignored when use_static_input is False.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int = 2,
        use_static_input: bool = False,
        init_static_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_channels != 2:
            raise NotImplementedError(
                f"mHC-Lite in this codebase only supports num_channels=2 "
                f"(got {num_channels}); the n! permutation-matrix parameterization "
                f"grows factorially and is not implemented for n > 2 here. "
                f"See arXiv:2603.20896 for the sHC alternative that scales polynomially."
            )
        if use_static_input:
            raise NotImplementedError(
                "use_static_input=True is not supported in this implementation; "
                "the static-input path would require re-deriving the init biases."
            )

        self.d_model = d_model
        self.num_channels = num_channels
        self.use_static_input = use_static_input

        # Flattened input dimension (always n*d since use_static_input=False).
        in_dim = num_channels * d_model

        # ── Dynamic (input-dependent) mixing matrices ─────────────────────
        # Linear projections: (in_dim,) → (num_channels,) for H_pre and H_post,
        # (in_dim,) → (num_channels!,) for H_res.  For n=2, n! = 2.
        self.W_pre = nn.Parameter(torch.zeros(in_dim, num_channels))
        self.W_post = nn.Parameter(torch.zeros(in_dim, num_channels))
        self.W_res = nn.Parameter(torch.zeros(in_dim, 2))  # 2 = n!

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

        # b_res: −8 in all entries except the identity-permutation entry = 0
        # (so softmax concentrates on the identity)
        b_res = torch.full((1, 2), -8.0)
        b_res[0, 0] = 0.0
        self.b_res = nn.Parameter(b_res)

        # RMSNorm for the input (parameter-free, matching the mHC paper)
        self.norm = RMSNorm(in_dim, eps=1e-6)
        # The paper uses parameter-free RMSNorm; we reuse the existing RMSNorm
        # module with its learnable scale zeroed out, which is mathematically
        # equivalent and keeps the codebase consistent.  We do *not* freeze
        # the scale — it is a regular learnable parameter, but starts at 1
        # (the default for RMSNorm), so the parameter-free behavior is the
        # init point.

        # Pre-compute the 2! = 2 permutation matrices of 2×2:
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

        # Dynamic H_res via softmax over the n! = 2 permutation logits
        H_res_logits = self.alpha_res * (H_norm @ self.W_res) + self.b_res  # (B, S, 2)
        H_res_weights = torch.softmax(H_res_logits, dim=-1)  # (B, S, 2)

        # H_res · H: for n=2, this is a convex combination of H and swap(H).
        # swap(H) flips the channel axis so channel 0 ↔ channel 1.
        # H_res_weights has shape (B, S, 2); unsqueeze to (B, S, 1, 1) for
        # broadcast multiplication against (B, S, 2, d).
        H_swap = H.flip(dims=(-2,))  # (B, S, n, d) — channels swapped
        H_res_mixed = (
            H_res_weights[..., 0:1].unsqueeze(-1) * H
            + H_res_weights[..., 1:2].unsqueeze(-1) * H_swap
        )  # (B, S, n, d)

        # H_post^T · y: distribute y to channels. For each channel c, the
        # contribution is H_post[..., c] * y.  Broadcasting: (B,S,n,1) * (B,S,1,d) → (B,S,n,d)
        y_post = H_post.unsqueeze(-1) * y.unsqueeze(-2)  # (B, S, n, d)

        # Final: H_new = H_res · H + H_post^T · y
        return H_res_mixed + y_post

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_channels={self.num_channels}, "
            f"use_static_input={self.use_static_input}"
        )
