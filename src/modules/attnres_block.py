"""Block Attention Residual (Block-AttnRes) module.

Paper-faithful implementation of Block AttnRes from:
  "Attention Residuals", Kimi Team / Moonshot AI, arXiv:2603.15031.

Design principles
-----------------
* **Stateless forward**: block history is a plain Python ``list[Tensor]`` owned
  by the caller (``TransformerDecoder``).  No hidden state lives inside this
  module — making it safe under DDP, gradient checkpointing, and AMP.
* **Per-sublayer projection**: each ``TransformerLayer`` instantiates *two*
  ``BlockAttnRes`` modules — one for the pre-Attn residual and one for the
  pre-FFN residual — each with an independent pseudo-query and RMSNorm scale.
* **RMSNorm keys**: keys are normalised before dot-product as recommended by
  the paper to keep logit magnitudes bounded independent of hidden-state scale.
* **Zero-init pseudo-query**: at init all weights → equal-weight average of
  blocks; the network learns to be selective from that neutral start.
* **Gradient flow**: ``partial_block`` tensors are stored without detach so
  that attention weights and aggregated hidden states remain in the autograd
  graph, enabling cross-block gradient signal.

Shapes
------
All tensors are ``(B, S, d_model)`` unless noted.  ``N`` denotes the number of
completed blocks (including the initial token-embedding block-0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockAttnRes(nn.Module):
    """Block Attention Residual: content-aware cross-block aggregation.

    Replaces the standard ``h = h_prev + y`` residual with a softmax-weighted
    sum over all previous block representations plus the current partial block.

    Implements (arXiv:2603.15031, §3.2):

    .. math::

        \\mathbf{K} &= \\text{RMSNorm}(\\mathbf{V}) \\\\
        \\alpha_{i \\to l} &= \\text{softmax}_N(\\mathbf{w}_l \\cdot \\mathbf{K}_i) \\\\
        \\mathbf{h}_l &= \\sum_{i} \\alpha_{i \\to l} \\cdot \\mathbf{V}_i

    Args:
        d_model: Hidden dimension ``d``.
        eps: Epsilon for numerical stability in RMSNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learned pseudo-query (one per module instance / sublayer).
        # Zero-init → equal-weight average at the start of training.
        self.pseudo_query: nn.Parameter = nn.Parameter(torch.zeros(d_model))

        # Learnable per-element scale for RMSNorm (no bias — matches paper).
        self.norm_scale: nn.Parameter = nn.Parameter(torch.ones(d_model))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation with learned scale.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            RMS-normalised tensor of the same shape.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.norm_scale * (x * rms)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted aggregation over block history.

        Args:
            blocks: ``N`` completed block tensors, each ``(B, S, d_model)``.
                    **Must** contain at least the initial token-embedding
                    (block-0) so that every layer has something to attend over.
            partial_block: Running intra-block accumulation ``(B, S, d_model)``.
                           Gradients flow through this tensor.

        Returns:
            Aggregated hidden state ``h`` of shape ``(B, S, d_model)``.

        Raises:
            ValueError: If ``blocks`` is empty.

        Preconditions:
            * ``len(blocks) >= 1`` (caller must supply at least block-0).
            * All tensors in ``blocks`` and ``partial_block`` share the same
              device, dtype, and ``(B, S, d_model)`` shape.
        """
        if not blocks:
            raise ValueError(
                "blocks must contain at least the token-embedding (block-0). "
                "Initialise blocks = [token_emb] in TransformerDecoder.forward."
            )

        # Stack completed blocks + partial_block → V ∈ (N+1, B, S, d)
        V: torch.Tensor = torch.stack([*blocks, partial_block], dim=0)

        # Normalise keys (prevents logit blow-up with deep networks)
        K: torch.Tensor = self._rms_norm(V)  # (N+1, B, S, d)

        # Logits: pseudo_query · K_i for every block i and position (B, S)
        # Shape: (N+1, B, S)
        logits: torch.Tensor = torch.einsum("d, n b s d -> n b s", self.pseudo_query, K)

        # Softmax over block dimension (N+1)
        weights: torch.Tensor = logits.softmax(dim=0)  # (N+1, B, S)

        # Weighted sum: h = Σ_i α_i · V_i
        h: torch.Tensor = torch.einsum("n b s, n b s d -> b s d", weights, V)

        return h