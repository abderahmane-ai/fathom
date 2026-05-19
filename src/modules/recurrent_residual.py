"""Recurrent Residual Cell for Transformer layers.

Augments standard additive residual connections with a global shared gated memory
that persists and evolves across the depth dimension. This architecture enables
long-range information persistence with O(1) memory and compute complexity.

Equations (per sublayer transition):
    h_norm  = LayerNorm(h_prev)
    r       = sigmoid(W_r · h_norm + b_r)          # Read gate
    m_inj   = W_m · RMSNorm(m)                     # Memory projection
    h_new   = LayerNorm(α · h_prev + y + r ⊙ m_inj) # DeepNorm update

    e_l     = depth_embedding[layer_idx * 2 + sublayer]
    α_gate  = sigmoid(W_α · y + b_α + e_l)         # Write gate
    m_new   = α_gate ⊙ y + (1 − α_gate) ⊙ m        # State transition

Design Rationale:
*   **Global Memory**: A single state 'm' is shared across all layers, enabling
    cross-layer information flow (Depth-as-RNN).
*   **DeepNorm Stability**: Residual branch scaling (α) and Post-LN normalization
    ensure gradient stability at extreme depths (1,000+ layers).
*   **DDP-Safe State**: Memory is managed as a transient instance attribute rather
    than a registered buffer to avoid DDP synchronisation overhead during updates.
*   **Depth Awareness**: Learnable embeddings provide distinct biases for each
    sublayer position (2N per stack), allowing position-dependent gating.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Normalization with learnable scaling."""
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


class RecurrentResidualCell(nn.Module):
    """Gated recurrent residual connection with DeepNorm stability.

    Args:
        d_model: Hidden dimension.
        num_layers: Total transformer layers (used to derive DeepNorm alpha).
        eps: Epsilon for normalization stability.
        gate_r_bias: Initial bias for the read (reset) gate.
        gate_alpha_bias: Initial bias for the write (update) gate.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        eps: float = 1e-5,
        gate_r_bias: float = -3.0,
        gate_alpha_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps

        # DeepNorm alpha constant scales the residual branch (h_prev).
        self.alpha = (2.0 * self.num_sublayers) ** 0.25

        # Read Gate: Controls memory injection magnitude.
        self.gate_r = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_r.weight)
        nn.init.constant_(self.gate_r.bias, gate_r_bias)

        # Memory Projection: Normalized memory influence.
        self.norm_m = RMSNorm(d_model, eps=eps)
        self.proj_m = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj_m.weight)

        # Write Gate: Controls EMA update to the shared memory state.
        self.gate_alpha = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_alpha.weight)
        nn.init.constant_(self.gate_alpha.bias, gate_alpha_bias)

        # Position-aware depth embeddings (2 per layer: Attn + FFN).
        self.depth_emb = nn.Embedding(self.num_sublayers, d_model)
        nn.init.zeros_(self.depth_emb.weight)

        # Learnable initial memory state expanded to (B, S, d) per batch.
        self.m_init = nn.Parameter(torch.zeros(d_model))
        self.m: torch.Tensor = torch.zeros(1, 1, d_model)

    def reset_memory(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialise shared memory with the learnable prior for a new batch."""
        if device is None:
            device = self.m_init.device
        
        # Expanded to full batch shape (B, S, d).
        self.m = self.m_init.view(1, 1, -1).expand(batch_size, seq_len, -1).contiguous()

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        layer_idx: int,
        sublayer: int = 0,
    ) -> torch.Tensor:
        """Performs one gated residual update and memory transition."""
        m = self.m

        # 1. Read Mechanism: Generate reset signal from normalized hidden state.
        h_norm_pre = F.layer_norm(h_prev, (self.d_model,))
        r = torch.sigmoid(self.gate_r(h_norm_pre))

        # 2. Injection: Merge projected memory into the hidden path.
        m_inj = self.proj_m(self.norm_m(m))

        # 3. DeepNorm Residual Update: Final Post-LN stabilization.
        h_combined = self.alpha * h_prev + y + r * m_inj
        h_new = F.layer_norm(h_combined, (self.d_model,))

        # 4. Write Mechanism: EMA update with depth-aware gating.
        sublayer_pos = layer_idx * 2 + sublayer
        depth_bias = self.depth_emb(
            torch.tensor(sublayer_pos, device=h_prev.device)
        )
        alpha = torch.sigmoid(self.gate_alpha(y) + depth_bias)
        
        m_new = alpha * y + (1.0 - alpha) * m
        self.m = m_new

        return h_new