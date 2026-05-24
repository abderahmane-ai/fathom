"""Recurrent Residual cell for Transformer sublayers.

The cell implements the methodology equations exactly: diagonal read/update
gates, an RMS-normalized memory read path, and an EMA memory update across
depth. At initialization the memory injection path is zero, so the hidden-state
update matches a standard Pre-LN residual addition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped


class RMSNorm(nn.Module):
    """Parameter-free root mean square normalization.

    Args:
        d_model: Hidden dimension used to validate the input shape.
        eps: Numerical stability constant.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "*batch d_model"]
    ) -> Float[torch.Tensor, "*batch d_model"]:
        """Normalize ``x`` by its RMS over the final dimension.

        Args:
            x: Tensor with final dimension ``d_model``.

        Returns:
            RMS-normalized tensor with the same shape as ``x``.

        Preconditions:
            ``x.shape[-1] == d_model``.
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected final dim {self.d_model}, got {x.size(-1)}.")
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale


class RecurrentResidualCell(nn.Module):
    """Diagonal gated recurrent residual transition.

    Args:
        d_model: Hidden dimension ``d``.
        num_layers: Total transformer layers; allocates two depth biases per layer.
        read_gate_bias: Initial read-gate bias.
        update_gate_bias: Initial update-gate bias.
        eps: Epsilon for parameter-free LayerNorm/RMSNorm.
        gate_init_std: Standard deviation for diagonal gate weights.
        memory_gain_init: Initial value for the memory gain vector.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        read_gate_bias: float = -3.0,
        update_gate_bias: float = -2.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
        memory_gain_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.eps = eps

        self.read_weight = nn.Parameter(torch.empty(d_model))
        self.read_bias = nn.Parameter(torch.full((d_model,), read_gate_bias))
        self.update_weight = nn.Parameter(torch.empty(d_model))
        self.update_bias = nn.Parameter(torch.full((d_model,), update_gate_bias))
        self.memory_gain = nn.Parameter(torch.full((d_model,), memory_gain_init))
        self.depth_bias = nn.Parameter(torch.zeros(self.num_sublayers, d_model))
        self.m_init = nn.Parameter(torch.zeros(d_model))
        self.memory_norm = RMSNorm(d_model, eps=eps)

        nn.init.normal_(self.read_weight, mean=0.0, std=gate_init_std)
        nn.init.normal_(self.update_weight, mean=0.0, std=gate_init_std)

        self.last_read_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)
        self.last_update_gate: torch.Tensor = torch.zeros((), dtype=torch.float32)

    @property
    def parameter_count_formula(self) -> int:
        """Return the methodology parameter count.

        Returns:
            ``(num_sublayers + 6) * d_model``.
        """
        return (self.num_sublayers + 6) * self.d_model

    @jaxtyped(typechecker=beartype)
    def get_initial_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Return the expanded learnable initial memory state.

        Args:
            batch_size: Batch size ``B``.
            seq_len: Sequence length ``S``.
            device: Optional target device.

        Returns:
            Initial memory tensor of shape ``(B, S, d_model)``.
        """
        target_device = self.m_init.device if device is None else device
        return self.m_init.to(target_device).view(1, 1, -1).expand(batch_size, seq_len, -1)

    def _sublayer_position(self, layer_idx: int, sublayer: int) -> int:
        """Map a layer/sublayer pair to a depth-bias row.

        Args:
            layer_idx: Zero-based transformer layer index.
            sublayer: ``0`` for attention and ``1`` for FFN.

        Returns:
            Row index in ``depth_bias``.

        Preconditions:
            ``0 <= layer_idx < num_layers`` and ``sublayer`` is ``0`` or ``1``.
        """
        if sublayer not in (0, 1):
            raise ValueError(f"sublayer must be 0 or 1, got {sublayer}.")
        position = layer_idx * 2 + sublayer
        if position >= self.num_sublayers:
            raise IndexError(f"sublayer position {position} exceeds {self.num_sublayers} entries.")
        return position

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h_prev: Float[torch.Tensor, "batch seq d_model"],
        y: Float[torch.Tensor, "batch seq d_model"],
        m: Float[torch.Tensor, "batch seq d_model"],
        layer_idx: int,
        sublayer: int = 0,
        h_norm: Float[torch.Tensor, "batch seq d_model"] | None = None,
    ) -> tuple[Float[torch.Tensor, "batch seq d_model"], Float[torch.Tensor, "batch seq d_model"]]:
        """Compute one recurrent residual transition.

        Args:
            h_prev: Hidden state entering the sublayer, shape ``(B, S, d_model)``.
            y: Sublayer output, shape ``(B, S, d_model)``.
            m: Memory state entering the sublayer, shape ``(B, S, d_model)``.
            layer_idx: Zero-based transformer layer index.
            sublayer: ``0`` for attention and ``1`` for FFN.
            h_norm: Optional pre-computed normalized representation of ``h_prev``.

        Returns:
            Tuple ``(h_new, m_new)`` after read and update.

        Preconditions:
            ``h_prev``, ``y``, and ``m`` have identical shapes.
        """
        if h_prev.shape != y.shape or h_prev.shape != m.shape:
            raise ValueError("h_prev, y, and m must have identical shapes.")

        if h_norm is None:
            h_norm = F.layer_norm(h_prev, (self.d_model,), eps=self.eps)
        read_gate = torch.sigmoid(self.read_weight * h_norm + self.read_bias)
        memory_read = self.memory_gain * self.memory_norm(m)
        h_new = h_prev + y + read_gate * memory_read

        position = self._sublayer_position(layer_idx, sublayer)
        update_gate = torch.sigmoid(
            self.update_weight * y + self.update_bias + self.depth_bias[position]
        )
        m_new = update_gate * y + (1.0 - update_gate) * m

        self.last_read_gate = read_gate.detach().mean()
        self.last_update_gate = update_gate.detach().mean()
        return h_new, m_new
