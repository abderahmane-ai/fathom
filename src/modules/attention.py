"""Multi-head causal self-attention with Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Precompute RoPE cosine/sine tables up to max_seq_len.

    Args:
        head_dim: Per-head feature dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Geometric base for the frequency bands.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

        # Precompute the tables up to max_seq_len
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i, j -> i j", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for positions 0 … seq_len-1.

        Args:
            x: Any tensor on the target device (used only to infer device/dtype).
            seq_len: Actual sequence length for this forward pass.

        Returns:
            ``(cos, sin)`` each of shape (seq_len, head_dim).
        """
        if seq_len > self.max_seq_len:
            # Fallback to dynamic computation if seq_len exceeds max_seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split x along the last dim and rotate: [-x2, x1] → standard RoPE half-rotation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: Query tensor, shape (B, H, S, D).
        k: Key tensor, shape (B, H, S, D).
        cos: Cosine table, shape (S, D).
        sin: Sine table, shape (S, D).

    Returns:
        Rotated ``(q_embed, k_embed)``.
    """
    # Align cos/sin to the device AND dtype of q.
    # Precomputed buffers live on CPU until the first CUDA forward; without this
    # an explicit device mismatch crashes under torch.compile.
    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head causal self-attention with RoPE and Flash Attention.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability (applied during training only).
        max_seq_len: Maximum sequence length for the RoPE table.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor, shape (B, S, d_model).

        Returns:
            Output tensor, shape (B, S, d_model).
        """
        B, S, _ = x.shape

        # (B, S, 3*d) → (3, B, H, S, D)
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self.rotary_emb(x, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # (B, H, S, D) → (B, S, d_model)
        out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.proj(out)
