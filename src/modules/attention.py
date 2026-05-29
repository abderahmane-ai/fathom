import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int):
        # x is just used to get the device
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # Cast to input dtype for mixed-precision compatibility
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0) 
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 4096) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # Native PyTorch reshape (Guaranteed torch.compile fullgraph=True compatibility)
        # Shape: (B, S, 3 * d_model) -> (B, S, 3, H, D) -> (3, B, H, S, D)
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply RoPE to Queries and Keys
        cos, sin = self.rotary_emb(x, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash / memory-efficient scaled dot-product attention
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Concatenate heads and project
        # Shape: (B, H, S, D) -> (B, S, H, D) -> (B, S, d_model)
        out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.proj(out)
