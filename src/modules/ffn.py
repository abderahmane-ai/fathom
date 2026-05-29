import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    Note: To match the parameter count of a standard GELU FFN, 
    set config.ff_dim = int(8/3 * d_model) rounded to the nearest multiple of 64.
    """
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False) # Gate projection
        self.w3 = nn.Linear(d_model, ff_dim, bias=False) # Up projection
        self.w2 = nn.Linear(ff_dim, d_model, bias=False) # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: w2(dropout(silu(w1(x)) * w3(x)))
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
