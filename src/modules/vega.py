import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class VEGACell(nn.Module):
    """Vertical EMA Gated Attention (VEGA).
    
    Multi-scale depth memory with two explicit timescale groups (fast / slow)
    implemented as head partitions within a single linear-attention EMA state.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        rank: int = 32,
        n_heads: int = 4,
        n_fast_heads: int = 2,
        fast_decay_range: tuple[float, float] = (0.0, 1.2),
        slow_decay_range: tuple[float, float] = (2.0, 4.5),
        read_gate_bias: float = -3.0,
        write_gate_bias: float = -2.0,
        damp_gate_bias: float = 3.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()

        assert rank % n_heads == 0, "rank must be divisible by n_heads"
        assert 0 < n_fast_heads < n_heads, "n_fast_heads must be in (0, n_heads)"

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.rank = rank
        self.n_heads = n_heads
        self.n_fast_heads = n_fast_heads
        self.r_head = rank // n_heads
        self.eps = eps

        # Projections
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        self.v_proj = nn.Linear(d_model, rank, bias=False)
        self.q_proj = nn.Linear(d_model, rank, bias=False)
        self.write_gate = nn.Linear(d_model, rank, bias=True)

        self.read_fast = nn.Linear(d_model, d_model, bias=True)
        self.read_slow = nn.Linear(d_model, d_model, bias=True)
        self.damp_weight = nn.Parameter(torch.empty(d_model))
        self.damp_bias = nn.Parameter(torch.full((d_model,), damp_gate_bias))

        self.out_fast = nn.Linear(n_fast_heads * self.r_head, d_model, bias=False)
        self.out_slow = nn.Linear((n_heads - n_fast_heads) * self.r_head, d_model, bias=False)
        self.norm_fast = RMSNorm(n_fast_heads * self.r_head, eps=eps)
        self.norm_slow = RMSNorm((n_heads - n_fast_heads) * self.r_head, eps=eps)

        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))
        self.key_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        self.decay = nn.Parameter(torch.empty(self.num_sublayers, n_heads, self.r_head))
        with torch.no_grad():
            for pos in range(self.num_sublayers):
                fast_logits = torch.linspace(*fast_decay_range, n_fast_heads * self.r_head)
                slow_logits = torch.linspace(*slow_decay_range, (n_heads - n_fast_heads) * self.r_head)
                self.decay[pos, :n_fast_heads] = fast_logits.view(n_fast_heads, self.r_head)
                self.decay[pos, n_fast_heads:] = slow_logits.view(n_heads - n_fast_heads, self.r_head)

        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.normal_(self.write_gate.weight, 0.0, gate_init_std)
        self.write_gate.bias.data.fill_(write_gate_bias)
        nn.init.normal_(self.read_fast.weight, 0.0, gate_init_std)
        self.read_fast.bias.data.fill_(read_gate_bias)
        nn.init.normal_(self.read_slow.weight, 0.0, gate_init_std)
        self.read_slow.bias.data.fill_(read_gate_bias)
        nn.init.normal_(self.damp_weight, 0.0, gate_init_std)
        nn.init.zeros_(self.out_fast.weight)
        nn.init.zeros_(self.out_slow.weight)

    def get_initial_state(self, B: int, S: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        S0 = torch.zeros(B, S, self.n_heads, self.r_head, self.r_head, device=device, dtype=torch.float32)
        z0 = torch.zeros(B, S, self.n_heads, self.r_head, device=device, dtype=torch.float32)
        return S0, z0

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        sublayer: int = 0,
        h_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        S_prev, z_prev = state
        pos = layer_idx * 2 + sublayer
        B, Seq, _ = y.shape

        K = self.q_proj(y).view(B, Seq, self.n_heads, self.r_head)
        V = self.v_proj(y).view(B, Seq, self.n_heads, self.r_head)
        Q = self.k_proj(y).view(B, Seq, self.n_heads, self.r_head)
        g = torch.sigmoid(self.write_gate(y)).view(B, Seq, self.n_heads, self.r_head)

        K_dep = K + self.key_bias[pos].view(1, 1, self.n_heads, self.r_head)
        Q_dep = Q + self.query_bias[pos].view(1, 1, self.n_heads, self.r_head)

        K_phi = (F.elu(K_dep) + 1.0).float()
        Q_phi = (F.elu(Q_dep) + 1.0).float()

        num = torch.matmul(Q_phi.unsqueeze(-2), S_prev.float()).squeeze(-2)
        den = (Q_phi * z_prev.float()).sum(-1, keepdim=True).clamp(min=self.eps)
        c = (num / den).to(y.dtype)

        c_fast = c[:, :, :self.n_fast_heads, :].contiguous().view(B, Seq, -1)
        c_slow = c[:, :, self.n_fast_heads:, :].contiguous().view(B, Seq, -1)
        c_out = self.out_fast(self.norm_fast(c_fast)) + self.out_slow(self.norm_slow(c_slow))

        r_fast = torch.sigmoid(self.read_fast(y))
        r_slow = torch.sigmoid(self.read_slow(y))
        damp = torch.sigmoid(self.damp_weight * y + self.damp_bias)

        h_new = damp * h_prev + y + r_fast * c_out + r_slow * c_out
        decay = torch.sigmoid(self.decay[pos]).view(1, 1, self.n_heads, self.r_head)

        outer = K_phi.unsqueeze(-1) * (g * V.float()).unsqueeze(-2)
        return h_new, (decay.unsqueeze(-1) * S_prev + outer, decay * z_prev + K_phi)
