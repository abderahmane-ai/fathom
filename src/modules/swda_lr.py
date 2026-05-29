import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.scale * x

class SWDALRCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        window_size: int = 8,
        rank: int = 16,
        n_heads: int = 4,
        v_dim: int | None = None,
        decay_bias_init: float = 3.0,
        read_gate_bias: float = -3.0,
        write_gate_bias: float = -2.0,
        eps: float = 1e-5,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = num_layers * 2
        self.window_size = window_size
        self.rank = rank
        self.n_heads = n_heads
        self.v_dim = v_dim if v_dim is not None else d_model
        self.eps = eps

        if rank % n_heads != 0: raise ValueError(f"rank ({rank}) must be divisible by n_heads ({n_heads}).")
        if self.v_dim % n_heads != 0: raise ValueError(f"v_dim ({self.v_dim}) must be divisible by n_heads ({n_heads}).")

        self.r_head = rank // n_heads
        self.d_head = self.v_dim // n_heads

        # Fused Read and Write Gate parameters (Row 0: Read, Row 1: Write)
        self.gate_weights = nn.Parameter(torch.empty(2, d_model))
        self.gate_biases = nn.Parameter(torch.empty(2, d_model))

        # State decay parameters (Logarithmic timescale init)
        self.decay_bias = nn.Parameter(torch.empty(self.num_sublayers, rank))
        self.key_decay_bias = nn.Parameter(torch.empty(self.num_sublayers, rank))

        with torch.no_grad():
            for pos in range(self.num_sublayers):
                # Use decay_bias_init as the base for logarithmic timescale spacing, ensuring order
                start = min(decay_bias_init, math.log(float(self.num_sublayers)))
                end = max(decay_bias_init, math.log(float(self.num_sublayers)))
                ts = torch.exp(torch.linspace(start, end, rank))
                alpha_init = torch.clamp(1.0 - (1.0 / ts), 0.001, 0.999)
                decay_val = torch.log(alpha_init / (1.0 - alpha_init))
                self.decay_bias[pos].copy_(decay_val)
                self.key_decay_bias[pos].copy_(decay_val)

        # Learned Depth Pseudo-Queries (initialized to zero)
        self.query_bias = nn.Parameter(torch.zeros(self.num_sublayers, n_heads, self.r_head))

        # Local Projections
        self.q_local_proj = nn.Linear(d_model, d_model, bias=False)
        self.local_norm = RMSNorm(d_model, eps=eps)
        
        # RMSNorm for input path
        self.h_norm = RMSNorm(d_model, eps=eps)
        
        # Relative Depth Bias for FIFO attention
        self.fifo_depth_bias = nn.Parameter(torch.zeros(window_size))

        # Deep Projections (MLA-inspired)
        self.k_deep_proj = nn.Linear(d_model, rank, bias=False)
        self.v_deep_proj = nn.Linear(d_model, self.v_dim, bias=False)
        self.q_deep_proj = nn.Linear(d_model, rank, bias=False)
        
        self.deep_norm = RMSNorm(self.v_dim, eps=eps)

        if self.v_dim != d_model:
            self.out_proj = nn.Linear(self.v_dim, d_model, bias=False)
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=gate_init_std)
        else:
            # pyrefly: ignore [bad-assignment]
            self.out_proj = nn.Identity()

        # Initializations
        with torch.no_grad():
            self.gate_weights[0].normal_(mean=0.0, std=gate_init_std)
            self.gate_weights[1].normal_(mean=0.0, std=gate_init_std)
            self.gate_biases[0].fill_(read_gate_bias)
            self.gate_biases[1].fill_(write_gate_bias)

        nn.init.orthogonal_(self.k_deep_proj.weight)
        nn.init.orthogonal_(self.v_deep_proj.weight)
        nn.init.orthogonal_(self.q_deep_proj.weight)
        nn.init.normal_(self.q_local_proj.weight, mean=0.0, std=gate_init_std)

    def get_initial_state(self, batch_size: int, seq_len: int, device: torch.device | None = None) -> tuple:
        target_device = device if device is not None else self.gate_weights.device
        fifo_buf = torch.zeros(batch_size, seq_len, self.window_size, self.d_model, device=target_device, dtype=torch.float32)
        fifo_norm_buf = torch.zeros(batch_size, seq_len, self.window_size, self.d_model, device=target_device, dtype=torch.float32)
        
        fifo_idx = 0 
        
        S_init = torch.zeros(batch_size, seq_len, self.n_heads, self.r_head, self.d_head, device=target_device, dtype=torch.float32)
        z_init = torch.zeros(batch_size, seq_len, self.n_heads, self.r_head, device=target_device, dtype=torch.float32)
        return fifo_buf, fifo_norm_buf, fifo_idx, S_init, z_init

    def forward(
        self,
        h_prev: torch.Tensor,
        y: torch.Tensor,
        m: tuple,
        layer_idx: int,
        sublayer: int = 0,
        h_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        fifo_buf, fifo_norm_buf, fifo_idx, S_prev, z_prev = m
        position = layer_idx * 2 + sublayer
        B, S, d = h_prev.shape

        if h_norm is None:
            # Use custom RMSNorm
            h_norm = self.h_norm(h_prev)

        # Fused Read/Write Gates: Stack inputs and evaluate in one call
        gate_inputs = torch.stack([h_norm, y], dim=0)
        gates = torch.sigmoid(gate_inputs * self.gate_weights.view(2, 1, 1, -1) + self.gate_biases.view(2, 1, 1, -1))
        read_gate, write_gate = gates[0], gates[1]

        # Local FIFO sliding window attention
        write_idx = fifo_idx % self.window_size
        
        # Use functional index_copy to avoid in-place errors WITHOUT cloning the massive buffer
        idx_tensor = torch.tensor([write_idx], device=y.device)
        fifo_buf = fifo_buf.index_copy(2, idx_tensor, y.unsqueeze(2).to(fifo_buf.dtype))
        fifo_norm_buf = fifo_norm_buf.index_copy(2, idx_tensor, self.local_norm(y).unsqueeze(2).to(fifo_norm_buf.dtype))
        
        next_idx = fifo_idx + 1

        # Attention over FIFO
        keys_local = fifo_norm_buf # (B, S, W, d)
        q_local = self.q_local_proj(y) # (B, S, d)
        
        # Matmul-based attention
        logits = torch.matmul(keys_local, q_local.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_model)
        logits = logits + self.fifo_depth_bias.view(1, 1, -1)

        # Pure Python control flow for masking (Triggers exactly 1 recompile at step W, then static)
        if fifo_idx < self.window_size - 1:
            mask = torch.zeros(self.window_size, device=y.device)
            mask[write_idx + 1:] = float('-inf')
            logits = logits + mask.view(1, 1, -1)

        weights = torch.softmax(logits, dim=-1)
        
        # Matmul-based context aggregation
        c_local = torch.matmul(weights.unsqueeze(-2), fifo_buf).squeeze(-2)

        # Deep multi-head history
        K = self.k_deep_proj(y)
        # Apply Write Gate to Value projection before deep storage
        V = self.v_deep_proj(y * write_gate) 
        Q = self.q_deep_proj(h_norm)

        K_reshaped = K.view(B, S, self.n_heads, self.r_head)
        Q_reshaped = Q.view(B, S, self.n_heads, self.r_head)
        V_reshaped = V.view(B, S, self.n_heads, self.d_head)

        Q_biased = Q_reshaped + self.query_bias[position].view(1, 1, self.n_heads, self.r_head)

        K_phi = F.elu(K_reshaped) + 1.0
        Q_phi = F.elu(Q_biased) + 1.0

        Q_phi_f = Q_phi.float()
        K_phi_f = K_phi.float()
        V_gated_f = V_reshaped.float()

        # Linear Attention Math
        num = torch.matmul(Q_phi_f.unsqueeze(-2), S_prev).squeeze(-2)
        den = torch.sum(Q_phi_f * z_prev, dim=-1, keepdim=True)
        c_deep = num / (den + self.eps)

        c_deep = c_deep.view(B, S, self.v_dim)
        c_deep = self.deep_norm(c_deep)
        c_deep = self.out_proj(c_deep.to(y.dtype))

        # Memory Injection
        h_new = h_prev + y + read_gate * (c_local + c_deep)

        # State updates
        decay = torch.sigmoid(self.decay_bias[position]).view(self.n_heads, self.r_head)
        key_decay = torch.sigmoid(self.key_decay_bias[position]).view(self.n_heads, self.r_head)

        outer = torch.matmul(K_phi_f.unsqueeze(-1), V_gated_f.unsqueeze(-2))
        
        # Fused EMA updates
        S_new = torch.addcmul(outer, decay.view(1, 1, self.n_heads, self.r_head, 1), S_prev)
        z_new = torch.addcmul(K_phi_f, key_decay.view(1, 1, self.n_heads, self.r_head), z_prev)

        return h_new, (fifo_buf, fifo_norm_buf, next_idx, S_new, z_new)
