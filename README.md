# Recurrent Residuals, VEGA, and Attention Residuals

A controlled comparison of **five** depth-stream residual mechanisms for causal transformer language models, organized as a **design ladder** of progressively richer approximations of the same operation: *letting a layer reach back into the history of hidden states produced earlier in the depth stream.*

## What Is Being Compared

Standard transformer residuals (`h = h + y`) accumulate every layer's output with equal weight, which dilutes early-layer signals in deep networks. This project benchmarks **five** mechanisms against the standard baseline:

| Rung | Mechanism | History representation | Complexity per Layer |
|---:|---|---|---|
| 0 | **Standard** | `h = h + y` (no history) | O(d) |
| 1 | **Recurrent Residuals (RR)** | Single gated memory vector | O(rank · d) |
| 2 | **VEGA** | Multi-head linear-attention state (depth-axis linear attention) | O(n_heads · r_head · d) |
| 3 | **mHC** (DeepSeek, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)) | Parallel residual channels with input-dependent pre/post/residual mixing, doubly-stochastic via Sinkhorn-Knopp. Supports `num_channels=2` and `num_channels=4` (paper's production choice). | O(m²·d + m·d·m²) per sublayer |
| 3' | **mHC-Lite** ([arXiv:2601.05732](https://arxiv.org/abs/2601.05732)) | Same as mHC, but $H_\text{res}$ is a convex combination of permutation matrices (Birkhoff-von Neumann). n=2 only because $n!$ blows up. | O(m²·d + m·d·m!) per sublayer |
| 4 | **Attention Residuals (AttnRes)** | Softmax aggregation over previous block states; [Moonshot AI, arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | O(B · d) per block |

Rungs 1–4 form a cost/expressivity frontier: **VEGA is to AttnRes what RWKV is to softmax attention** — same query-conditioned retrieval idea, but a closed-form linear recurrence over a fixed-size state at $O(L)$ total cost instead of an explicit softmax at $O(L^2)$. mHC and mHC-Lite are included as orthogonal concurrent-work baselines (parallel channels, not history aggregation); the default config uses the original DeepSeek Sinkhorn-Knopp algorithm. Full mathematical derivations and the design-ladder framing are in [METHODOLOGY.md](METHODOLOGY.md).

## Installation

```bash
git clone https://github.com/your-repo/recurrent-residuals.git
cd recurrent-residuals

# Core library
pip install .

# Development tools (pytest, ruff, pyrefly)
pip install ".[dev]"

# Benchmark runners (modal, wandb)
pip install ".[benchmarks]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

## Quick Start

```bash
# Standard baseline
python src/train.py model=standard

# Recurrent Residuals
python src/train.py model=recurrent_residual

# VEGA
python src/train.py model=vega

# Block Attention Residuals (Moonshot AI)
python src/train.py model=attnres

# Override any parameter
python src/train.py model.d_model=512 trainer.precision=bf16-mixed
```

## Benchmarks

Four evaluation tasks are available via Modal. Each task runs all four modes in parallel on A100 GPUs.

| Benchmark | Task | Metric |
|---|---|---|
| `lm_quality` | TinyStories language modelling | Validation perplexity |
| `depth_preservation` | Linear probing of intermediate layers | DRI, GPI |
| `scaling_efficiency` | Pareto frontier across model sizes | Val perplexity vs. parameters |
| `depth_needle` | Synthetic token-retrieval at depth | Accuracy |

```bash
# Run a benchmark (requires modal auth)
modal run benchmarks/lm_quality/modal_lm_quality.py
modal run benchmarks/depth_preservation/modal_dps.py
modal run benchmarks/depth_needle/modal_depth_needle.py
```

## Testing

```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

## Residual Mechanisms — Key Equations

### Recurrent Residuals (RR)

```
read_gate  = σ(read_proj(y)  + depth_bias[pos])
damp_gate  = σ(damp_proj(y)  + depth_bias[pos])
forget_gate= σ(forget_proj(RMSNorm(m)) + depth_bias[pos])
update_gate= σ(update_proj(y) + depth_bias[pos])

h_new = damp_gate * h_prev + y + read_gate * (memory_gain * memory_out(RMSNorm(m)))
m_new = forget_gate * m + update_gate * tanh(y)
```

All gate weights are low-rank (d → rank → d). All gates start near their identity value (zero-start compliant).

### VEGA

```
K = key_proj(y),  V = val_proj(y),  Q = query_proj(y)
K_dep = RMSNorm(K) / sqrt(r_head)
Q_dep = RMSNorm(Q + query_bias[pos]) / sqrt(r_head)
φ(x) = ELU(x) + 1

# If r_head <= 64 (Vector state):
c = (φ(Q_dep) * S_prev) / sum(φ(Q_dep) * z_prev + ε)
S_new = σ(decay) * S_prev + φ(K_dep) * (write_gate * V)

# If r_head >= 128 (Matrix state):
c = φ(Q_dep) S_prev / (φ(Q_dep) z_prev + ε)
S_new = σ(decay) * S_prev + φ(K_dep) ⊗ (write_gate * V)

# Hidden state update:
h_new = σ(damp) * h_prev + y + read_gate * W_out(RMSNorm(c))
z_new = σ(decay) * z_prev + φ(K_dep)
```

Uses conditional Vector/Matrix state based on head rank, query-only bias, and soft decay.

### Block Attention Residuals (AttnRes — Moonshot AI)

```
values  = stack([*block_history, current_block])
logits  = (pseudo_query · RMSNorm(values)) / sqrt(d_model)      [per position, over depth axis]
weights = softmax(logits)
h_new   = sum(weights * values)
```

`pseudo_query` starts at zero → uniform aggregation → degrades to a mean residual.

## Project Structure

```
src/
  modules/
    norm.py              # RMSNorm (single authoritative implementation)
    attention.py         # Multi-head causal self-attention with RoPE
    ffn.py               # SwiGLU feed-forward network
    recurrent_residual.py # RR gated depth-memory cell
    vega.py              # VEGA multi-scale EMA cell
    attnres_block.py     # BlockAttnRes and FullAttnRes
    transformer_layer.py # Universal layer (all modes)
    transformer.py       # TransformerDecoder with weight-tied LM head
  data.py
  train.py
benchmarks/
  common/               # Lightning engine, artifact management, metrics
  lm_quality/
  depth_preservation/
  depth_needle/
  scaling_efficiency/
conf/                   # Hydra YAML configs
tests/
```

## Citation

```bibtex
@misc{recurrent-residuals2026,
  author    = {Abdou Magico},
  title     = {Recurrent Residuals, VEGA, and Attention Residuals for Deep Transformers},
  year      = {2026},
  publisher = {GitHub},
}

@misc{kimi2025attnres,
  title  = {Attention Residuals},
  author = {Kimi Team / Moonshot AI},
  year   = {2025},
  url    = {https://arxiv.org/abs/2603.15031},
}
```
