# Recurrent Residuals, VEGA, and Attention Residuals

[![CI](https://github.com/abderahmane-ai/fathom/actions/workflows/ci.yml/badge.svg)](https://github.com/abderahmane-ai/fathom/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-276_passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)

A controlled comparison of **four** depth-stream residual mechanisms for causal transformer language models, organized as a **design ladder** of progressively richer approximations of the same operation: *letting a layer reach back into the history of hidden states produced earlier in the depth stream.*

## What Is Being Compared

Standard transformer residuals (`h = h + y`) accumulate every layer's output with equal weight, which dilutes early-layer signals in deep networks. This project benchmarks **four** mechanisms against the standard baseline:

| Rung | Mechanism | History representation | Complexity per Layer |
|---:|---|---|---|
| 0 | **Standard** | $h = h + y$ (no history) | $O(d)$ |
| 1 | **Recurrent Residuals (RR)** | Single gated memory vector | $O(r \cdot d)$ |
| 2 | **VEGA** | Multi-head linear-attention state (depth-axis linear attention) | $O(n_h \, r_h \, d)$ |
| 3 | **Attention Residuals (AttnRes)** | Softmax aggregation over previous block states; [Moonshot AI, arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | $O(B \cdot d)$ per block |

Rungs 1–3 form the history-aggregation cost/expressivity frontier: **VEGA is to AttnRes what RWKV is to softmax attention** — same query-conditioned retrieval idea, but a closed-form linear recurrence over a fixed-size state at $O(L)$ total cost instead of an explicit softmax at $O(L^2)$. Full mathematical derivations and the design-ladder framing are in [METHODOLOGY.md](METHODOLOGY.md).

## Installation

```bash
git clone https://github.com/abderahmane-ai/fathom.git
cd fathom

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
python src/train.py model=block_attnres

# Override any parameter
python src/train.py model.d_model=512 trainer.precision=bf16-mixed
```

## Benchmarks

Four evaluation tasks are available via Modal.

| Benchmark | Task | Metric |
|---|---|---|
| `lm_quality` | TinyStories language modelling | Validation perplexity |
| `depth_preservation` | Linear probing of intermediate layers | DRI, GPI |
| `scaling_efficiency` | Pareto frontier across model sizes | Val perplexity vs. parameters |
| `depth_needle` | Synthetic token-retrieval at depth | Accuracy |

```bash
# Run a benchmark (requires modal auth)
modal run benchmarks/lm_quality/modal_lm_quality.py
modal run benchmarks/depth_preservation/modal_depth_preservation.py
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
read_gate  = σ(read_up(read_down(RMSNorm(y))) + depth_bias[pos])
damp_gate  = σ(damp_up(damp_down(RMSNorm(y))) + depth_bias[pos])
forget_gate= σ(forget_proj(RMSNorm(m)) + depth_bias[pos])
update_gate= σ(update_up(update_down(RMSNorm(y))) + depth_bias[pos])

h_new = damp_gate * h_prev + y + read_gate * (memory_gain * memory_out(RMSNorm(m)))
m_new = forget_gate * m + update_gate * tanh(y)
```

All gate weights are low-rank (d → rank → d). All gates start near their identity value (zero-start compliant).

### VEGA

```
K = key_proj(RMSNorm(y)),  V = val_proj(RMSNorm(y)),  Q = query_proj(RMSNorm(y))
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
	logits  = pseudo_query · RMSNorm(values)                         [per position, over depth axis; no √d — paper's formula]
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
scripts/
  ingest/               # Walk the artifact root -> per-benchmark CSVs
  plots/                # PNG+PDF publication-quality plots
  tables/               # Per-benchmark markdown summary tables
  render_summary.py     # Build a top-level SUMMARY.md from per-benchmark ones
tests/
```

## Reporting (plots + tables from Modal artifacts)

After a benchmark run (the Modal volume is mounted at `/artifacts` by default in the
Docker image), pull it back to a local directory and run:

```bash
make report ARTIFACT_ROOT=./artifacts
```

This drives the full reporting pipeline:

1. **Ingest** (`scripts/ingest/collect.py`): walks the artifact root, reads
   `run.json` / `status.json` / `dps.json` / `metrics.csv` per run, and writes
   per-benchmark CSVs to `results/aggregate/`.
2. **Plots** (`scripts/plots/*.py`): reads the CSVs and writes PNG+PDF figures
   to `plots/<benchmark>/`.  All plots share a common style (seaborn colorblind,
   DejaVu Sans, 4×3 inches at 150 DPI).
3. **Tables** (`scripts/tables/*.py`): reads the CSVs and writes per-benchmark
   GitHub-flavored markdown summaries to `results/<benchmark>/SUMMARY.md`.  The
   best value in each column is bolded.
4. **Render** (`scripts/render_summary.py`): concatenates the per-benchmark
   summaries into a top-level `results/SUMMARY.md` with a TOC and a
   one-paragraph description per benchmark.

Individual sub-pipelines can be run as `make ingest`, `make plots`, `make tables`,
or `make summary`.  Override `ARTIFACT_ROOT=...` to point at a different volume
or local copy.

The reporting pipeline is end-to-end smoke-tested by `tests/test_render_summary.py`
and the per-table tests in `tests/test_tables.py`.

## Citation

```bibtex
@misc{fathom2026,
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
