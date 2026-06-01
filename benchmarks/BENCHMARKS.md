# Benchmarks Suite Documentation

The `benchmarks` directory contains a comprehensive suite of tools designed to evaluate and compare different residual stream architectures in transformer models.

It focuses on testing baseline Pre-LN residual transformers, Recurrent Residual (RR) mechanisms, Sliding-Window Depth Attention with Low-Rank History (Vega), and Block Attention Residuals (`block_attnres`).

## Structure Overview

### 1. Common Utilities (`benchmarks/common/`)
Contains shared infrastructure used across all benchmarks:
- **`artifacts.py`**: Manages saving and organizing benchmark artifacts like checkpoints, logs, and metrics.
- **`configs.py`**: Shared configuration management.
- **`dps_extractor.py`**: Extracts Depth Preservation Scores and related metrics.
- **`lightning_engine.py`**: PyTorch Lightning integration and runner logic.
- **`metrics.py`**: Common metric definitions.
- **`modal_utils.py`**: Utilities for running tasks on Modal (cloud execution).
- **`param_count.py`**: Tools for calculating model parameter counts accurately.

### 2. Depth Needle Benchmark (`benchmarks/depth_needle/`)
**Purpose:** Tests whether residual mechanisms preserve an early payload token across long sequence distances and many transformer layers. It is a targeted diagnostic for depth-wise information persistence.
- **Run Command:** `modal run --detach benchmarks/depth_needle/modal_depth_needle.py`

### 3. Depth Preservation Score (DPS) Benchmark (`benchmarks/depth_preservation/`)
**Purpose:** Measures the preservation of activation identity and alignment across layers for untrained models.
- **Run Command:** `modal run --detach benchmarks/depth_preservation/modal_depth_preservation.py`

### 4. LM Quality Benchmark (`benchmarks/lm_quality/`)
**Purpose:** Measures real causal language-model quality and throughput on TinyStories (or Wikitext).
- **Run Command:** `modal run --detach benchmarks/lm_quality/modal_lm_quality.py`

### 5. Scaling Efficiency Benchmark (`benchmarks/scaling_efficiency/`)
**Purpose:** Sweeps small depth/width configurations under a hard 60M-parameter cap to compare loss, speed, and memory efficiency.
- **Run Command:** `modal run --detach benchmarks/scaling_efficiency/modal_scaling_efficiency.py`

### 6. Natural Text Passkey Retrieval (`benchmarks/natural_niah/`)
**Purpose:** Needle-In-A-Haystack (NIAH) on Natural Text. Proves that VEGA and RR can preserve discrete, arbitrary information through the "noise" of real semantic processing.
- **Run Command:** `modal run --detach benchmarks/natural_niah/modal_natural_niah.py`

### 7. Depth vs. Width IsoFLOP Tradeoff (`benchmarks/iso_flop/`)
**Purpose:** Evaluates scaling laws by comparing Wide & Shallow architectures against Narrow & Deep architectures with exactly matched parameter counts and FLOPs.
- **Run Command:** `modal run --detach benchmarks/iso_flop/modal_iso_flop.py`

### 8. Ablation Suite (`benchmarks/ablation/`)
**Purpose:** Targeted Component Removal runs to prove that components like VEGA Variance Regularization, Multi-Scale Initialization, and RR Depth Biases are strictly necessary.
- **Run Command:** `modal run --detach benchmarks/ablation/modal_ablation.py`

### 9. Inference Memory Profiling (`benchmarks/inference_memory/`)
**Purpose:** Profiles peak activation memory to prove that inference memory scales $O(1)$ with depth for VEGA/RR, while naive block structures slope linearly.
- **Run Command:** `modal run --detach benchmarks/inference_memory/modal_inference_memory.py`

---
*Note: All benchmarks write their artifacts to a configured artifact root directory (defaulting to `benchmarks/artifacts/`). All benchmarks support execution on Modal cloud infrastructure for distributed training and evaluation.*