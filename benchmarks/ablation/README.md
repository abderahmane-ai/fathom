# Ablation Suite Benchmark

## Purpose
Targeted Component Removal runs using the Depth Needle task to prove that components like VEGA Variance Regularization, Multi-Scale Initialization, and RR Depth Biases are strictly necessary for optimal performance.

## Modes
- `vega_no_var_reg`: VEGA without Variance Regularization on decay.
- `vega_no_multiscale`: VEGA initialized uniformly (no multi-scale log-linear init).
- `rr_no_depth_biases`: RR without layer-specific depth biases.

## Run
```bash
modal run --detach benchmarks/ablation/modal_ablation.py
```
