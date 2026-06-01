# Inference Memory Benchmark

## Purpose
Profiles peak activation memory to prove that inference memory scales O(1) with depth for VEGA/RR, while naive block structures slope linearly.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `vega`: Sliding-Window Depth Attention with Low-Rank History.
- `block_attnres`: Block Attention Residuals.

## Run
```bash
modal run --detach benchmarks/inference_memory/modal_inference_memory.py
```
