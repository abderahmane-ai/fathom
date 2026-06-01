# Depth vs. Width IsoFLOP Tradeoff Benchmark

## Purpose
Evaluates scaling laws by comparing Wide & Shallow architectures against Narrow & Deep architectures with exactly matched parameter counts and FLOPs. Tests whether VEGA/RR unlocks deep network scaling where standard Pre-LN architectures suffer from depth dilution.

## Modes
- `wide_shallow_std`: Standard architecture (6 Layers, d_model=1024)
- `narrow_deep_vega`: VEGA architecture (24 Layers, d_model=512)
- `narrow_deep_rr`: Recurrent Residual architecture (24 Layers, d_model=512)

## Run
```bash
modal run --detach benchmarks/iso_flop/modal_iso_flop.py
```
