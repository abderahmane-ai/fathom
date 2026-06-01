# Natural Text Passkey Retrieval (NIAH) Benchmark

## Purpose
Needle-In-A-Haystack (NIAH) on Natural Text. Proves that VEGA and RR can preserve discrete, arbitrary information through the "noise" of real semantic processing, not just synthetic random tokens.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `vega`: Sliding-Window Depth Attention with Low-Rank History.
- `block_attnres`: Block Attention Residuals.

## Run
```bash
modal run benchmarks/natural_niah/modal_natural_niah.py --lm-run-id <run_id>
```
