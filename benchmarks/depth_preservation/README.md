# Depth Preservation Score (DPS) Benchmark

## Purpose
Measures the preservation of activation identity and alignment across layers for untrained models under different residual stream architectures. It evaluates Depth Preservation Score (DPS) and Gradient Preservation Score (GPS) using closed-form linear probes to check signal propagation decay across depth.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `vega`: Sliding-Window Depth Attention with Low-Rank History.
- `block_attnres`: Block Attention Residuals.

## Metrics
The run logs Depth Preservation Score (DPS) per layer, Gradient Preservation Score (GPS) per layer, mean dissimilarity, Depth Retention Index (DRI), and Gradient Preservation Index (GPI) across all intermediate layers.

## Run
```bash
modal run --detach benchmarks/depth_preservation/modal_dps.py
```

Use `--wait true` to block until remote jobs finish.

## Artifacts
Artifacts are written to:

```text
${BENCHMARK_ARTIFACT_ROOT:-benchmarks/artifacts}/results/depth_preservation/<run_id>/
  <mode>_dps.json
```
