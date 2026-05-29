# Scaling Efficiency Benchmark

## Purpose
Sweeps small depth/width configurations under a hard 60M-parameter cap to compare
loss, speed, and memory efficiency. This tests whether RR can use depth without
Block AttnRes cache overhead.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `swda_lr`: Sliding-Window Depth Attention with Low-Rank History.
- `block_attnres`: paper-aligned Block Attention Residuals.

## Metrics
The run logs loss per parameter, loss per tokens/sec, validation loss, throughput,
wall time, peak CUDA memory, parameter count, and run status for every sweep point.

## Run
```bash
modal run --detach benchmarks/scaling_efficiency/modal_scaling_efficiency.py
```

Use `--wait true` to block until all remote jobs finish.

## Artifacts
Artifacts are written to:

```text
${BENCHMARK_ARTIFACT_ROOT:-benchmarks/artifacts}/scaling_efficiency/<mode>/<run_id>-d<d_model>-l<num_layers>/
  checkpoints/
  logs/
  metrics/
  status.json
```

