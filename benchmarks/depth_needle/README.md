# Depth Needle Benchmark

## Purpose
Tests whether residual mechanisms preserve an early payload token across long
sequence distance and many transformer layers. This is a targeted diagnostic for
depth-wise information persistence, not a replacement for language-model loss.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `block_attnres`: practical Attention Residuals comparison target.
- `full_attnres`: tiny correctness reference that stores every previous state.

## Metrics
The run logs needle loss, validation loss, convergence step, tokens per second,
peak CUDA memory, parameter count, RR gate statistics, and activation/gradient
signals emitted by the Lightning runner.

## Run
```bash
modal run --detach benchmarks/depth_needle/modal_depth_needle.py
```

Use `--include-full true` to include the small `full_attnres` reference run.

## Artifacts
Artifacts are written to:

```text
${BENCHMARK_ARTIFACT_ROOT:-benchmarks/artifacts}/depth_needle/<mode>/<run_id>/
  checkpoints/
  logs/
  metrics/
  status.json
```

