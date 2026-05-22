# LM Quality Benchmark

## Purpose
Measures real causal language-model quality and throughput for Standard residuals,
Recurrent Residuals, and Block Attention Residuals on TinyStories. Wikitext can be
used by editing `config.yaml` data fields.

## Modes
- `standard`: baseline Pre-LN residual transformer.
- `recurrent_residual`: RR depth memory mechanism.
- `block_attnres`: paper-aligned Block AttnRes with `block_size` counted in sublayers.

## Metrics
The run logs train loss, validation loss, perplexity, tokens per second, wall time,
peak CUDA memory, parameter count, and RR gate statistics when available.

## Run
```bash
modal run --detach benchmarks/lm_quality/modal_lm_quality.py
```

Use `--wait true` to block until remote jobs finish.

## Artifacts
Artifacts are written to:

```text
${BENCHMARK_ARTIFACT_ROOT:-benchmarks/artifacts}/lm_quality/<mode>/<run_id>/
  checkpoints/
  logs/
  metrics/
  status.json
```

