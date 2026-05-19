# Modal benchmarks

Parallel **Recurrent Residual** vs **Block AttnRes** on 2× A100 (one GPU per job).

## Detached (default)

```bash
modal run benchmarks/modal_entrypoint.py
```

Writes `benchmarks/last_spawn.json` with Modal object IDs.

## Optional: wait for completion

```bash
modal run benchmarks/modal_entrypoint.py --wait
```

## Artifacts (volume `rr-benchmark-artifacts`)

| Path | Contents |
|------|----------|
| `/artifacts/<mode>/status.json` | `running` / `completed` / `failed`, step, errors |
| `/artifacts/<mode>/checkpoints/` | `last.ckpt` + top-3 by `val/loss` (every 500 steps) |
| `/artifacts/<mode>/csv_logs/` | CSV metrics |
| `/artifacts/hf_cache/` | HF dataset cache (survives restarts) |

```bash
modal volume ls rr-benchmark-artifacts
```

## Resume

Re-run the same mode; auto-resumes from `last.ckpt` when present (`BENCHMARK_RESUME=1`).
