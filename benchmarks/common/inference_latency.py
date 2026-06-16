"""End-to-end inference latency profiling for residual-mechanism comparison.

Measures single forward-pass latency across a (model, batch_size, sequence
length) grid. Used to produce the latency-vs-depth figure that demonstrates
the O(L) inference cost of block_attnres vs. the O(1)-per-token cost of
RR / VEGA / mHC / standard.

Autoregressive decoding with KV caching is not modeled here because the
transformer does not implement a KV cache. The forward-pass latency is a
clean upper bound on per-token decode cost and is directly comparable
across residual mechanisms.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class LatencyResult:
    """Latency profile for one (model, batch_size, sequence_length) configuration.

    All time fields are in milliseconds. ``peak_vram_mb`` is in MiB and
    reported as 0.0 when CUDA is unavailable.

    Args:
        mean_ms: Mean forward-pass latency across measured runs.
        p50_ms: Median latency.
        p99_ms: 99th percentile latency.
        min_ms: Minimum latency.
        max_ms: Maximum latency.
        stddev_ms: Standard deviation across runs.
        tokens_per_second: Forward-pass throughput in tokens/second.
        peak_vram_mb: Peak allocated CUDA memory in MiB.
    """

    mean_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    tokens_per_second: float
    peak_vram_mb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_forward(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 50257,
    n_warmup: int = 3,
    n_runs: int = 20,
    device: str = "cuda",
) -> LatencyResult:
    """Profile single forward-pass latency for a model on synthetic input.

    Args:
        model: The transformer to profile. Must accept (B, S) input_ids.
        batch_size: Generation batch size.
        seq_len: Sequence length of the synthetic input.
        vocab_size: Vocabulary size for random token sampling.
        n_warmup: Warmup runs (not timed).
        n_runs: Number of measured runs to aggregate.
        device: "cuda" or "cpu".

    Returns:
        LatencyResult with mean, percentiles, throughput, and peak memory.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    x_template = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

    for _ in range(max(0, n_warmup)):
        with torch.no_grad():
            _ = model(x_template)
    _synchronize(dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings_ms: list[float] = []
    for _ in range(max(1, n_runs)):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            _ = model(x_template)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings_ms.append(elapsed * 1000.0)

    peak_vram_mb = 0.0
    if dev.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    mean_ms = statistics.fmean(timings_ms)
    total_tokens = batch_size * seq_len
    tokens_per_second = total_tokens / (mean_ms / 1000.0)
    sorted_timings = sorted(timings_ms)
    n = len(sorted_timings)
    p99_ms = sorted_timings[-1] if n < 100 else statistics.quantiles(sorted_timings, n=100)[98]

    return LatencyResult(
        mean_ms=mean_ms,
        p50_ms=statistics.median(sorted_timings),
        p99_ms=p99_ms,
        min_ms=sorted_timings[0],
        max_ms=sorted_timings[-1],
        stddev_ms=statistics.pstdev(sorted_timings) if n > 1 else 0.0,
        tokens_per_second=tokens_per_second,
        peak_vram_mb=peak_vram_mb,
    )


def latency_sweep(
    model: torch.nn.Module,
    depths: list[int],
    seq_len: int,
    batch_size: int = 1,
    vocab_size: int = 50257,
    n_warmup: int = 3,
    n_runs: int = 20,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Profile latency across a list of depths.

    IMPORTANT: This function does NOT rebuild the model at different depths —
    it profiles the same model repeatedly.  To compare latency across depths,
    construct a fresh model at each target depth and call ``profile_forward``
    individually.  The ``depth`` key in each result dict records the intended
    depth for bookkeeping but the measurements are all taken on the same model.

    Args:
        model: A transformer model (profiled as-is at its current depth).
        depths: List of depth labels to record in the results.
        seq_len: Sequence length to profile at.
        batch_size: Batch size.
        vocab_size: Vocabulary size for synthetic inputs.
        n_warmup: Warmup runs.
        n_runs: Measured runs.
        device: "cuda" or "cpu".

    Returns:
        List of dicts with keys ``depth``, ``latency_ms``, ``tokens_per_second``,
        ``peak_vram_mb``.  All entries measure the same model; the ``depth``
        field records the intended label only.
    """
    results: list[dict[str, Any]] = []
    for depth in depths:
        result = profile_forward(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            n_warmup=n_warmup,
            n_runs=n_runs,
            device=device,
        )
        results.append(
            {
                "depth": depth,
                "mean_ms": result.mean_ms,
                "p50_ms": result.p50_ms,
                "p99_ms": result.p99_ms,
                "tokens_per_second": result.tokens_per_second,
                "peak_vram_mb": result.peak_vram_mb,
            }
        )
    return results
