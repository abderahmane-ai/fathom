"""Walk the artifact root and emit a per-benchmark aggregate CSV.

Usage:
    python -m scripts.ingest.collect \\
        --root /artifacts \\
        --out results/aggregate \\
        [--benchmarks lm_quality,depth_preservation,...]

This walks the canonical layout ``<root>/<benchmark>/<mode>/<run_id>/`` and
emits one CSV per benchmark type:
  * results/aggregate/lm_quality.csv   (from metrics/summary.json + metrics.csv)
  * results/aggregate/depth_preservation.csv  (from <mode>_dps.json)
  * results/aggregate/inference_memory.csv    (from profile_results.json)
  * results/aggregate/natural_niah.csv        (from niah_result.json)
  * results/aggregate/all_runs.csv            (master index: one row per status.json)

Falls back to the legacy layout ``<root>/results/<benchmark>/<run_id>/<mode>_dps.json``
and ``<root>/<benchmark>/<run_id>/profile_results.json`` so previously-run
artifacts continue to work.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

from scripts.ingest.schemas import (
    DPSResult,
    LatencyProfile,
    LMRunStep,
    LMRunSummary,
    NIAHResult,
)

log = logging.getLogger(__name__)

ALL_BENCHMARKS = [
    "ablation",
    "depth_needle",
    "depth_preservation",
    "inference_memory",
    "iso_flop",
    "lm_quality",
    "natural_niah",
    "scaling_efficiency",
]

TRAINING_BENCHMARKS = {"ablation", "depth_needle", "iso_flop", "lm_quality", "scaling_efficiency"}


def collect_all_runs(root: Path, benchmark_name: str) -> list[dict[str, Any]]:
    """Walk ``<root>/<benchmark>/<mode>/<run_id>/status.json`` and return one row per file.

    Each row contains the parsed status.json plus ``_path`` and ``_run_dir``.
    """
    rows: list[dict[str, Any]] = []
    base = root / benchmark_name
    if not base.is_dir():
        return rows
    for status_file in sorted(base.glob("*/*/status.json")):
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping corrupt status file %s: %s", status_file, exc)
            continue
        payload["_path"] = str(status_file)
        payload["_run_dir"] = str(status_file.parent)
        rows.append(payload)
    return rows


def collect_dps(root: Path, benchmark_name: str = "depth_preservation") -> list[DPSResult]:
    """Walk ``<root>/<benchmark>/<mode>/<run_id>/dps.json`` and parse each one.

    Falls back to the legacy ``<root>/results/<benchmark>/<run_id>/<mode>_dps.json`` layout.
    """
    out: list[DPSResult] = []
    seen: set[str] = set()
    base = root / benchmark_name
    if base.is_dir():
        for dps_file in sorted(base.glob("*/*/dps.json")):
            try:
                out.append(DPSResult.from_json(dps_file))
                seen.add(str(dps_file))
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                log.warning("Skipping corrupt DPS file %s: %s", dps_file, exc)
    legacy = root / "results" / benchmark_name
    if legacy.is_dir():
        for dps_file in sorted(legacy.glob("*/*_dps.json")):
            if str(dps_file) in seen:
                continue
            try:
                out.append(DPSResult.from_json(dps_file))
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                log.warning("Skipping legacy DPS file %s: %s", dps_file, exc)
    return out


def collect_latency(root: Path, benchmark_name: str = "inference_memory") -> list[LatencyProfile]:
    """Walk ``<root>/<benchmark>/<run_id>/profile_results.json`` and parse."""
    out: list[LatencyProfile] = []
    base = root / benchmark_name
    if not base.is_dir():
        return out
    for profile_file in sorted(base.glob("*/profile_results.json")):
        try:
            out.extend(LatencyProfile.from_json(profile_file))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Skipping corrupt profile file %s: %s", profile_file, exc)
    return out


def collect_niah(root: Path, benchmark_name: str = "natural_niah") -> list[NIAHResult]:
    """Walk ``<root>/<benchmark>/<mode>/<run_id>/niah_result.json``."""
    out: list[NIAHResult] = []
    base = root / benchmark_name
    if not base.is_dir():
        return out
    for niah_file in sorted(base.glob("*/*/niah_result.json")):
        try:
            out.append(NIAHResult.from_json(niah_file))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Skipping corrupt NIAH file %s: %s", niah_file, exc)
    return out


def collect_lm_summaries(root: Path, benchmark_name: str) -> list[LMRunSummary]:
    """Walk ``<root>/<benchmark>/<mode>/<run_id>/metrics/summary.json``."""
    out: list[LMRunSummary] = []
    base = root / benchmark_name
    if not base.is_dir():
        return out
    for summary_file in sorted(base.glob("*/*/metrics/summary.json")):
        try:
            out.append(LMRunSummary.from_json(summary_file))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Skipping corrupt summary %s: %s", summary_file, exc)
    return out


def collect_lm_steps(root: Path, benchmark_name: str) -> Iterable[LMRunStep]:
    """Walk ``<root>/<benchmark>/<mode>/<run_id>/logs/version_*/metrics.csv``."""
    base = root / benchmark_name
    if not base.is_dir():
        return
    for metrics_file in sorted(base.glob("*/*/logs/*/metrics.csv")):
        # path: <root>/<benchmark>/<mode>/<run_id>/logs/version_0/metrics.csv
        parts = metrics_file.parts
        try:
            run_id = parts[-4]
            residual_mode = parts[-5]
        except IndexError:
            log.warning("Unexpected path layout: %s", metrics_file)
            continue
        try:
            yield from LMRunStep.from_metrics_csv(
                metrics_file,
                benchmark_name=benchmark_name,
                residual_mode=residual_mode,
                run_id=run_id,
            )
        except (OSError, KeyError) as exc:
            log.warning("Skipping corrupt metrics.csv %s: %s", metrics_file, exc)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a CSV, using the union of keys as the header."""
    if not rows:
        log.info("No rows to write to %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def collect(root: Path, out_dir: Path, benchmarks: list[str]) -> dict[str, int]:
    """Walk ``root`` and emit one CSV per benchmark into ``out_dir``.

    Returns a dict mapping benchmark -> row_count.
    """
    counts: dict[str, int] = {}
    for benchmark in benchmarks:
        log.info("Collecting %s ...", benchmark)
        if benchmark == "depth_preservation":
            rows = [r.to_row() for r in collect_dps(root, benchmark)]
            write_csv(rows, out_dir / f"{benchmark}.csv")
            counts[benchmark] = len(rows)
        elif benchmark == "inference_memory":
            rows = [r.to_row() for r in collect_latency(root, benchmark)]
            write_csv(rows, out_dir / f"{benchmark}.csv")
            counts[benchmark] = len(rows)
        elif benchmark == "natural_niah":
            rows = [r.to_row() for r in collect_niah(root, benchmark)]
            write_csv(rows, out_dir / f"{benchmark}.csv")
            counts[benchmark] = len(rows)
        elif benchmark in TRAINING_BENCHMARKS:
            summaries = collect_lm_summaries(root, benchmark)
            summary_rows = [r.to_row() for r in summaries]
            write_csv(summary_rows, out_dir / f"{benchmark}_summary.csv")
            counts[f"{benchmark}_summary"] = len(summary_rows)
            steps = list(collect_lm_steps(root, benchmark))
            step_rows = [r.to_row() for r in steps]
            write_csv(step_rows, out_dir / f"{benchmark}_steps.csv")
            counts[f"{benchmark}_steps"] = len(step_rows)
        else:
            log.warning("Unknown benchmark: %s", benchmark)

    # Master all_runs.csv: one row per status.json
    all_rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        all_rows.extend(collect_all_runs(root, benchmark))
    write_csv(all_rows, out_dir / "all_runs.csv")
    counts["all_runs"] = len(all_rows)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="Artifact root (e.g. /artifacts or benchmarks/artifacts)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for CSVs")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(ALL_BENCHMARKS),
        help=f"Comma-separated list of benchmarks to collect (default: {','.join(ALL_BENCHMARKS)})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if not args.root.is_dir():
        log.error("Root directory does not exist: %s", args.root)
        return 1
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    counts = collect(args.root, args.out, benchmarks)
    log.info("Done. Row counts: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
