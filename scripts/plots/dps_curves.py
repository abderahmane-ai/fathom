"""Plot Depth Preservation Score (DPS) and Gradient Preservation Score (GPS) per layer.

Reads ``results/aggregate/depth_preservation.csv`` (produced by
``scripts.ingest.collect``) and produces:
  * plots/depth_preservation/dps_vs_layer.png  + .pdf
  * plots/depth_preservation/gps_vs_layer.png  + .pdf

One line per residual mode, x-axis is layer index, y-axis is the score in
[0, 1].  Markers are colored by mode.

Usage:
    python -m scripts.plots.dps_curves \\
        --csv results/aggregate/depth_preservation.csv \\
        --out plots/depth_preservation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.plots.style import (
    color_for,
    label_for,
    marker_for,
    save_figure,
    setup_style,
)

log = logging.getLogger(__name__)


def _dps_per_layer_from_csv(csv_path: Path) -> dict[str, list[float]]:
    """The aggregate CSV has one row per (mode, run) with ``dps_mean`` etc., not
    per-layer scores (per-layer scores are too high-cardinality for a flat CSV).

    This function reads the *per-run* dps.json files instead to recover the
    per-layer curves.  For the rare case where only the aggregate CSV is
    available, we fall back to a single-point bar plot of the mean.
    """
    raise NotImplementedError("Use per-run dps.json files for per-layer curves.")


def plot_dps_curves(
    dps_files: Iterable[Path],
    out_dir: Path,
) -> list[Path]:
    """Plot DPS and GPS per layer for each (mode, run) pair.

    Args:
        dps_files: Iterable of paths to ``<mode>/<run_id>/dps.json`` files.
        out_dir: Output directory; ``out_dir/dps_vs_layer.png`` etc. are written.

    Returns:
        List of written plot paths.
    """
    setup_style()
    import json

    # Group by (mode, run_id) -> list of (layer, score)
    series_dps: dict[str, list[tuple[int, float]]] = {}
    series_gps: dict[str, list[tuple[int, float]]] = {}
    series_label: dict[str, str] = {}
    for path in dps_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", path, exc)
            continue
        mode = payload.get("residual_mode", "?")
        dps_scores = payload.get("dps_scores", []) or []
        gps_scores = payload.get("gps_scores", []) or []
        if not dps_scores:
            continue
        run_id = payload.get("run_id", "?")
        # Use the shortest run_id (just timestamp suffix) in the label.
        short_run = run_id.split("-")[-1] if "-" in run_id else run_id
        label = f"{label_for(mode)} ({short_run})"
        series_label[mode] = label
        # DPS is per layer 1..L-1 (length L-1), so x = 1..L-1
        for i, score in enumerate(dps_scores, start=1):
            series_dps.setdefault(mode, []).append((i, float(score)))
        for i, score in enumerate(gps_scores, start=1):
            series_gps.setdefault(mode, []).append((i, float(score)))

    if not series_dps:
        log.warning("No DPS data to plot.")
        return []

    written: list[Path] = []

    # DPS plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for mode, points in sorted(series_dps.items()):
        points_sorted = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        ax.plot(xs, ys, marker=marker_for(mode), color=color_for(mode),
                label=series_label[mode], linewidth=1.5, markersize=6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("DPS (Depth Preservation Score)")
    ax.set_title("Depth Preservation Score per Layer")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    written.extend(save_figure(fig, out_dir / "dps_vs_layer"))

    # GPS plot (if any)
    if series_gps:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for mode, points in sorted(series_gps.items()):
            points_sorted = sorted(points, key=lambda p: p[0])
            xs = [p[0] for p in points_sorted]
            ys = [p[1] for p in points_sorted]
            ax.plot(xs, ys, marker=marker_for(mode), color=color_for(mode),
                    label=series_label[mode], linewidth=1.5, markersize=6)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("GPS (Gradient Preservation Score)")
        ax.set_title("Gradient Preservation Score per Layer")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
        written.extend(save_figure(fig, out_dir / "gps_vs_layer"))

    return written


def plot_dps_summary(csv_path: Path, out_dir: Path) -> list[Path]:
    """Plot a bar chart of DRI per mode from the aggregate CSV.

    A complement to ``plot_dps_curves``: shows the headline number per mode
    at a glance, useful for a quick read of the data.
    """
    setup_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    if df.empty or "dri" not in df.columns:
        log.warning("CSV %s has no dri column or is empty", csv_path)
        return []
    # If multiple runs per mode, take the mean.
    agg = df.groupby("residual_mode", as_index=False).agg(dri=("dri", "mean"), gpi=("gpi", "mean"))
    agg = agg.sort_values("dri", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = list(range(len(agg)))
    ax.bar(xs, agg["dri"].tolist(), color=[color_for(m) for m in agg["residual_mode"]])
    ax.set_xticks(xs)
    ax.set_xticklabels([label_for(m) for m in agg["residual_mode"]], rotation=15, ha="right")
    ax.set_ylabel("DRI (Dilution Resistance Index)")
    ax.set_title("DRI per Residual Mode (mean across runs)")
    ax.set_ylim(0.0, 1.05)
    return save_figure(fig, out_dir / "dri_per_mode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dps-files",
        nargs="*",
        type=Path,
        default=None,
        help="Per-run dps.json files (preferred for per-layer curves)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Aggregate CSV (used for the bar chart of DRI per mode)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for plots")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="If given, walks <root>/depth_preservation/*/*/dps.json for per-layer curves",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.artifact_root is not None:
        files = sorted(args.artifact_root.glob("depth_preservation/*/*/dps.json"))
        written.extend(plot_dps_curves(files, args.out))
    elif args.dps_files:
        written.extend(plot_dps_curves(args.dps_files, args.out))

    if args.csv is not None:
        written.extend(plot_dps_summary(args.csv, args.out))

    if not written:
        log.error("No plots produced.  Pass --dps-files or --artifact-root or --csv.")
        return 1
    log.info("Wrote %d plots to %s", len(written), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
