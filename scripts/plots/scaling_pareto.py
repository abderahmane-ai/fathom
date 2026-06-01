"""Plot scaling Pareto: validation loss vs parameter count, one curve per mode.

Reads ``results/aggregate/<benchmark>_summary.csv`` and/or
``results/aggregate/<benchmark>_steps.csv`` and produces:
  * plots/scaling_efficiency/pareto.png  + .pdf

X-axis is parameter count (log scale), Y-axis is validation loss (lower is
better).  One marker per (mode, sweep point), color by mode.

Usage:
    python -m scripts.plots.scaling_pareto \\
        --csv results/aggregate/scaling_efficiency_summary.csv \\
        --steps-csv results/aggregate/scaling_efficiency_steps.csv \\
        --out plots/scaling_efficiency
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from scripts.plots.style import (
    color_for,
    label_for,
    marker_for,
    save_figure,
    setup_style,
)

log = logging.getLogger(__name__)


def plot_pareto(
    summary_csv: Path,
    steps_csv: Path | None,
    out_dir: Path,
) -> list[Path]:
    """Plot validation loss vs parameter count per mode.

    Args:
        summary_csv: Path to <benchmark>_summary.csv.
        steps_csv: Optional path to <benchmark>_steps.csv.  If given, the
            final val/loss per (mode, run) is used instead of summary's
            elapsed_seconds (more accurate for "final loss").
        out_dir: Output directory.
    """
    setup_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(summary_csv)
    if df.empty or "parameter_count" not in df.columns or "residual_mode" not in df.columns:
        log.warning("CSV %s missing required columns", summary_csv)
        return []
    # If we have per-step val/loss, extract the final val/loss per run.
    if steps_csv is not None and steps_csv.is_file():
        steps = pd.read_csv(steps_csv)
        if "val_loss" in steps.columns and "run_id" in steps.columns:
            final_val = steps.dropna(subset=["val_loss"]).groupby("run_id", as_index=False).tail(1)
            final_val = final_val[["run_id", "val_loss"]]
            df = df.merge(final_val, on="run_id", how="left")
        else:
            df["val_loss"] = None
    else:
        df["val_loss"] = None

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for mode, group in df.groupby("residual_mode"):
        # Sort by parameter count to draw the Pareto curve.
        group = group.sort_values("parameter_count")
        xs = group["parameter_count"].tolist()
        ys = group["val_loss"].tolist() if "val_loss" in group.columns else None
        if ys and all(y is not None for y in ys):
            ax.plot(xs, ys, marker=marker_for(mode), color=color_for(mode),
                    label=label_for(mode), linewidth=1.5, markersize=7)
        else:
            # Fall back to a single-point marker.
            ax.scatter(xs, [0] * len(xs), marker=marker_for(mode), color=color_for(mode),
                       label=label_for(mode), s=60)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Scaling Pareto: Val Loss vs Params")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    return save_figure(fig, out_dir / "pareto")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, required=True, help="<benchmark>_summary.csv")
    parser.add_argument("--steps-csv", type=Path, default=None, help="<benchmark>_steps.csv (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    written = plot_pareto(args.csv, args.steps_csv, args.out)
    if not written:
        log.error("No plots produced.")
        return 1
    log.info("Wrote %d plots to %s", len(written), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
