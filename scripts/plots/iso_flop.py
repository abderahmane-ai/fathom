"""Plot iso-FLOP comparison: val loss vs FLOPs, wide-shallow vs narrow-deep.

Reads ``results/aggregate/iso_flop_summary.csv`` and/or
``results/aggregate/iso_flop_steps.csv`` and produces:
  * plots/iso_flop/iso_flop.png  + .pdf

Each residual mode has a "wide_shallow" and "narrow_deep" variant.  The
plot shows val loss vs FLOPs, with each variant as a separate marker.

Usage:
    python -m scripts.plots.iso_flop \\
        --csv results/aggregate/iso_flop_summary.csv \\
        --steps-csv results/aggregate/iso_flop_steps.csv \\
        --out plots/iso_flop
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


def plot_iso_flop(
    summary_csv: Path,
    steps_csv: Path | None,
    out_dir: Path,
) -> list[Path]:
    """Plot val_loss vs FLOPs per (mode, shape) variant."""
    setup_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(summary_csv)
    if df.empty or "residual_mode" not in df.columns:
        log.warning("CSV %s missing required columns", summary_csv)
        return []
    # Estimate FLOPs from parameter count (rough proxy: ~6 * params * tokens).
    # The "iso" qualifier means the proxy is *consistent* across variants,
    # not necessarily accurate in absolute terms.
    if "parameter_count" in df.columns:
        df["estimated_flops"] = df["parameter_count"] * 6  # proxy
    else:
        log.warning("No parameter_count column; cannot estimate FLOPs.")
        return []

    if steps_csv is not None and steps_csv.is_file():
        steps = pd.read_csv(steps_csv)
        if "val_loss" in steps.columns and "run_id" in steps.columns:
            final_val = steps.dropna(subset=["val_loss"]).groupby("run_id", as_index=False).tail(1)
            df = df.merge(final_val[["run_id", "val_loss"]], on="run_id", how="left")
        else:
            df["val_loss"] = None
    else:
        df["val_loss"] = None

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for mode, group in df.groupby("residual_mode"):
        group = group.sort_values("estimated_flops")
        # If the mode has "wide_shallow" or "narrow_deep" in its name, use that as the label.
        short_mode = mode.replace("wide_shallow_", "").replace("narrow_deep_", "")
        if "wide_shallow" in mode:
            label = f"{label_for(short_mode)} (wide-shallow)"
            marker = "o"
        elif "narrow_deep" in mode:
            label = f"{label_for(short_mode)} (narrow-deep)"
            marker = "s"
        else:
            label = label_for(mode)
            marker = marker_for(mode)
        xs = group["estimated_flops"].tolist()
        ys = group["val_loss"].tolist() if "val_loss" in group.columns else None
        if ys and all(y is not None for y in ys):
            ax.scatter(xs, ys, marker=marker, color=color_for(short_mode),
                       label=label, s=80, edgecolor="black", linewidth=0.5)
        else:
            ax.scatter(xs, [0] * len(xs), marker=marker, color=color_for(short_mode),
                       label=label, s=80)

    ax.set_xscale("log")
    ax.set_xlabel("Estimated FLOPs (proxy: 6 × params, log scale)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Iso-FLOP Comparison: Wide-Shallow vs Narrow-Deep")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
    return save_figure(fig, out_dir / "iso_flop")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, required=True, help="iso_flop_summary.csv")
    parser.add_argument("--steps-csv", type=Path, default=None, help="iso_flop_steps.csv (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    written = plot_iso_flop(args.csv, args.steps_csv, args.out)
    if not written:
        log.error("No plots produced.")
        return 1
    log.info("Wrote %d plots to %s", len(written), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
