"""Plot inference memory: peak VRAM and p99 latency vs depth, per mode.

Reads ``results/aggregate/inference_memory.csv`` (produced by
``scripts.ingest.collect``) and produces:
  * plots/inference_memory/vram_vs_depth.png  + .pdf
  * plots/inference_memory/latency_vs_depth.png  + .pdf

Usage:
    python -m scripts.plots.inference_memory \\
        --csv results/aggregate/inference_memory.csv \\
        --out plots/inference_memory
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


def _plot_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> list[Path]:
    """Plot y_col vs x_col per mode, returning list of written paths."""
    setup_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for mode, group in df.groupby("residual_mode"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker=marker_for(mode), color=color_for(mode),
                label=label_for(mode), linewidth=1.5, markersize=7)
    ax.set_xlabel("Number of layers")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    return save_figure(fig, out_path)


def plot_inference(csv_path: Path, out_dir: Path) -> list[Path]:
    """Plot VRAM and latency from the aggregate CSV."""
    df = pd.read_csv(csv_path)
    if df.empty or "layers" not in df.columns or "residual_mode" not in df.columns:
        log.warning("CSV %s missing required columns", csv_path)
        return []
    written: list[Path] = []
    written.extend(_plot_xy(df, "layers", "peak_vram_mb", "Peak VRAM (MiB)",
                            "Peak VRAM vs Depth", out_dir / "vram_vs_depth"))
    written.extend(_plot_xy(df, "layers", "p99_latency_ms", "p99 Latency (ms)",
                            "p99 Latency vs Depth", out_dir / "latency_vs_depth"))
    written.extend(_plot_xy(df, "layers", "tokens_per_second", "Throughput (tok/s)",
                            "Throughput vs Depth", out_dir / "throughput_vs_depth"))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, required=True, help="inference_memory.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    written = plot_inference(args.csv, args.out)
    if not written:
        log.error("No plots produced.")
        return 1
    log.info("Wrote %d plots to %s", len(written), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
