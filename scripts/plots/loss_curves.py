"""Plot training loss curves (train + val) and gradient norm curves per mode.

Reads ``results/aggregate/<benchmark>_steps.csv`` and produces:
  * plots/<benchmark>/loss_curves.png  + .pdf
  * plots/<benchmark>/grad_norms.png  + .pdf

Usage:
    python -m scripts.plots.loss_curves \\
        --csv results/aggregate/lm_quality_steps.csv \\
        --out plots/lm_quality
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


def plot_loss_curves(steps_csv: Path, out_dir: Path) -> list[Path]:
    """Plot train + val loss per step, one line per (mode, run) for train, one per (mode, run) for val."""
    setup_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(steps_csv)
    if df.empty or "step" not in df.columns or "residual_mode" not in df.columns:
        log.warning("CSV %s missing required columns", steps_csv)
        return []
    written: list[Path] = []

    # Aggregate per (mode, step): mean loss across runs (if multiple).
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, group in df.groupby("residual_mode"):
        train = group.dropna(subset=["train_loss"]).groupby("step", as_index=False)["train_loss"].mean()
        ax.plot(train["step"], train["train_loss"], marker=marker_for(mode), color=color_for(mode),
                label=f"{label_for(mode)} (train)", linewidth=1.0, markersize=3, alpha=0.8)
    for mode, group in df.groupby("residual_mode"):
        val = group.dropna(subset=["val_loss"]).groupby("step", as_index=False)["val_loss"].mean()
        if not val.empty:
            ax.plot(val["step"], val["val_loss"], marker=marker_for(mode), color=color_for(mode),
                    label=f"{label_for(mode)} (val)", linewidth=2.0, markersize=6, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("Training & Validation Loss Curves")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8, ncol=2)
    written.extend(save_figure(fig, out_dir / "loss_curves"))

    # Per-mode subplots: one row per mode.
    modes = sorted(df["residual_mode"].unique())
    if len(modes) > 1:
        fig, axes = plt.subplots(1, len(modes), figsize=(3 * len(modes), 3), sharey=True)
        if len(modes) == 1:
            axes = [axes]
        for ax, mode in zip(axes, modes):
            sub = df[df["residual_mode"] == mode]
            train = sub.dropna(subset=["train_loss"])
            val = sub.dropna(subset=["val_loss"])
            ax.plot(train["step"], train["train_loss"], color=color_for(mode),
                    linewidth=1.0, alpha=0.6, label="train")
            ax.plot(val["step"], val["val_loss"], color=color_for(mode),
                    linewidth=2.0, label="val")
            ax.set_title(label_for(mode))
            ax.set_xlabel("Step")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        axes[0].set_ylabel("Loss (nats)")
        written.extend(save_figure(fig, out_dir / "loss_curves_per_mode"))

    return written


def plot_grad_norms(steps_csv: Path, out_dir: Path) -> list[Path]:
    """Plot grad/global_norm per step, one line per mode."""
    setup_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(steps_csv)
    if df.empty or "step" not in df.columns or "residual_mode" not in df.columns:
        log.warning("CSV %s missing required columns", steps_csv)
        return []
    if "grad_global_norm" not in df.columns:
        log.warning("CSV %s has no grad_global_norm column", steps_csv)
        return []
    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for mode, group in df.groupby("residual_mode"):
        grad = group.dropna(subset=["grad_global_norm"]).groupby("step", as_index=False)["grad_global_norm"].mean()
        ax.plot(grad["step"], grad["grad_global_norm"], marker=marker_for(mode), color=color_for(mode),
                label=label_for(mode), linewidth=1.0, markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient L2 norm")
    ax.set_title("Gradient Norm vs Step")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
    written.extend(save_figure(fig, out_dir / "grad_norms"))

    # Optionally: learning rate curve
    if "learning_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for mode, group in df.groupby("residual_mode"):
            lr = group.dropna(subset=["learning_rate"]).groupby("step", as_index=False)["learning_rate"].mean()
            ax.plot(lr["step"], lr["learning_rate"], marker=marker_for(mode), color=color_for(mode),
                    label=label_for(mode), linewidth=1.0, markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
        written.extend(save_figure(fig, out_dir / "lr_schedule"))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, required=True, help="<benchmark>_steps.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--no-grad-norms", action="store_true", help="Skip the grad norm plot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(plot_loss_curves(args.csv, args.out))
    if not args.no_grad_norms:
        written.extend(plot_grad_norms(args.csv, args.out))
    if not written:
        log.error("No plots produced.")
        return 1
    log.info("Wrote %d plots to %s", len(written), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
