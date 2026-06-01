"""Plotting style and helpers shared by all scripts.plots.* scripts.

The convention is:
  * One consistent color per residual mode across ALL plots
  * Sans-serif fonts (DejaVu Sans, available on every system with matplotlib)
  * 4x3 inch figures at 150 DPI (publication-quality)
  * Both PNG and PDF saved side-by-side (PDF for inclusion in papers)
  * Grid on all panels, faint (alpha=0.3)
  * x-axis labels with units in parentheses
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)

import matplotlib.pyplot as plt

# Five residual modes + a few ablation variants.
# Color palette: seaborn "colorblind" (colorblind-friendly).
MODE_COLORS: dict[str, str] = {
    "standard": "#0173b2",            # blue
    "recurrent_residual": "#de8f05",  # orange
    "vega": "#029e73",                # green
    "block_attnres": "#d55e00",       # vermillion
    "full_attnres": "#cc78bc",        # pink-purple
    "hyper_connection": "#ece133",   # yellow
    "mhc": "#ece133",                 # alias
    "mhc_lite": "#fbafe4",            # light pink
    "vega_no_var_reg": "#56b4e9",     # sky blue
    "vega_no_multiscale": "#a8a8a8",  # grey
    "rr_no_depth_biases": "#7b6cd9",  # purple
    # IsoFLOP-style "wide_shallow" / "narrow_deep" prefixes get the base mode's color.
    "wide_shallow_std": "#0173b2",
    "narrow_deep_vega": "#029e73",
    "narrow_deep_rr": "#de8f05",
    "narrow_deep_hc": "#ece133",
}

MODE_MARKERS: dict[str, str] = {
    "standard": "o",
    "recurrent_residual": "s",
    "vega": "^",
    "block_attnres": "D",
    "full_attnres": "P",
    "hyper_connection": "v",
    "mhc": "v",
    "mhc_lite": "<",
    "vega_no_var_reg": "X",
    "vega_no_multiscale": "p",
    "rr_no_depth_biases": "*",
}

# Pretty labels for the legends.
MODE_LABELS: dict[str, str] = {
    "standard": "Standard",
    "recurrent_residual": "RR",
    "vega": "VEGA",
    "block_attnres": "AttnRes (block)",
    "full_attnres": "AttnRes (full)",
    "hyper_connection": "mHC",
    "mhc": "mHC",
    "mhc_lite": "mHC-Lite",
    "vega_no_var_reg": "VEGA -var-reg",
    "vega_no_multiscale": "VEGA -multiscale",
    "rr_no_depth_biases": "RR -depth-bias",
}


def setup_style() -> None:
    """Apply the project-wide matplotlib style.  Call once per script."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def color_for(mode: str) -> str:
    """Return the color for a given mode (with fallback)."""
    return MODE_COLORS.get(mode, "#404040")


def marker_for(mode: str) -> str:
    """Return the marker shape for a given mode (with fallback)."""
    return MODE_MARKERS.get(mode, "o")


def label_for(mode: str) -> str:
    """Return the human-readable label for a given mode (with fallback)."""
    return MODE_LABELS.get(mode, mode)


def save_figure(fig: Any, base_path: Path | str, *, formats: Iterable[str] = ("png", "pdf")) -> list[Path]:
    """Save a matplotlib figure in multiple formats.  Returns the list of paths written.

    Args:
        fig: Matplotlib figure.
        base_path: Path WITHOUT extension.  E.g. ``plots/dps_vs_layer`` produces
            ``plots/dps_vs_layer.png`` and ``plots/dps_vs_layer.pdf``.
        formats: Iterable of file extensions (without dot).

    Returns:
        List of paths actually written.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        path = base_path.with_suffix(f".{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
