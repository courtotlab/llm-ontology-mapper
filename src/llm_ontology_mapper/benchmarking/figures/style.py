"""Shared matplotlib style for the evaluation-figure suites.

Deliberately mirrors the visual language already established by
scripts/plot_model_benchmark_comparison.py (rcParams, spine/grid treatment,
percentage formatting, PNG+SVG+PDF triple-save at 300 dpi) so Scenario 1/2
figures read as one family with the existing model-comparison figures under
outputs/benchmark_figures/. That script is intentionally left untouched --
these are copies of its small style primitives, not imports from it, since it
is a standalone script (not an importable package module). A future cleanup
can deduplicate the two once visual parity is confirmed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 mode identity: fixed order, display names, colorblind-safe colors
# ─────────────────────────────────────────────────────────────────────────────

MODE_ORDER: tuple[str, ...] = ("public", "local", "disabled")

MODE_DISPLAY: dict[str, str] = {
    "public": "Public",
    "local": "Local",
    "disabled": "Disabled",
}

# Okabe-Ito colorblind-safe palette (same family as the model-comparison
# figures), a distinct trio from the 4 model colors so mode-colored and
# model-colored figures are never confused for one another.
MODE_COLORS: dict[str, str] = {
    "public": "#0072B2",  # blue
    "local": "#E69F00",  # orange
    "disabled": "#009E73",  # bluish green
}


def apply_style() -> None:
    """Set the shared rcParams block. Call once before building any figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.axisbelow": True,
        }
    )


def style_axis(ax) -> None:
    ax.tick_params(axis="both", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def bar_positions(n_groups: int, n_series: int, width: float = 0.8):
    group_centers = np.arange(n_groups)
    series_width = width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * series_width
    return group_centers, offsets, series_width


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def save_figure(
    fig, name: str, subdir: str, output_dir: Path, *, formats: tuple[str, ...] = ("svg", "pdf", "png")
) -> None:
    """Save fig (dpi=300, bbox_inches='tight') under output_dir/subdir/name.{ext}
    for each extension in `formats`, then close it. Defaults to the existing
    svg+pdf+png triple used by every pre-existing figure suite; pass
    formats=("png", "svg") for a PDF-free suite (e.g. published_comparison.py)
    without touching the default for callers that still need PDFs."""
    target_dir = output_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(target_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
