"""
Scenario 2 (retrieval-mode ablation) shared reliability diagram (Part 16/28).

Built only after all three (public/local/disabled) calibration_bins.csv files
exist -- never fabricated for a subset of modes. Uses matplotlib directly (no
seaborn), the same fixed 10-bin definition already shared across modes by
scenario2_calibration.expected_calibration_error, and the same axes for every
curve so the three modes are visually comparable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MODE_COLORS: dict[str, str] = {
    "public": "#1f77b4",
    "local": "#ff7f0e",
    "disabled": "#2ca02c",
}
_MODE_LABELS: dict[str, str] = {
    "public": "Public",
    "local": "Local",
    "disabled": "Disabled",
}


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def plot_reliability_diagram(
    bins_by_mode: dict[str, list[dict[str, object]]],
    *,
    output_png: Path,
    output_svg: Path,
    output_pdf: Path,
) -> None:
    """bins_by_mode: {"public": [row, ...], "local": [...], "disabled": [...]}
    where each row is a calibration_bins.csv record (bin_lower, bin_upper,
    mean_confidence, empirical_accuracy, count as strings or numbers).
    Empty/None mean_confidence bins (no predictions fell in that bin) are
    skipped in the line for that mode -- never interpolated or fabricated."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=100)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.5, label="Perfect calibration")

    for mode in ("public", "local", "disabled"):
        rows = bins_by_mode.get(mode) or []
        xs: list[float] = []
        ys: list[float] = []
        for row in sorted(rows, key=lambda r: _to_float(r.get("bin_lower")) or 0.0):
            x = _to_float(row.get("mean_confidence"))
            y = _to_float(row.get("empirical_accuracy"))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=_MODE_COLORS[mode],
            label=_MODE_LABELS[mode],
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical Top-1 accuracy")
    ax.set_title("Scenario 2 -- retrieval-mode calibration reliability")
    ax.legend(loc="upper left", frameon=True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)
    fig.savefig(output_pdf)
    plt.close(fig)
