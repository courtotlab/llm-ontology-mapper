"""Scenario 2 (retrieval-mode ablation) figure suite.

Consumes ALREADY-COMPLETED public/local/disabled run directories and produces
publication figures + derived data tables under an output directory (default
outputs/evaluation_figures/scenario2/). Makes ZERO mapping/LLM/retrieval/
validator/network calls -- every input is a file already on disk.

Reuses, rather than reimplements:
    - scenario2_compare.load_and_validate_runs / build_paired_predictions /
      transition_counts / build_comparison_table / read_mode_summary_values /
      write_paired_predictions_csv / write_comparison_csv / write_comparison_md
      for dataset/config compatibility checks and cross-mode pairing.
    - scenario2_reliability_plot.plot_reliability_diagram for the reliability
      diagram (called directly, not redesigned).
    - scenario2_metrics.score_prediction/aggregate/abstention_stats/
      execution_diagnostics and scenario2_output.csv_row_to_prediction_record
      for every recomputed metric, so a figure's numbers are always derived
      the same way the benchmark itself derives them (Part 18 reconciliation).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.ticker import PercentFormatter

from llm_ontology_mapper.benchmarking.figures.common import (
    to_float,
    write_csv,
    write_markdown_table,
)
from llm_ontology_mapper.benchmarking.figures.style import (
    MODE_COLORS,
    MODE_DISPLAY,
    MODE_ORDER,
    apply_style,
    bar_positions,
    pct,
    save_figure,
    style_axis,
)
from llm_ontology_mapper.benchmarking.scenario2_compare import (
    LoadedRun,
    build_comparison_table,
    build_paired_predictions,
    load_and_validate_runs,
    read_mode_summary_values,
    transition_counts,
    write_comparison_csv,
    write_comparison_md,
    write_paired_predictions_csv,
)
from llm_ontology_mapper.benchmarking.scenario2_metrics import (
    STATUS_ERROR,
    STATUS_MAPPED,
    abstention_stats,
    aggregate,
    execution_diagnostics,
    is_abstention,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario2_output import csv_row_to_prediction_record
from llm_ontology_mapper.benchmarking.scenario2_reliability_plot import plot_reliability_diagram

MODES = MODE_ORDER  # ("public", "local", "disabled") -- always this order, never sorted by value

_RECONCILE_REL_TOL = 1e-6
_RECONCILE_ABS_TOL = 1e-9

EXPECTED_ONTOLOGY_ORDER: tuple[str, ...] = ("HPO", "MONDO", "LOINC", "ICD10", "SNOMED", "NCIT", "RxNorm")


class ScenarioCompatibilityError(RuntimeError):
    """Raised when a required run is not completed, or is otherwise unsafe
    to plot (beyond what scenario2_compare's own checks already cover)."""


class ReconciliationError(RuntimeError):
    """Raised when a value recomputed from predictions.csv does not match the
    corresponding value already persisted in mode_summary.csv, beyond
    floating-point tolerance. Never silently resolved by picking one value."""


# ─────────────────────────────────────────────────────────────────────────────
# Loading + compatibility (Part 5)
# ─────────────────────────────────────────────────────────────────────────────


def load_completed_runs(*, public_dir: Path, local_dir: Path, disabled_dir: Path) -> dict[str, LoadedRun]:
    """Reuses scenario2_compare.load_and_validate_runs for dataset/config
    compatibility (identical SHA/N/row IDs/source fields/golds/target
    ontologies, and identical provider/model/reasoning_effort/temperature/
    seed/max_alternatives/strict_target_ontology across all three), then adds
    the one check that module does not perform: experiment_config["completed"]
    must be true for every run -- refusing to plot a partial/in-progress run.
    """
    runs = load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    incomplete = [mode for mode, run in runs.items() if run.config.get("completed") is not True]
    if incomplete:
        raise ScenarioCompatibilityError(
            "Refusing to plot: the following run directories are not marked "
            f"completed=true in experiment_config.json: {incomplete}. "
            "Only fully-completed Scenario 2 runs may be used for these figures."
        )
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation (Part 18): recomputed-from-predictions.csv vs mode_summary.csv
# ─────────────────────────────────────────────────────────────────────────────


def _recompute_mode_metrics(run: LoadedRun) -> dict[str, float]:
    csv_rows = list(run.rows_by_id.values())
    records = [csv_row_to_prediction_record(r) for r in csv_rows]
    mapped_codes = [r.get("mapped_code") for r in csv_rows]
    row_metrics = [score_prediction(r) for r in records]
    agg = aggregate(row_metrics)
    abst = abstention_stats(records, mapped_codes)
    exec_diag = execution_diagnostics(records)
    return {
        "n": float(agg.n),
        "top1_accuracy": agg.top1,
        "top3_accuracy": agg.top3,
        "top5_accuracy": agg.top5,
        "mrr": agg.mrr,
        "recall_at_gt": agg.recall_at_gt,
        "abstention_rate": abst.abstention_rate,
        "execution_error_count": float(exec_diag.error_count),
    }


def reconcile_mode(run: LoadedRun, mode_summary: dict[str, str]) -> list[str]:
    """Returns a list of human-readable mismatch messages (empty if clean)."""
    issues: list[str] = []
    recomputed = _recompute_mode_metrics(run)
    for key, computed in recomputed.items():
        reported = to_float(mode_summary.get(key))
        if reported is None:
            issues.append(f"{run.mode}: {key!r} missing/blank in mode_summary.csv (recomputed={computed!r})")
            continue
        if not math.isclose(computed, reported, rel_tol=_RECONCILE_REL_TOL, abs_tol=_RECONCILE_ABS_TOL):
            issues.append(
                f"{run.mode}: {key!r} recomputed from predictions.csv={computed!r} != "
                f"mode_summary.csv={reported!r}"
            )
    return issues


def reconcile_all(runs: dict[str, LoadedRun], mode_summaries: dict[str, dict[str, str]]) -> None:
    all_issues: list[str] = []
    for mode in MODES:
        all_issues.extend(reconcile_mode(runs[mode], mode_summaries[mode]))
    if all_issues:
        raise ReconciliationError(
            "Refusing to plot: values recomputed from predictions.csv do not match "
            "the persisted mode_summary.csv beyond floating-point tolerance:\n  "
            + "\n  ".join(all_issues)
        )


def load_mode_summaries(runs: dict[str, LoadedRun]) -> dict[str, dict[str, str]]:
    return {mode: read_mode_summary_values(runs[mode].output_dir / "mode_summary.csv") for mode in MODES}


# ─────────────────────────────────────────────────────────────────────────────
# S2A -- mapping performance
# ─────────────────────────────────────────────────────────────────────────────

_S2A_HEADLINE_METRICS: tuple[tuple[str, str], ...] = (
    ("top1_accuracy", "Top-1"),
    ("top3_accuracy", "Top-3"),
    ("mrr", "MRR"),
    ("recall_at_gt", "Recall@GT"),
)
_S2A_TABLE_METRICS: tuple[tuple[str, str], ...] = _S2A_HEADLINE_METRICS + (("top5_accuracy", "Top-5"),)


def fig_s2a_mapping_performance(mode_summaries: dict[str, dict[str, str]], output_dir: Path) -> None:
    """Grouped bar: Top-1/Top-3/MRR/Recall@GT across modes. Top-5 is
    deliberately excluded from the chart (near-identical to Top-3 in these
    runs: public 0.6239 vs 0.6239, local 0.7798 vs 0.7798, disabled 0.4908 vs
    0.4908) but is retained in the underlying data table."""
    fig, ax = _new_ax(figsize=(8.0, 5.2))
    n_groups, n_series = len(_S2A_HEADLINE_METRICS), len(MODES)
    centers, offsets, width = bar_positions(n_groups, n_series)
    for i, mode in enumerate(MODES):
        values = [to_float(mode_summaries[mode][key]) or 0.0 for key, _ in _S2A_HEADLINE_METRICS]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=MODE_COLORS[mode], label=MODE_DISPLAY[mode])
        for b, v in zip(bars, values, strict=True):
            ax.annotate(
                pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in _S2A_HEADLINE_METRICS])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Score (N=218 rows/mode)")
    ax.set_title("Mapping performance by retrieval mode")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "figure_01_mapping_performance", "main", output_dir)

    rows = [
        {"mode": mode, **{key: to_float(mode_summaries[mode][key]) for key, _ in _S2A_TABLE_METRICS}}
        for mode in MODES
    ]
    write_csv(rows, ["mode", *[k for k, _ in _S2A_TABLE_METRICS]], output_dir / "data" / "scenario2_mapping_performance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# S2B -- retrieval behavior (abstention / hallucination / grounding)
# ─────────────────────────────────────────────────────────────────────────────

_S2B_METRICS: tuple[tuple[str, str], ...] = (
    ("abstention_rate", "Abstention"),
    ("hallucination_rate", "Hallucination"),
    ("grounding_rate", "Grounding"),
)


def fig_s2b_retrieval_behavior(mode_summaries: dict[str, dict[str, str]], output_dir: Path) -> None:
    """Grouped bar: abstention/hallucination/grounding rates. Validation
    coverage is read (never hardcoded) and shown as a small annotation above
    each mode's hallucination bar rather than a fourth, visually dominant
    bar series -- see figure_captions.md for why."""
    fig, ax = _new_ax(figsize=(8.0, 5.2))
    n_groups, n_series = len(_S2B_METRICS), len(MODES)
    centers, offsets, width = bar_positions(n_groups, n_series)
    hallucination_idx = [key for key, _ in _S2B_METRICS].index("hallucination_rate")
    for i, mode in enumerate(MODES):
        values = [to_float(mode_summaries[mode][key]) or 0.0 for key, _ in _S2B_METRICS]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=MODE_COLORS[mode], label=MODE_DISPLAY[mode])
        for j, (b, v) in enumerate(zip(bars, values, strict=True)):
            ax.annotate(
                pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.3,
            )
            if j == hallucination_idx:
                coverage = to_float(mode_summaries[mode].get("validation_coverage"))
                cov_label = pct(coverage) if coverage is not None else "N/A"
                ax.annotate(
                    f"cov={cov_label}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 15),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.6, color="dimgray",
                )
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in _S2B_METRICS])
    # Headroom above the tallest bar (grounding can reach 100%) so its value
    # annotation never collides with the title -- see figure_captions.md for
    # the full "cov=" / hallucination-denominator explanation.
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Rate")
    ax.set_title("Retrieval behavior by mode")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axis(ax)
    save_figure(fig, "figure_02_retrieval_behavior", "main", output_dir)

    rows = [
        {
            "mode": mode,
            "abstention_rate": to_float(mode_summaries[mode]["abstention_rate"]),
            "hallucination_rate": to_float(mode_summaries[mode]["hallucination_rate"]),
            "validation_coverage": to_float(mode_summaries[mode].get("validation_coverage")),
            "grounding_rate": to_float(mode_summaries[mode]["grounding_rate"]),
        }
        for mode in MODES
    ]
    write_csv(
        rows,
        ["mode", "abstention_rate", "hallucination_rate", "validation_coverage", "grounding_rate"],
        output_dir / "data" / "scenario2_retrieval_behavior.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# S2C -- calibration metrics (3 separate panels, raw values, no normalization)
# ─────────────────────────────────────────────────────────────────────────────


def fig_s2c_calibration_metrics(mode_summaries: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    panels = [
        ("roc_auc", "ROC AUC\n(higher is better)"),
        ("brier_score", "Brier score\n(lower is better)"),
        ("ece", "ECE\n(lower is better)"),
    ]
    for ax, (key, title) in zip(axes, panels, strict=True):
        values = [to_float(mode_summaries[mode][key]) or 0.0 for mode in MODES]
        bars = ax.bar(range(len(MODES)), values, color=[MODE_COLORS[m] for m in MODES], width=0.6)
        for b, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.6,
            )
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([MODE_DISPLAY[m] for m in MODES])
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
        ax.set_title(title)
        style_axis(ax)
    fig.suptitle("Confidence calibration and discrimination by retrieval mode (raw values, not normalized)", y=1.03)
    save_figure(fig, "figure_03_calibration_metrics", "main", output_dir)

    rows = [
        {
            "mode": mode,
            "roc_auc": to_float(mode_summaries[mode]["roc_auc"]),
            "brier_score": to_float(mode_summaries[mode]["brier_score"]),
            "ece": to_float(mode_summaries[mode]["ece"]),
            "cohens_d": to_float(mode_summaries[mode].get("cohens_d")),
        }
        for mode in MODES
    ]
    write_csv(rows, ["mode", "roc_auc", "brier_score", "ece", "cohens_d"], output_dir / "data" / "scenario2_calibration_metrics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# S2D -- reliability diagram (reuses scenario2_reliability_plot verbatim)
# ─────────────────────────────────────────────────────────────────────────────


def fig_s2d_reliability_diagram(runs: dict[str, LoadedRun], output_dir: Path) -> None:
    bins_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        bins_path = runs[mode].output_dir / "calibration_bins.csv"
        with bins_path.open(newline="", encoding="utf-8") as fh:
            bins_by_mode[mode] = list(csv.DictReader(fh))
    target_dir = output_dir / "main"
    target_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        bins_by_mode,
        output_png=target_dir / "figure_04_reliability_diagram.png",
        output_svg=target_dir / "figure_04_reliability_diagram.svg",
        output_pdf=target_dir / "figure_04_reliability_diagram.pdf",
    )


# ─────────────────────────────────────────────────────────────────────────────
# S2E -- Top-1 accuracy by target ontology x mode
# ─────────────────────────────────────────────────────────────────────────────


def build_ontology_top1(runs: dict[str, LoadedRun]) -> tuple[list[dict[str, Any]], list[str]]:
    """Returns (rows, ontology_order). rows: one dict per (mode, ontology)
    with n and top1_accuracy, both derived directly from predictions.csv
    (never hardcoded). ontology_order follows EXPECTED_ONTOLOGY_ORDER
    filtered/extended to whatever is actually present in the data."""
    rows: list[dict[str, Any]] = []
    seen_ontologies: set[str] = set()
    for mode in MODES:
        csv_rows = list(runs[mode].rows_by_id.values())
        by_ontology: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in csv_rows:
            by_ontology[row["target_ontology"]].append(row)
        for ontology, ont_rows in by_ontology.items():
            seen_ontologies.add(ontology)
            records = [csv_row_to_prediction_record(r) for r in ont_rows]
            row_metrics = [score_prediction(r) for r in records]
            n = len(records)
            top1 = sum(1 for m in row_metrics if m.top1_hit) / n if n else 0.0
            rows.append({"mode": mode, "ontology": ontology, "n": n, "top1_accuracy": top1})

    ordered = [o for o in EXPECTED_ONTOLOGY_ORDER if o in seen_ontologies]
    ordered += sorted(seen_ontologies - set(ordered))
    return rows, ordered


def fig_s2e_ontology_heatmap(runs: dict[str, LoadedRun], output_dir: Path) -> None:

    rows, ontology_order = build_ontology_top1(runs)
    by_key = {(r["mode"], r["ontology"]): r for r in rows}
    n_by_ontology = {
        ont: next(iter({by_key[(m, ont)]["n"] for m in MODES if (m, ont) in by_key}), 0)
        for ont in ontology_order
    }
    matrix = [[by_key.get((mode, ont), {}).get("top1_accuracy", math.nan) for ont in ontology_order] for mode in MODES]

    fig, ax = _new_ax(figsize=(9.5, 3.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ontology_order)))
    ax.set_xticklabels([f"{ont}\n(n={n_by_ontology[ont]})" for ont in ontology_order])
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([MODE_DISPLAY[m] for m in MODES])
    for i in range(len(MODES)):
        for j in range(len(ontology_order)):
            v = matrix[i][j]
            if isinstance(v, float) and math.isnan(v):
                continue
            text_color = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9.2, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Top-1 accuracy")
    ax.set_title("Top-1 accuracy by target ontology and retrieval mode")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, "figure_05_ontology_top1_heatmap", "main", output_dir)

    write_csv(rows, ["mode", "ontology", "n", "top1_accuracy"], output_dir / "data" / "ontology_top1_by_mode.csv")


# ─────────────────────────────────────────────────────────────────────────────
# S2F -- paired correctness transitions
# ─────────────────────────────────────────────────────────────────────────────

_PAIRS: tuple[tuple[str, str], ...] = (("public", "local"), ("public", "disabled"), ("local", "disabled"))


def build_full_transition_matrices(paired: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """Full 2x2 paired-correctness matrix for each of the 3 mode pairs,
    derived directly from build_paired_predictions() rows -- reuses the same
    `{mode}_correct` fields transition_counts() reads, just tabulates all 4
    cells (both correct / both wrong included) instead of only the 2
    off-diagonal mismatch cells that function returns."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for a, b in _PAIRS:
        both_correct = sum(1 for r in paired if r[f"{a}_correct"] is True and r[f"{b}_correct"] is True)
        a_correct_b_wrong = sum(1 for r in paired if r[f"{a}_correct"] is True and r[f"{b}_correct"] is False)
        a_wrong_b_correct = sum(1 for r in paired if r[f"{a}_correct"] is False and r[f"{b}_correct"] is True)
        both_wrong = sum(1 for r in paired if r[f"{a}_correct"] is False and r[f"{b}_correct"] is False)
        out[(a, b)] = {
            "both_correct": both_correct,
            "a_correct_b_wrong": a_correct_b_wrong,
            "a_wrong_b_correct": a_wrong_b_correct,
            "both_wrong": both_wrong,
        }
    return out


def fig_s2f_paired_transitions(runs: dict[str, LoadedRun], output_dir: Path) -> list[dict[str, Any]]:

    paired = build_paired_predictions(runs)
    write_paired_predictions_csv(paired, output_dir / "data" / "paired_predictions.csv")

    matrices = build_full_transition_matrices(paired)
    transitions = transition_counts(paired)
    # Reconcile the two off-diagonal cells against the existing library
    # function's own tabulation -- they must agree exactly (Part 18).
    for a, b in _PAIRS:
        m = matrices[(a, b)]
        lib_a_wrong_b_correct = transitions.get(f"correct_in_{b}_wrong_in_{a}")
        lib_a_correct_b_wrong = transitions.get(f"correct_in_{a}_wrong_in_{b}")
        if lib_a_correct_b_wrong is not None and lib_a_correct_b_wrong != m["a_correct_b_wrong"]:
            raise ReconciliationError(
                f"paired transitions ({a} vs {b}): recomputed a_correct_b_wrong={m['a_correct_b_wrong']} "
                f"!= scenario2_compare.transition_counts()={lib_a_correct_b_wrong}"
            )
        if lib_a_wrong_b_correct is not None and lib_a_wrong_b_correct != m["a_wrong_b_correct"]:
            raise ReconciliationError(
                f"paired transitions ({a} vs {b}): recomputed a_wrong_b_correct={m['a_wrong_b_correct']} "
                f"!= scenario2_compare.transition_counts()={lib_a_wrong_b_correct}"
            )

    fig, axes = _new_fig_axes(1, 3, figsize=(13.0, 4.2))
    for ax, (a, b) in zip(axes, _PAIRS, strict=True):
        m = matrices[(a, b)]
        n = m["both_correct"] + m["a_correct_b_wrong"] + m["a_wrong_b_correct"] + m["both_wrong"]
        grid = [[m["both_correct"], m["a_correct_b_wrong"]], [m["a_wrong_b_correct"], m["both_wrong"]]]
        ax.imshow(grid, cmap="Purples", vmin=0, vmax=max(1, n))
        for i in range(2):
            for j in range(2):
                count = grid[i][j]
                share = count / n if n else 0.0
                text_color = "white" if count > n * 0.4 else "black"
                ax.text(j, i, f"{count}\n({share:.0%})", ha="center", va="center", fontsize=9.0, color=text_color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{MODE_DISPLAY[b]} correct", f"{MODE_DISPLAY[b]} wrong"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"{MODE_DISPLAY[a]} correct", f"{MODE_DISPLAY[a]} wrong"])
        ax.set_title(f"{MODE_DISPLAY[a]} vs {MODE_DISPLAY[b]}  (n={n})")
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Paired Top-1 correctness transitions (same 218 rows, all three modes)", y=1.03)
    save_figure(fig, "figure_06_paired_correctness_transitions", "main", output_dir)

    transition_rows = [
        {
            "mode_a": a,
            "mode_b": b,
            "n": matrices[(a, b)]["both_correct"]
            + matrices[(a, b)]["a_correct_b_wrong"]
            + matrices[(a, b)]["a_wrong_b_correct"]
            + matrices[(a, b)]["both_wrong"],
            **matrices[(a, b)],
        }
        for a, b in _PAIRS
    ]
    write_csv(
        transition_rows,
        ["mode_a", "mode_b", "n", "both_correct", "a_correct_b_wrong", "a_wrong_b_correct", "both_wrong"],
        output_dir / "data" / "paired_correctness_transitions.csv",
    )
    return transition_rows


# ─────────────────────────────────────────────────────────────────────────────
# Cross-mode comparison artifacts (Part 16) -- reused verbatim from
# scenario2_compare, just written to the figures output dir.
# ─────────────────────────────────────────────────────────────────────────────


def write_comparison_artifacts(runs: dict[str, LoadedRun], mode_summaries: dict[str, dict[str, str]], output_dir: Path) -> None:
    comparison_rows = build_comparison_table(mode_summaries)
    write_comparison_csv(comparison_rows, output_dir / "data" / "scenario2_comparison.csv")
    paired = build_paired_predictions(runs)
    transitions = transition_counts(paired)
    write_comparison_md(comparison_rows, transitions, output_dir / "data" / "scenario2_comparison.md")


# ─────────────────────────────────────────────────────────────────────────────
# S2G -- outcome / first-gold-rank composition (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_CATEGORIES: tuple[str, ...] = (
    "Execution error",
    "Abstained",
    "Gold rank 1",
    "Gold rank 2-3",
    "Gold rank 4-5",
    "No gold in Top 5",
)


def classify_row_outcome(row: dict[str, str]) -> str:
    """Mutually-exclusive classification, in this priority order (never
    double-counted): execution error -> abstained -> gold-rank bucket ->
    no gold returned."""
    status = row.get("status", "")
    if status == STATUS_ERROR:
        return "Execution error"
    if is_abstention(status=status, mapped_code=row.get("mapped_code")):
        return "Abstained"
    rank_raw = row.get("first_gold_rank")
    if not rank_raw:
        return "No gold in Top 5"
    rank = int(float(rank_raw))
    if rank == 1:
        return "Gold rank 1"
    if rank in (2, 3):
        return "Gold rank 2-3"
    if rank in (4, 5):
        return "Gold rank 4-5"
    return "No gold in Top 5"


def build_outcome_distribution(runs: dict[str, LoadedRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        csv_rows = list(runs[mode].rows_by_id.values())
        total = len(csv_rows)
        counts = {cat: 0 for cat in OUTCOME_CATEGORIES}
        for row in csv_rows:
            counts[classify_row_outcome(row)] += 1
        for cat in OUTCOME_CATEGORIES:
            rows.append({"mode": mode, "outcome": cat, "count": counts[cat], "percent": counts[cat] / total if total else 0.0})
    return rows


_OUTCOME_COLORS: dict[str, str] = {
    "Execution error": "#d62728",
    "Abstained": "#bdbdbd",
    "Gold rank 1": "#08519c",
    "Gold rank 2-3": "#6baed6",
    "Gold rank 4-5": "#c6dbef",
    "No gold in Top 5": "#fdae6b",
}


def fig_s2g_outcome_distribution(runs: dict[str, LoadedRun], output_dir: Path) -> None:
    outcome_rows = build_outcome_distribution(runs)
    fig, ax = _new_ax(figsize=(7.6, 5.4))
    x = range(len(MODES))
    bottoms = [0.0] * len(MODES)
    for cat in OUTCOME_CATEGORIES:
        values = [next(r["percent"] for r in outcome_rows if r["mode"] == m and r["outcome"] == cat) for m in MODES]
        bars = ax.bar(list(x), values, bottom=bottoms, color=_OUTCOME_COLORS[cat], label=cat, width=0.55, edgecolor="white", linewidth=0.6)
        for b, v, bottom in zip(bars, values, bottoms, strict=True):
            if v >= 0.04:
                ax.text(
                    b.get_x() + b.get_width() / 2, bottom + v / 2, f"{v * 100:.0f}%",
                    ha="center", va="center", fontsize=7.6,
                    color="white" if cat in ("Gold rank 1", "Execution error") else "black",
                )
        bottoms = [bo + v for bo, v in zip(bottoms, values, strict=True)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODE_DISPLAY[m] for m in MODES])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Share of 218 rows")
    ax.set_title("Outcome / first-gold-rank composition by mode")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axis(ax)
    save_figure(fig, "supp_figure_01_rank_outcome_distribution", "supplementary", output_dir)

    write_csv(outcome_rows, ["mode", "outcome", "count", "percent"], output_dir / "data" / "scenario2_outcome_distribution.csv")


# ─────────────────────────────────────────────────────────────────────────────
# S2H -- confidence by correctness (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

_UNMAPPED_SENTINEL = "UNKNOWN:UNMAPPED"


def _mapped_non_abstained_rows(csv_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in csv_rows:
        if row.get("status") != STATUS_MAPPED:
            continue
        if is_abstention(status=row["status"], mapped_code=row.get("mapped_code")):
            continue
        if not row.get("confidence"):
            continue
        out.append(row)
    return out


def fig_s2h_confidence_by_correctness(runs: dict[str, LoadedRun], output_dir: Path) -> None:
    import numpy as np

    fig, ax = _new_ax(figsize=(8.4, 5.4))
    n_groups, n_series = len(MODES), 2  # Correct / Incorrect
    centers, offsets, width = bar_positions(n_groups, n_series)
    rng = np.random.default_rng(42)
    box_handles = {}
    for i, mode in enumerate(MODES):
        csv_rows = _mapped_non_abstained_rows(list(runs[mode].rows_by_id.values()))
        correct = [to_float(r["confidence"]) for r in csv_rows if str(r.get("semantic_correctness", "")).strip().lower() == "true"]
        incorrect = [to_float(r["confidence"]) for r in csv_rows if str(r.get("semantic_correctness", "")).strip().lower() == "false"]
        for j, (label, values) in enumerate((("Correct", correct), ("Incorrect", incorrect))):
            xpos = centers[i] + offsets[j]
            values = [v for v in values if v is not None]
            if not values:
                continue
            bp = ax.boxplot(
                [values], positions=[xpos], widths=width * 0.8, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.3),
            )
            bp["boxes"][0].set_facecolor(MODE_COLORS[mode])
            bp["boxes"][0].set_alpha(0.85 if label == "Correct" else 0.35)
            box_handles[label] = bp["boxes"][0]
            jitter = rng.uniform(-width * 0.15, width * 0.15, size=len(values))
            ax.scatter([xpos] * len(values) + jitter, values, s=8, color="black", alpha=0.35, zorder=3)
    ax.set_xticks(centers)
    ax.set_xticklabels([MODE_DISPLAY[m] for m in MODES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mapper confidence")
    ax.set_title("Confidence by Top-1 correctness, per mode\n(mapped, non-abstained rows only; individual points jittered)")
    if box_handles:
        ax.legend(list(box_handles.values()), list(box_handles.keys()), frameon=False, loc="lower left")
    style_axis(ax)
    save_figure(fig, "supp_figure_02_confidence_by_correctness", "supplementary", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# S2I -- latency (supplementary; simplified per Part 14's own escape hatch)
# ─────────────────────────────────────────────────────────────────────────────


def audit_latency_field_completeness(runs: dict[str, LoadedRun]) -> dict[str, dict[str, int]]:
    """Returns {mode: {field: non_blank_count}} for the 4 stage-timing
    fields, so the caller can decide whether a stacked per-stage breakdown is
    representative before building one."""
    fields = ("planner_seconds", "retrieval_seconds", "reranker_seconds", "llm_seconds", "end_to_end_seconds")
    out: dict[str, dict[str, int]] = {}
    for mode in MODES:
        csv_rows = list(runs[mode].rows_by_id.values())
        out[mode] = {f: sum(1 for r in csv_rows if r.get(f)) for f in fields}
    return out


def fig_s2i_latency(runs: dict[str, LoadedRun], output_dir: Path) -> dict[str, dict[str, int]]:
    """end_to_end_seconds is fully populated (218/218) for every mode, so
    that is the only latency quantity compared across all three modes here.
    A stage-level (planner/retrieval/reranker) stacked breakdown, matching
    figure_05_latency_breakdown in the model-comparison figures, is
    deliberately NOT built for all three modes: retrieval_seconds/
    reranker_seconds are inapplicable by design for disabled (no retrieval
    stage), and -- more importantly -- planner_seconds/llm_seconds are
    populated for only a small, non-representative subset of disabled's 218
    rows (see figure_captions.md for the exact count). Public and local do
    have complete stage timing; their breakdown is written to a supplementary
    data file instead of forced into a 3-way chart."""
    completeness = audit_latency_field_completeness(runs)

    fig, ax = _new_ax(figsize=(7.2, 5.2))
    box_data = []
    for mode in MODES:
        csv_rows = list(runs[mode].rows_by_id.values())
        values = [to_float(r["end_to_end_seconds"]) for r in csv_rows if r.get("end_to_end_seconds")]
        box_data.append([v for v in values if v is not None])
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True, widths=0.55, medianprops=dict(color="black", linewidth=1.4))
    for patch, mode in zip(bp["boxes"], MODES, strict=True):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(MODES) + 1))
    ax.set_xticklabels([MODE_DISPLAY[m] for m in MODES])
    ax.set_ylabel("End-to-end seconds per row")
    ax.set_title("End-to-end latency by retrieval mode (n=218/mode)\n(stage-level breakdown omitted -- see caption)")
    style_axis(ax)
    save_figure(fig, "supp_figure_03_latency_breakdown", "supplementary", output_dir)

    def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
        values = [v for v in (to_float(r.get(field)) for r in rows) if v is not None]
        return sum(values) / len(values) if values else None

    stage_rows = []
    for mode in ("public", "local"):
        csv_rows = list(runs[mode].rows_by_id.values())
        stage_rows.append(
            {
                "mode": mode,
                "n": len(csv_rows),
                "mean_planner_seconds": _mean_field(csv_rows, "planner_seconds"),
                "mean_retrieval_seconds": _mean_field(csv_rows, "retrieval_seconds"),
                "mean_reranker_seconds": _mean_field(csv_rows, "reranker_seconds"),
                "mean_llm_seconds": _mean_field(csv_rows, "llm_seconds"),
                "mean_end_to_end_seconds": _mean_field(csv_rows, "end_to_end_seconds"),
            }
        )
    write_csv(
        stage_rows,
        ["mode", "n", "mean_planner_seconds", "mean_retrieval_seconds", "mean_reranker_seconds", "mean_llm_seconds", "mean_end_to_end_seconds"],
        output_dir / "data" / "scenario2_latency_stage_breakdown_public_local.csv",
    )
    return completeness


# ─────────────────────────────────────────────────────────────────────────────
# Cost/summary table (table only -- no dedicated cost figure)
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_TABLE_METRICS: tuple[tuple[str, str], ...] = (
    ("top1_accuracy", "Top-1"),
    ("top3_accuracy", "Top-3"),
    ("top5_accuracy", "Top-5"),
    ("mrr", "MRR"),
    ("recall_at_gt", "Recall@GT"),
    ("abstention_rate", "Abstention"),
    ("hallucination_rate", "Hallucination"),
    ("validation_coverage", "Validation coverage"),
    ("grounding_rate", "Grounding"),
    ("roc_auc", "ROC AUC"),
    ("brier_score", "Brier"),
    ("ece", "ECE"),
    ("execution_error_rate", "Execution error rate"),
    ("mean_end_to_end_seconds", "Mean E2E latency (s)"),
    ("mean_llm_seconds", "Mean LLM latency (s)"),
    ("mean_api_cost_per_row_usd", "Mean cost/row (USD)"),
    ("total_api_cost_usd", "Total cost (USD)"),
)


def write_summary_table(mode_summaries: dict[str, dict[str, str]], output_dir: Path) -> None:
    rows = [{"mode": mode, **{key: to_float(mode_summaries[mode].get(key)) for key, _ in _SUMMARY_TABLE_METRICS}} for mode in MODES]
    write_csv(rows, ["mode", *[k for k, _ in _SUMMARY_TABLE_METRICS]], output_dir / "data" / "scenario2_summary_table.csv")

    headers = ["Metric", *[MODE_DISPLAY[m] for m in MODES]]
    table_rows = []
    for key, label in _SUMMARY_TABLE_METRICS:
        table_rows.append([label, *[to_float(mode_summaries[m].get(key)) for m in MODES]])
    write_markdown_table(
        headers, table_rows, output_dir / "data" / "scenario2_summary_table.md",
        title="Scenario 2 -- retrieval-mode ablation -- summary table",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Captions
# ─────────────────────────────────────────────────────────────────────────────


def write_captions(*, completeness: dict[str, dict[str, int]], output_dir: Path) -> None:
    disabled_planner_n = completeness["disabled"]["planner_seconds"]
    disabled_llm_n = completeness["disabled"]["llm_seconds"]
    captions = f"""# Scenario 2 (retrieval-mode ablation) figure captions

All figures compare three retrieval modes (Public, Local, Disabled) evaluated
on the same 218-row `dict_mapped_all.xlsx` dataset, holding model
(gpt-5.6-luna), reasoning effort (low), temperature, seed (42), and
max_alternatives (4) fixed. The same 218 rows were mapped under all three
modes (a paired design). Mode colors are held constant across every figure:
Public=blue, Local=orange, Disabled=bluish-green (Okabe-Ito palette).

**Metric definitions** (see `src/llm_ontology_mapper/benchmarking/scenario2_metrics.py`,
`scenario2_validation.py`, `scenario2_grounding.py` for the canonical code):
- **Top-1**: exact match between the top-ranked predicted code and any
  acceptable gold code for that row (`semantic_correctness`, locked identical
  to `top1_hit`). An ontology-valid but semantically wrong code is NOT correct.
- **Abstention**: the pipeline declined to map (`status="unmapped"`) or
  returned the `UNKNOWN:UNMAPPED` sentinel. Execution errors are a distinct
  outcome and are never counted as an abstention.
- **Hallucination**: among mapped predictions with a *definitive* ontology
  validation outcome (VALID or INVALID -- excludes UNRESOLVED and
  NOT_APPLICABLE rows), the fraction that are INVALID. Unresolved validator
  outcomes are never counted as hallucinations; see validation coverage.
- **Validation coverage**: (VALID + INVALID) / mapped_count -- how much of
  hallucination's own denominator was actually resolved.
- **Grounding**: among mapped predictions, the fraction whose selected code
  was present among the retrieved candidates. Mechanically 0 for Disabled
  (no retrieval stage exists to ground against) -- this figure reads the
  stored `grounding_rate` rather than assuming that value.
- **ROC AUC / Brier / ECE**: computed on mapped rows only, with exact Top-1
  correctness as the binary outcome and mapper confidence as the score
  (`scenario2_calibration.py`). Raw values shown, never normalized against
  each other -- AUC is higher-is-better, Brier and ECE are lower-is-better.
- **Paired transition**: rows are paired by the same source row (`row_id`)
  across all three modes, since all three modes mapped the identical 218 rows.

**figure_01_mapping_performance.** Top-1, Top-3, MRR, and Recall@GT per mode,
from each run's `mode_summary.csv`. Top-5 is excluded from this chart because
it is numerically indistinguishable from Top-3 in all three modes (public
0.6239 vs 0.6239; local 0.7798 vs 0.7798; disabled 0.4908 vs 0.4908) --
identical to 4 decimal places -- but Top-5 remains in
`data/scenario2_mapping_performance.csv`.

**figure_02_retrieval_behavior.** Abstention, hallucination, and grounding
rates per mode. "cov=" annotations above each hallucination bar report that
mode's validation coverage, read directly from `mode_summary.csv` (not
hardcoded) -- a low coverage would mean the hallucination rate itself rests
on fewer resolved codes than mapped_count suggests.

**figure_03_calibration_metrics.** Three independent panels (ROC AUC, Brier,
ECE), each with its own y-axis scaled to its own values -- never combined into
one "higher is better" chart, since the three metrics have opposite
desirable directions and very different natural ranges.

**figure_04_reliability_diagram.** Reuses
`scenario2_reliability_plot.plot_reliability_diagram()` unmodified: mean
predicted confidence (x) vs. empirical Top-1 accuracy (y) over the same fixed
10 equal-width confidence bins for every mode, plus the y=x perfect-
calibration reference. Bins with zero predictions in a given mode are omitted
from that mode's line, never interpolated or fabricated. Marker size was
deliberately left unscaled by bin count to avoid touching a function whose
current rendering behavior other code (`--compare` in
`run_scenario2_retrieval_ablation.py`) and its own tests already depend on.

**figure_05_ontology_top1_heatmap.** Top-1 accuracy by `target_ontology` and
mode, recomputed directly from each mode's `predictions.csv` (never from a
pre-aggregated summary). Per-ontology N is shown in the column labels and in
`data/ontology_top1_by_mode.csv`. NCIT (n=14), RxNorm (n=13), SNOMED (n=15),
and ICD10 (n=16) are far smaller strata than HPO (n=64), MONDO (n=49), and
LOINC (n=47); percentages in the small strata should not be over-interpreted
as precisely as the larger ones.

**figure_06_paired_correctness_transitions.** Three 2x2 matrices (Public vs
Local, Public vs Disabled, Local vs Disabled), built from
`scenario2_compare.build_paired_predictions()` over the same 218 paired rows.
Each cell is a mutually exclusive outcome (both correct / A correct-B wrong /
A wrong-B correct / both correct... i.e. both wrong), counts and row-total
percentages annotated. The two single-direction mismatch cells in each matrix
are cross-checked against `scenario2_compare.transition_counts()` and must
agree exactly. Correctness is exact Top-1 gold match (`semantic_correctness`),
never ontology-valid-but-wrong.

**supp_figure_01_rank_outcome_distribution.** Every one of the 218 rows per
mode assigned to exactly one mutually exclusive outcome, in priority order:
execution error, then abstained, then gold rank 1 / ranks 2-3 / ranks 4-5 /
no gold in the top 5. An unmapped row is classified as "Abstained", never as
"No gold in Top 5" (that category is reserved for mapped rows whose returned
candidates did not include a gold code).

**supp_figure_02_confidence_by_correctness.** Confidence distributions
(boxplot, outliers as individual jittered points, matplotlib only) split by
exact Top-1 correctness, for mapped, non-abstained rows only. Visually
supports (does not duplicate) the ROC AUC / rank-sum / Cohen's d statistics
already in `mode_summary.csv`.

**supp_figure_03_latency_breakdown.** End-to-end latency (boxplot,
n=218/mode -- the only latency field populated for every row in every mode).
A per-stage (planner/retrieval/reranker/LLM) stacked breakdown across all
three modes, matching `figure_05_latency_breakdown` in the model-comparison
figures, was deliberately NOT built: `retrieval_seconds`/`reranker_seconds`
are inapplicable by design for Disabled (no retrieval stage), and
`planner_seconds`/`llm_seconds` are populated for only {disabled_planner_n}/218
and {disabled_llm_n}/218 rows respectively in the Disabled run -- exactly its
13 execution-error rows, not a representative sample of its 218 mapped/
unmapped/error outcomes. Public and Local do have complete stage timing;
their breakdown is written to
`data/scenario2_latency_stage_breakdown_public_local.csv` rather than forced
into a 3-way chart that would misrepresent Disabled.

**data/scenario2_summary_table.{{csv,md}}.** All headline metrics per mode in
one table (no dedicated cost figure was built -- three numbers do not
warrant a standalone chart).

**data/scenario2_comparison.{{csv,md}} and data/paired_predictions.csv.**
Written via the existing `scenario2_compare` module's own table/CSV builders
(same functions `run_scenario2_retrieval_ablation.py --compare` uses),
unmodified.

No confidence intervals or error bars are shown anywhere in this figure set
(v1 scope). Paired-transition derivations are structured so a paired
bootstrap CI could be added to figure_06 later without re-deriving pairing.
"""
    (output_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _new_ax(*, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _new_fig_axes(nrows: int, ncols: int, *, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    runs: dict[str, LoadedRun]
    mode_summaries: dict[str, dict[str, str]]


def build_all(*, public_dir: Path, local_dir: Path, disabled_dir: Path, output_dir: Path) -> BuildResult:
    """Full Scenario 2 figure suite. Zero mapping/LLM/retrieval/validator/
    network calls -- every input is a file already on disk under
    public_dir/local_dir/disabled_dir."""
    apply_style()
    for subdir in ("main", "supplementary", "data"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    runs = load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    mode_summaries = load_mode_summaries(runs)
    reconcile_all(runs, mode_summaries)

    fig_s2a_mapping_performance(mode_summaries, output_dir)
    fig_s2b_retrieval_behavior(mode_summaries, output_dir)
    fig_s2c_calibration_metrics(mode_summaries, output_dir)
    fig_s2d_reliability_diagram(runs, output_dir)
    fig_s2e_ontology_heatmap(runs, output_dir)
    fig_s2f_paired_transitions(runs, output_dir)
    write_comparison_artifacts(runs, mode_summaries, output_dir)

    fig_s2g_outcome_distribution(runs, output_dir)
    fig_s2h_confidence_by_correctness(runs, output_dir)
    completeness = fig_s2i_latency(runs, output_dir)

    write_summary_table(mode_summaries, output_dir)
    write_captions(completeness=completeness, output_dir=output_dir)

    return BuildResult(output_dir=output_dir, runs=runs, mode_summaries=mode_summaries)


__all__ = [
    "EXPECTED_ONTOLOGY_ORDER",
    "MODES",
    "OUTCOME_CATEGORIES",
    "BuildResult",
    "ReconciliationError",
    "ScenarioCompatibilityError",
    "build_all",
    "build_full_transition_matrices",
    "build_ontology_top1",
    "build_outcome_distribution",
    "classify_row_outcome",
    "load_completed_runs",
    "load_mode_summaries",
    "reconcile_all",
    "reconcile_mode",
]
