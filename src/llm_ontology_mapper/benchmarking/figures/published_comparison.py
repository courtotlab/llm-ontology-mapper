"""
Scenario 1 published-baseline comparison figure suite.

Compares three ALREADY-COMPLETED Scenario 1 EFO runs (UKBB-EFO,
Biomappings-EFO, OLS-EFO full) against published MetaHarmonizer-paper
baseline values for two other tools:

    ours (this repository)      -> "LLM Ontology Mapper"
    metaharmonizer_om            -> "MetaHarmonizer (OM)"   (OM = OntologyMapper,
                                     the ontology-standardization component of
                                     MetaHarmonizer)
    text2term                    -> "text2term (t2t)"

Analysis/visualization only. Makes ZERO mapping/LLM/retrieval/validator/
network calls -- every input is a file already on disk:
    - our values come from each official run's scenario1_metrics.csv,
      reconciled against predictions.csv using the existing
      scenario1_metrics.score_prediction/aggregate utilities (never a new
      scoring definition);
    - OM/text2term values come from a single structured CSV of published
      baselines (default: outputs/evaluation_figures/scenario1/
      published_comparison/data/published_baselines_used.csv) -- never
      hardcoded a second time inside any plotting function here.

Excluded by design (see FIGURES.md):
    - OLS-EFO (disease): our Scenario 1 experiments never ran that subset,
      so there is no three-method comparison for it.
    - cross-method Precision/Recall/F1 and graph-distance (Same/More
      Specific/.../Unrelated): the published table does not supply per-
      method values for these, so they are never inferred or fabricated.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from llm_ontology_mapper.benchmarking.figures.common import to_float, write_csv
from llm_ontology_mapper.benchmarking.figures.style import (
    apply_style,
    bar_positions,
    pct,
    save_figure,
    style_axis,
)
from llm_ontology_mapper.benchmarking.scenario1_metrics import (
    aggregate,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario1_output import csv_row_to_prediction_record

# Every figure in this suite is PNG + SVG only -- no PDF (strict requirement,
# see scripts/plot_scenario1_published_comparison.py and
# tests/benchmarking/test_scenario1_published_comparison.py).
FORMATS: tuple[str, ...] = ("png", "svg")

# ─────────────────────────────────────────────────────────────────────────────
# Fixed, never-sorted-by-performance orders + stable per-method colors
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_ORDER: tuple[str, ...] = ("UKBB-EFO", "Biomappings-EFO", "OLS-EFO (full)")

METHOD_ORDER: tuple[str, ...] = ("ours", "metaharmonizer_om", "text2term")

METHOD_DISPLAY: dict[str, str] = {
    "ours": "LLM Ontology Mapper",
    "metaharmonizer_om": "MetaHarmonizer (OM)",
    "text2term": "text2term (t2t)",
}

# Short axis-tick-only labels for the outcome-distribution stacked-bar
# figures (10/11/12): with three bars per panel plus an "n=" line
# underneath, the full METHOD_DISPLAY strings collide (e.g. "LLM Ontology
# Mapper" running into "MetaHarmonizer (OM)"). Used ONLY for x-tick labels
# on those figures -- legends, prose, FIGURES.md, and every other figure in
# this suite keep using the full METHOD_DISPLAY names.
METHOD_DISPLAY_SHORT: dict[str, str] = {
    "ours": "Our method",
    "metaharmonizer_om": "OM",
    "text2term": "t2t",
}

# Okabe-Ito colorblind-safe family. Chosen to be distinct from the colors
# already used for model identity (scripts/plot_model_benchmark_comparison.py)
# and retrieval-mode identity (figures/style.py MODE_COLORS) so this figure
# suite is never visually confused with either of those.
METHOD_COLORS: dict[str, str] = {
    "ours": "#000000",  # black -- "our method" accent, reused unchanged in delta figures
    "metaharmonizer_om": "#56B4E9",  # sky blue
    "text2term": "#CC79A7",  # reddish purple
}

TOPK_METRICS: tuple[str, ...] = ("Top-1", "Top-3", "Top-5")
ALL_METRICS: tuple[str, ...] = (*TOPK_METRICS, "MRR")

_BASELINE_TOOL_TO_METHOD: dict[str, str] = {
    "metaharmonizer_ontology_mapper": "metaharmonizer_om",
    "text2term": "text2term",
}

# Known to exist in the supplied published table but deliberately never
# plotted here (Part 1/17): our Scenario 1 experiments never ran the
# separate OLS disease subset, so there is no matching three-method
# comparison for it. Rows for this benchmark are silently skipped rather
# than hard-failing as an "unknown benchmark" (Part 24).
KNOWN_EXCLUDED_BENCHMARKS: frozenset[str] = frozenset({"OLS-EFO (disease)"})

RECONCILE_TOLERANCE = 1e-6


class PublishedComparisonError(RuntimeError):
    """Base class for every hard-fail condition in this module (Part 24:
    never silently continue with a partial/incorrect figure set)."""


class RunLoadError(PublishedComparisonError):
    pass


class BaselineTableError(PublishedComparisonError):
    pass


class ReconciliationError(PublishedComparisonError):
    pass


@dataclass(frozen=True)
class MethodMetrics:
    n: int
    top1: float
    top3: float
    top5: float
    mrr: float

    def value(self, metric: str) -> float:
        return {"Top-1": self.top1, "Top-3": self.top3, "Top-5": self.top5, "MRR": self.mrr}[metric]


@dataclass(frozen=True)
class OurRun:
    benchmark: str
    run_dir: Path
    metrics: MethodMetrics
    model: str
    retrieval_mode: str
    target_ontology: str
    strict_target_ontology: bool
    experiment_name: str


# The three official completed runs this figure suite is locked to (Part 4).
# Identity is checked against experiment_config.json's source_dataset_path
# rather than experiment_name: the shared scenario1 runner script stamps the
# same literal experiment_name ("scenario1_ols_efo") into every EFO variant's
# config regardless of which dataset was actually run, so source_dataset_path
# is the field that actually varies and is trustworthy here.
OFFICIAL_RUN_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "UKBB-EFO": {"expected_n": 888, "source_dataset_substring": "UKBB-EFO"},
    "Biomappings-EFO": {"expected_n": 795, "source_dataset_substring": "Biomappings-EFO"},
    "OLS-EFO (full)": {"expected_n": 7377, "source_dataset_substring": "OLS-EFO"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Our-model metrics: read from scenario1_metrics.csv, reconciled against
# predictions.csv (Part 5) -- never typed into this module by hand.
# ─────────────────────────────────────────────────────────────────────────────


def _read_metric_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise RunLoadError(f"missing scenario1_metrics.csv: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["metric"]: row for row in csv.DictReader(fh)}


def load_official_run(benchmark: str, run_dir: Path) -> OurRun:
    if benchmark not in OFFICIAL_RUN_EXPECTATIONS:
        raise RunLoadError(f"unknown benchmark {benchmark!r}")
    expectation = OFFICIAL_RUN_EXPECTATIONS[benchmark]

    config_path = run_dir / "experiment_config.json"
    if not config_path.exists():
        raise RunLoadError(f"missing experiment_config.json under {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("completed") is not True:
        raise RunLoadError(
            f"{run_dir}: experiment_config.json completed != true -- refusing to use an "
            "incomplete/partial run"
        )

    experiment_name = str(config.get("experiment_name", ""))
    source_dataset_path = str(config.get("source_dataset_path", ""))
    if expectation["source_dataset_substring"] not in source_dataset_path:
        raise RunLoadError(
            f"{run_dir}: source_dataset_path {source_dataset_path!r} does not contain expected "
            f"substring {expectation['source_dataset_substring']!r} for benchmark {benchmark!r}"
        )

    metrics_path = run_dir / "scenario1_metrics.csv"
    table = _read_metric_table(metrics_path)
    for metric in ALL_METRICS:
        if metric not in table:
            raise RunLoadError(f"{metrics_path}: missing required metric {metric!r}")
        if table[metric]["status"] != "OK":
            raise RunLoadError(f"{metrics_path}: metric {metric!r} status != OK ({table[metric]['status']!r})")

    denominators = {int(table[m]["denominator"]) for m in ALL_METRICS}
    if len(denominators) != 1:
        raise RunLoadError(f"{metrics_path}: inconsistent denominators across Top-1/3/5/MRR: {denominators}")
    n = denominators.pop()

    expected_n = int(expectation["expected_n"])
    if n != expected_n:
        raise RunLoadError(
            f"{metrics_path}: denominator N={n} does not match expected official N={expected_n} for "
            f"{benchmark!r} -- refusing to use a stale/partial run"
        )

    config_n = config.get("rows_completed")
    if config_n is not None and int(config_n) != expected_n:
        raise RunLoadError(
            f"{run_dir}: experiment_config.json rows_completed={config_n}, expected {expected_n} for {benchmark!r}"
        )

    metrics = MethodMetrics(
        n=n,
        top1=float(table["Top-1"]["value"]),
        top3=float(table["Top-3"]["value"]),
        top5=float(table["Top-5"]["value"]),
        mrr=float(table["MRR"]["value"]),
    )
    return OurRun(
        benchmark=benchmark,
        run_dir=run_dir,
        metrics=metrics,
        model=str(config.get("model", "")),
        retrieval_mode=str(config.get("retrieval_mode", "")),
        target_ontology=str(config.get("target_ontology", "")),
        strict_target_ontology=bool(config.get("strict_target_ontology", False)),
        experiment_name=experiment_name,
    )


def reconcile_official_run(run: OurRun) -> None:
    """Recompute Top-1/3/5/MRR straight from predictions.csv with the
    canonical Scenario 1 scoring utilities (scenario1_metrics.score_prediction
    / aggregate, unmodified) and hard-fail if they disagree with the saved
    scenario1_metrics.csv values beyond floating-point tolerance."""
    predictions_path = run.run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise RunLoadError(f"missing predictions.csv: {predictions_path}")
    with predictions_path.open(newline="", encoding="utf-8") as fh:
        records = [csv_row_to_prediction_record(row) for row in csv.DictReader(fh)]
    recomputed = aggregate([score_prediction(r) for r in records])

    issues = []
    for label, saved, new in (
        ("n", run.metrics.n, recomputed.n),
        ("Top-1", run.metrics.top1, recomputed.top1),
        ("Top-3", run.metrics.top3, recomputed.top3),
        ("Top-5", run.metrics.top5, recomputed.top5),
        ("MRR", run.metrics.mrr, recomputed.mrr),
    ):
        if abs(float(saved) - float(new)) > RECONCILE_TOLERANCE:
            issues.append(f"{label}: saved={saved} recomputed_from_predictions_csv={new}")
    if issues:
        raise ReconciliationError(
            f"{run.run_dir}: scenario1_metrics.csv disagrees with predictions.csv beyond "
            "floating-point tolerance -- refusing to plot:\n  " + "\n  ".join(issues)
        )


def load_all_official_runs(*, ols_dir: Path, ukbb_dir: Path, biomappings_dir: Path) -> dict[str, OurRun]:
    runs = {
        "UKBB-EFO": load_official_run("UKBB-EFO", ukbb_dir),
        "Biomappings-EFO": load_official_run("Biomappings-EFO", biomappings_dir),
        "OLS-EFO (full)": load_official_run("OLS-EFO (full)", ols_dir),
    }
    for run in runs.values():
        reconcile_official_run(run)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Published baselines: ONE structured source of truth (Part 3) -- never
# hardcoded a second time in any plotting function below.
# ─────────────────────────────────────────────────────────────────────────────


def load_baselines(path: Path) -> dict[tuple[str, str], MethodMetrics]:
    if not path.exists():
        raise BaselineTableError(f"missing published baselines CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    raw: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        benchmark, tool, metric = row["benchmark"], row["tool"], row["metric"]
        if benchmark in KNOWN_EXCLUDED_BENCHMARKS:
            continue
        if benchmark not in BENCHMARK_ORDER:
            raise BaselineTableError(f"unknown benchmark {benchmark!r} in {path}")
        if tool not in _BASELINE_TOOL_TO_METHOD:
            raise BaselineTableError(f"unknown baseline tool {tool!r} in {path}")
        if metric not in ALL_METRICS:
            raise BaselineTableError(f"unknown baseline metric {metric!r} in {path}")
        key = (benchmark, tool, metric)
        if key in raw:
            raise BaselineTableError(f"duplicate baseline row for benchmark/tool/metric={key} in {path}")
        raw[key] = row

    result: dict[tuple[str, str], MethodMetrics] = {}
    for benchmark in BENCHMARK_ORDER:
        for tool, method in _BASELINE_TOOL_TO_METHOD.items():
            values: dict[str, float] = {}
            denominators: set[int] = set()
            for metric in ALL_METRICS:
                key = (benchmark, tool, metric)
                if key not in raw:
                    raise BaselineTableError(
                        f"missing baseline row for benchmark={benchmark!r} tool={tool!r} metric={metric!r} in {path}"
                    )
                row = raw[key]
                value = to_float(row["value"])
                if value is None or not (0.0 <= value <= 1.0):
                    raise BaselineTableError(f"malformed baseline value for {key} in {path}: {row['value']!r}")
                values[metric] = value
                denominators.add(int(row["denominator"]))
            if len(denominators) != 1:
                raise BaselineTableError(
                    f"inconsistent denominators for benchmark={benchmark!r} tool={tool!r} in {path}: {denominators}"
                )
            result[(benchmark, method)] = MethodMetrics(
                n=denominators.pop(),
                top1=values["Top-1"],
                top3=values["Top-3"],
                top5=values["Top-5"],
                mrr=values["MRR"],
            )
    return result


def write_baselines_snapshot(source_path: Path, dest_path: Path) -> None:
    """Copy the exact baseline CSV that was used into the output data/ dir
    (Part 23), preserving its own schema/fieldnames verbatim -- a provenance
    snapshot, not a re-derivation."""
    with source_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Combine ours + baselines into one all-methods structure
# ─────────────────────────────────────────────────────────────────────────────


def build_all_methods(
    our_runs: dict[str, OurRun], baselines: dict[tuple[str, str], MethodMetrics]
) -> dict[str, dict[str, MethodMetrics]]:
    combined: dict[str, dict[str, MethodMetrics]] = {}
    for benchmark in BENCHMARK_ORDER:
        combined[benchmark] = {
            "ours": our_runs[benchmark].metrics,
            "metaharmonizer_om": baselines[(benchmark, "metaharmonizer_om")],
            "text2term": baselines[(benchmark, "text2term")],
        }
    return combined


def compute_deltas(
    all_methods: dict[str, dict[str, MethodMetrics]], baseline_method: str
) -> list[dict[str, Any]]:
    """Δ = ours - baseline_method. Top-k in percentage points; MRR as a raw
    (unit-fraction) difference, never called a percentage-point value."""
    rows = []
    for benchmark in BENCHMARK_ORDER:
        ours = all_methods[benchmark]["ours"]
        base = all_methods[benchmark][baseline_method]
        rows.append(
            {
                "benchmark": benchmark,
                "delta_top1_pp": (ours.top1 - base.top1) * 100.0,
                "delta_top3_pp": (ours.top3 - base.top3) * 100.0,
                "delta_top5_pp": (ours.top5 - base.top5) * 100.0,
                "delta_mrr": ours.mrr - base.mrr,
                "ours_n": ours.n,
                "baseline_n": base.n,
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method ranked-outcome (first-gold-rank) distribution.
#
# The published MetaHarmonizer comparison and our own runs share only one
# outcome vocabulary in common: cumulative Top-1/Top-3/Top-5. MetaHarmonizer
# Figure 3 does not report a first-gold-rank composition, and the original
# text2term paper's Same/More-Specific/.../Unrelated Top-1 graph-relationship
# distribution is a DIFFERENT quantity (Top-1 graph relationship, not
# first-gold rank) from a DIFFERENT evaluation (the original text2term paper,
# not the MetaHarmonizer-controlled rerun used as our t2t baseline elsewhere
# in this suite) -- so it is never mixed in here (see FIGURES.md "Online
# source note").
#
# Four mutually-exclusive bins are reconstructed identically for every
# method (ours, OM, t2t) from cumulative Top-k alone:
#     Gold rank 1        = Top-1
#     Gold rank 2-3       = Top-3 - Top-1
#     Gold rank 4-5       = Top-5 - Top-3
#     No gold in Top 5   = 1 - Top-5
# This is a deliberate fairness collapse: our own predictions.csv carries
# richer status detail (execution error / abstained / mapped-but-missed),
# but OM/t2t's published aggregate does not expose those separately, so for
# THESE cross-method figures only, all of our non-Top-5 outcomes are folded
# into "No gold in Top 5" too. Our existing internal Scenario 1/2 figures may
# keep showing the richer breakdown -- this collapse applies only here.
# ─────────────────────────────────────────────────────────────────────────────

# Identical category names/colors to scenario2.OUTCOME_CATEGORIES /
# scenario2._OUTCOME_COLORS's four rank-based entries (Execution error and
# Abstained are dropped -- Part 11: not available cross-method), so this
# suite's stacked bars read as visually related to the Scenario 2
# outcome-distribution supplementary figure. Kept as a local literal rather
# than importing scenario2's private module attribute across files.
CROSS_METHOD_OUTCOME_CATEGORIES: tuple[str, ...] = (
    "Gold rank 1",
    "Gold rank 2-3",
    "Gold rank 4-5",
    "No gold in Top 5",
)

CROSS_METHOD_OUTCOME_COLORS: dict[str, str] = {
    "Gold rank 1": "#08519c",
    "Gold rank 2-3": "#6baed6",
    "Gold rank 4-5": "#c6dbef",
    "No gold in Top 5": "#fdae6b",
}

# Published Top-k baselines are rounded to one decimal percentage point
# (±0.0005 per metric); summed across three differenced metrics that is
# comfortably covered by a 1 percentage-point tolerance, which also absorbs
# ordinary floating-point noise in our own full-precision metrics.
OUTCOME_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class OutcomeDistribution:
    gold_rank_1: float
    gold_rank_2_3: float
    gold_rank_4_5: float
    no_gold_top5: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Gold rank 1": self.gold_rank_1,
            "Gold rank 2-3": self.gold_rank_2_3,
            "Gold rank 4-5": self.gold_rank_4_5,
            "No gold in Top 5": self.no_gold_top5,
        }


def derive_outcome_distribution(m: MethodMetrics) -> OutcomeDistribution:
    """Reconstruct the four mutually-exclusive rank bins from cumulative
    Top-1/Top-3/Top-5 alone (Part 1/4) -- never derived any other way, and
    never separately hardcoded per method. Hard-fails if the implied bins
    are inconsistent with monotonic cumulative accuracy or do not sum to
    ~1.0 within published-rounding tolerance (Part 5)."""
    rank1 = m.top1
    rank2_3 = m.top3 - m.top1
    rank4_5 = m.top5 - m.top3
    no_gold = 1.0 - m.top5

    for label, value in (("Gold rank 2-3", rank2_3), ("Gold rank 4-5", rank4_5), ("No gold in Top 5", no_gold)):
        if value < -1e-6:
            raise PublishedComparisonError(
                f"non-monotonic Top-k values produced a negative {label!r} bin ({value:.6f}) for "
                f"n={m.n} top1={m.top1} top3={m.top3} top5={m.top5} -- refusing to plot"
            )
    rank2_3 = max(rank2_3, 0.0)
    rank4_5 = max(rank4_5, 0.0)
    no_gold = max(no_gold, 0.0)

    total = rank1 + rank2_3 + rank4_5 + no_gold
    if abs(total - 1.0) > OUTCOME_SUM_TOLERANCE:
        raise PublishedComparisonError(
            f"outcome distribution sums to {total:.6f}, not ~1.0, for n={m.n} top1={m.top1} "
            f"top3={m.top3} top5={m.top5} -- refusing to plot"
        )
    return OutcomeDistribution(gold_rank_1=rank1, gold_rank_2_3=rank2_3, gold_rank_4_5=rank4_5, no_gold_top5=no_gold)


def build_outcome_distributions(
    all_methods: dict[str, dict[str, MethodMetrics]],
) -> dict[str, dict[str, OutcomeDistribution]]:
    return {
        benchmark: {method: derive_outcome_distribution(all_methods[benchmark][method]) for method in METHOD_ORDER}
        for benchmark in BENCHMARK_ORDER
    }


def write_outcome_distribution_csv(
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    path: Path,
) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        for method in METHOD_ORDER:
            m = all_methods[benchmark][method]
            d = distributions[benchmark][method]
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": METHOD_DISPLAY[method],
                    "n": m.n,
                    "top1": m.top1,
                    "top3": m.top3,
                    "top5": m.top5,
                    "gold_rank_1": d.gold_rank_1,
                    "gold_rank_2_3": d.gold_rank_2_3,
                    "gold_rank_4_5": d.gold_rank_4_5,
                    "no_gold_top5": d.no_gold_top5,
                }
            )
    write_csv(
        rows,
        ["benchmark", "method", "n", "top1", "top3", "top5", "gold_rank_1", "gold_rank_2_3", "gold_rank_4_5",
         "no_gold_top5"],
        path,
    )


def write_outcome_distribution_md(
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    path: Path,
) -> None:
    lines = [
        "# Scenario 1 -- cross-method ranked-outcome distribution (percentages)",
        "",
        "Four mutually-exclusive bins reconstructed from cumulative Top-1/Top-3/Top-5 "
        "(Gold rank 1 = Top-1; Gold rank 2-3 = Top-3 - Top-1; Gold rank 4-5 = Top-5 - Top-3; "
        "No gold in Top 5 = 1 - Top-5). See FIGURES.md 'Outcome-distribution derivation' for "
        "the full explanation and caveats.",
        "",
        "| Benchmark | Method | n | Gold rank 1 | Gold rank 2-3 | Gold rank 4-5 | No gold in Top 5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark in BENCHMARK_ORDER:
        for method in METHOD_ORDER:
            m = all_methods[benchmark][method]
            d = distributions[benchmark][method]
            lines.append(
                f"| {benchmark} | {METHOD_DISPLAY[method]} | {m.n} | {_fmt_pct(d.gold_rank_1)} | "
                f"{_fmt_pct(d.gold_rank_2_3)} | {_fmt_pct(d.gold_rank_4_5)} | {_fmt_pct(d.no_gold_top5)} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Data table writers (Part 23)
# ─────────────────────────────────────────────────────────────────────────────


def write_our_scenario1_metrics_used_csv(our_runs: dict[str, OurRun], path: Path) -> None:
    rows = [
        {
            "benchmark": benchmark,
            "n": run.metrics.n,
            "top1": run.metrics.top1,
            "top3": run.metrics.top3,
            "top5": run.metrics.top5,
            "mrr": run.metrics.mrr,
            "run_dir": str(run.run_dir),
            "model": run.model,
            "retrieval_mode": run.retrieval_mode,
            "target_ontology": run.target_ontology,
            "strict_target_ontology": run.strict_target_ontology,
        }
        for benchmark, run in ((b, our_runs[b]) for b in BENCHMARK_ORDER)
    ]
    write_csv(
        rows,
        ["benchmark", "n", "top1", "top3", "top5", "mrr", "run_dir", "model", "retrieval_mode",
         "target_ontology", "strict_target_ontology"],
        path,
    )


def write_all_methods_topk_csv(all_methods: dict[str, dict[str, MethodMetrics]], path: Path) -> None:
    rows = [
        {
            "benchmark": benchmark,
            "method": METHOD_DISPLAY[method],
            "n": all_methods[benchmark][method].n,
            "top1": all_methods[benchmark][method].top1,
            "top3": all_methods[benchmark][method].top3,
            "top5": all_methods[benchmark][method].top5,
        }
        for benchmark in BENCHMARK_ORDER
        for method in METHOD_ORDER
    ]
    write_csv(rows, ["benchmark", "method", "n", "top1", "top3", "top5"], path)


def write_all_methods_comparison_csv(all_methods: dict[str, dict[str, MethodMetrics]], path: Path) -> None:
    rows = [
        {
            "benchmark": benchmark,
            "method": METHOD_DISPLAY[method],
            "n": all_methods[benchmark][method].n,
            "top1": all_methods[benchmark][method].top1,
            "top3": all_methods[benchmark][method].top3,
            "top5": all_methods[benchmark][method].top5,
            "mrr": all_methods[benchmark][method].mrr,
        }
        for benchmark in BENCHMARK_ORDER
        for method in METHOD_ORDER
    ]
    write_csv(rows, ["benchmark", "method", "n", "top1", "top3", "top5", "mrr"], path)


def write_pairwise_csv(
    all_methods: dict[str, dict[str, MethodMetrics]], baseline_method: str, path: Path
) -> None:
    rows = [
        {
            "benchmark": benchmark,
            "method": METHOD_DISPLAY[method],
            "n": all_methods[benchmark][method].n,
            "top1": all_methods[benchmark][method].top1,
            "top3": all_methods[benchmark][method].top3,
            "top5": all_methods[benchmark][method].top5,
            "mrr": all_methods[benchmark][method].mrr,
        }
        for benchmark in BENCHMARK_ORDER
        for method in ("ours", baseline_method)
    ]
    write_csv(rows, ["benchmark", "method", "n", "top1", "top3", "top5", "mrr"], path)


def write_delta_csv(rows: list[dict[str, Any]], path: Path) -> None:
    write_csv(rows, ["benchmark", "delta_top1_pp", "delta_top3_pp", "delta_top5_pp", "delta_mrr",
                      "ours_n", "baseline_n"], path)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────


def _new_axes(*, ncols: int, figsize: tuple[float, float], sharey: bool = False):
    apply_style()
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=sharey)
    return fig, axes


def _n_label(all_methods: dict[str, dict[str, MethodMetrics]], benchmark: str) -> str:
    ours_n = all_methods[benchmark]["ours"].n
    om_n = all_methods[benchmark]["metaharmonizer_om"].n
    t2t_n = all_methods[benchmark]["text2term"].n
    if ours_n == om_n == t2t_n:
        return f"n={ours_n}"
    return f"n ours/OM={ours_n}, n t2t={t2t_n}"


def fig_01_all_methods_topk(all_methods: dict[str, dict[str, MethodMetrics]], output_dir: Path) -> None:
    fig, axes = _new_axes(ncols=3, figsize=(14.5, 5.2), sharey=True)
    for ax, benchmark in zip(axes, BENCHMARK_ORDER, strict=True):
        centers, offsets, width = bar_positions(len(TOPK_METRICS), len(METHOD_ORDER))
        for i, method in enumerate(METHOD_ORDER):
            m = all_methods[benchmark][method]
            values = [m.top1, m.top3, m.top5]
            xpos = centers + offsets[i]
            bars = ax.bar(xpos, values, width=width * 0.92, color=METHOD_COLORS[method], label=METHOD_DISPLAY[method])
            for b, v in zip(bars, values, strict=True):
                ax.annotate(
                    pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.6,
                )
        ax.set_xticks(centers)
        ax.set_xticklabels(TOPK_METRICS)
        ax.set_title(f"{benchmark}\n({_n_label(all_methods, benchmark)})", fontsize=10.8)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Scenario 1 -- Top-k accuracy vs. published baselines")
    save_figure(fig, "figure_01_all_methods_topk", "main", output_dir, formats=FORMATS)


def fig_02_all_methods_mrr(all_methods: dict[str, dict[str, MethodMetrics]], output_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    centers, offsets, width = bar_positions(len(BENCHMARK_ORDER), len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        values = [all_methods[b][method].mrr for b in BENCHMARK_ORDER]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=METHOD_COLORS[method], label=METHOD_DISPLAY[method])
        for b, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.0,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(BENCHMARK_ORDER)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MRR (unit fraction -- not a percentage)")
    ax.set_title("Scenario 1 -- Mean Reciprocal Rank vs. published baselines")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axis(ax)
    save_figure(fig, "figure_02_all_methods_mrr", "main", output_dir, formats=FORMATS)


def _pairwise_topk_figure(
    all_methods: dict[str, dict[str, MethodMetrics]], baseline_method: str, basename: str, output_dir: Path,
    title: str,
) -> None:
    fig, axes = _new_axes(ncols=3, figsize=(13.0, 5.2), sharey=True)
    methods = ("ours", baseline_method)
    for ax, benchmark in zip(axes, BENCHMARK_ORDER, strict=True):
        centers, offsets, width = bar_positions(len(TOPK_METRICS), len(methods))
        for i, method in enumerate(methods):
            m = all_methods[benchmark][method]
            values = [m.top1, m.top3, m.top5]
            xpos = centers + offsets[i]
            bars = ax.bar(xpos, values, width=width * 0.85, color=METHOD_COLORS[method], label=METHOD_DISPLAY[method])
            for b, v in zip(bars, values, strict=True):
                ax.annotate(
                    pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.0,
                )
        ax.set_xticks(centers)
        ax.set_xticklabels(TOPK_METRICS)
        ours_n = all_methods[benchmark]["ours"].n
        base_n = all_methods[benchmark][baseline_method].n
        subtitle = f"n={ours_n}" if ours_n == base_n else f"n ours={ours_n}, n {METHOD_DISPLAY[baseline_method]}={base_n}"
        ax.set_title(f"{benchmark}\n({subtitle})", fontsize=10.5)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title)
    save_figure(fig, basename, "pairwise", output_dir, formats=FORMATS)


def _pairwise_mrr_figure(
    all_methods: dict[str, dict[str, MethodMetrics]], baseline_method: str, basename: str, output_dir: Path,
    title: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    methods = ("ours", baseline_method)
    centers, offsets, width = bar_positions(len(BENCHMARK_ORDER), len(methods))
    for i, method in enumerate(methods):
        values = [all_methods[b][method].mrr for b in BENCHMARK_ORDER]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.85, color=METHOD_COLORS[method], label=METHOD_DISPLAY[method])
        for b, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(BENCHMARK_ORDER)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MRR (unit fraction -- not a percentage)")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axis(ax)
    save_figure(fig, basename, "pairwise", output_dir, formats=FORMATS)


def _delta_figure(
    delta_rows: list[dict[str, Any]], basename: str, output_dir: Path, title: str,
) -> None:
    """Zero-centered horizontal bar chart of Top-1/3/5 percentage-point
    deltas (ours - baseline). MRR deltas are deliberately NOT plotted here
    (they are a raw unit-fraction difference, not percentage points) --
    they live in the accompanying delta_vs_*.csv and FIGURES.md instead."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    labels: list[str] = []
    values: list[float] = []
    for row in delta_rows:
        for key, metric_label in (("delta_top1_pp", "Top-1"), ("delta_top3_pp", "Top-3"), ("delta_top5_pp", "Top-5")):
            labels.append(f"{row['benchmark']} — {metric_label}")
            values.append(float(row[key]))
    y_positions = list(range(len(labels)))[::-1]
    ax.barh(y_positions, values, color=METHOD_COLORS["ours"], height=0.6)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    max_abs = max(abs(v) for v in values) if values else 1.0
    pad = max(max_abs * 0.35, 1.0)
    ax.set_xlim(-(max_abs + pad), max_abs + pad)
    for yi, v in zip(y_positions, values, strict=True):
        ax.annotate(
            f"{v:+.1f} pp", (v, yi), xytext=(6 if v >= 0 else -6, 0),
            textcoords="offset points", ha="left" if v >= 0 else "right", va="center", fontsize=8.3,
        )
    ax.set_xlabel("Percentage points (LLM Ontology Mapper − baseline)")
    ax.set_title(title)
    style_axis(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    save_figure(fig, basename, "pairwise", output_dir, formats=FORMATS)


def fig_09_top1_summary(all_methods: dict[str, dict[str, MethodMetrics]], output_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    centers, offsets, width = bar_positions(len(BENCHMARK_ORDER), len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        values = [all_methods[b][method].top1 for b in BENCHMARK_ORDER]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=METHOD_COLORS[method], label=METHOD_DISPLAY[method])
        for b, v in zip(bars, values, strict=True):
            ax.annotate(
                pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(BENCHMARK_ORDER)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Scenario 1 -- Top-1 accuracy summary vs. published baselines")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axis(ax)
    save_figure(fig, "figure_09_top1_summary", "main", output_dir, formats=FORMATS)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method ranked-outcome distribution figures (10/11/12)
# ─────────────────────────────────────────────────────────────────────────────

_OUTCOME_FIELD_BY_CATEGORY: dict[str, str] = {
    "Gold rank 1": "gold_rank_1",
    "Gold rank 2-3": "gold_rank_2_3",
    "Gold rank 4-5": "gold_rank_4_5",
    "No gold in Top 5": "no_gold_top5",
}

# Segments below this share are not annotated -- an unreadable "1%" label
# crowded into a sliver is worse than no label (Part 7).
_OUTCOME_LABEL_MIN_SHARE = 0.03


def _plot_outcome_stack(
    ax,
    methods: tuple[str, ...],
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    benchmark: str,
) -> None:
    x = list(range(len(methods)))
    bottoms = [0.0] * len(methods)
    for category in CROSS_METHOD_OUTCOME_CATEGORIES:
        field = _OUTCOME_FIELD_BY_CATEGORY[category]
        values = [getattr(distributions[benchmark][m], field) for m in methods]
        bars = ax.bar(
            x, values, bottom=bottoms, color=CROSS_METHOD_OUTCOME_COLORS[category], label=category,
            width=0.6, edgecolor="white", linewidth=0.6,
        )
        for b, v, bottom in zip(bars, values, bottoms, strict=True):
            if v >= _OUTCOME_LABEL_MIN_SHARE:
                ax.text(
                    b.get_x() + b.get_width() / 2, bottom + v / 2, f"{v * 100:.0f}%",
                    ha="center", va="center", fontsize=7.6,
                    color="white" if category == "Gold rank 1" else "black",
                )
        bottoms = [bo + v for bo, v in zip(bottoms, values, strict=True)]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{METHOD_DISPLAY_SHORT[m]}\nn={all_methods[benchmark][m].n:,}" for m in methods], fontsize=9.0
    )


def fig_10_all_methods_outcome_distribution(
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    output_dir: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), sharey=True)
    for ax, benchmark in zip(axes, BENCHMARK_ORDER, strict=True):
        _plot_outcome_stack(ax, METHOD_ORDER, all_methods, distributions, benchmark)
        ax.set_title(benchmark, fontsize=11.5)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of evaluated queries")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Scenario 1 -- ranked-outcome distribution (first-gold-rank composition)")
    save_figure(fig, "figure_10_all_methods_outcome_distribution", "main", output_dir, formats=FORMATS)


def _pairwise_outcome_distribution_figure(
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    baseline_method: str,
    basename: str,
    output_dir: Path,
    title: str,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 5.8), sharey=True)
    methods = ("ours", baseline_method)
    for ax, benchmark in zip(axes, BENCHMARK_ORDER, strict=True):
        _plot_outcome_stack(ax, methods, all_methods, distributions, benchmark)
        ax.set_title(benchmark, fontsize=11.0)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of evaluated queries")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title)
    save_figure(fig, basename, "pairwise", output_dir, formats=FORMATS)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES.md
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return pct(v)


def write_figures_md(
    *,
    our_runs: dict[str, OurRun],
    all_methods: dict[str, dict[str, MethodMetrics]],
    distributions: dict[str, dict[str, OutcomeDistribution]],
    delta_om_rows: list[dict[str, Any]],
    delta_t2t_rows: list[dict[str, Any]],
    baselines_source_path: Path,
    path: Path,
) -> None:
    ours = our_runs["OLS-EFO (full)"]

    def method_row(benchmark: str, method: str) -> str:
        m = all_methods[benchmark][method]
        return f"{METHOD_DISPLAY[method]} (n={m.n}): Top-1={_fmt_pct(m.top1)}, Top-3={_fmt_pct(m.top3)}, Top-5={_fmt_pct(m.top5)}, MRR={m.mrr:.3f}"

    delta_lines_om = "\n".join(
        f"- {row['benchmark']}: ΔTop-1={row['delta_top1_pp']:+.1f} pp, "
        f"ΔTop-3={row['delta_top3_pp']:+.1f} pp, ΔTop-5={row['delta_top5_pp']:+.1f} pp, "
        f"ΔMRR={row['delta_mrr']:+.3f}"
        for row in delta_om_rows
    )
    delta_lines_t2t = "\n".join(
        f"- {row['benchmark']}: ΔTop-1={row['delta_top1_pp']:+.1f} pp, "
        f"ΔTop-3={row['delta_top3_pp']:+.1f} pp, ΔTop-5={row['delta_top5_pp']:+.1f} pp, "
        f"ΔMRR={row['delta_mrr']:+.3f}"
        for row in delta_t2t_rows
    )

    lines: list[str] = []
    lines.append("# Scenario 1 -- published-baseline comparison figures")
    lines.append("")
    lines.append(
        "This suite compares **LLM Ontology Mapper** (this repository's Scenario 1 EFO "
        "experiments) against two published baselines from the MetaHarmonizer paper: "
        "**MetaHarmonizer (OM)** and **text2term (t2t)**. It is analysis/visualization only "
        "-- it makes zero mapping/LLM/retrieval/ontology-validator/network calls, and never "
        "modifies the original completed run directories or the pre-existing model-comparison "
        "or Scenario 1 figures."
    )
    lines.append("")

    # ── Part 19: terminology ────────────────────────────────────────────────
    lines.append("### Published method terminology")
    lines.append("")
    lines.append(
        "- **OM = OntologyMapper.** In the MetaHarmonizer paper, OM is the "
        "ontology-standardization component of MetaHarmonizer, not the whole system. Figures "
        'in this suite therefore label it **"MetaHarmonizer (OM)"** rather than bare '
        '"MetaHarmonizer", so a reader is never left to infer what the acronym covers.'
    )
    lines.append(
        '- **t2t = text2term.** Figures label it **"text2term (t2t)"**.'
    )
    lines.append(
        '- Our own method is labeled **"LLM Ontology Mapper"** (this repository\'s formal '
        'name, per `README.md`), never a bare "model" or "our model", since a proper system '
        "name already exists."
    )
    lines.append(
        "- Internally (CSV `tool` column, Python identifiers) OM is keyed as "
        '`metaharmonizer_ontology_mapper` / `metaharmonizer_om` and t2t as `text2term`; only '
        "the *display* labels above are used on any figure or in prose."
    )
    lines.append("")

    # ── Part 20: denominators ───────────────────────────────────────────────
    lines.append("### Benchmark denominators")
    lines.append("")
    lines.append("| Benchmark | LLM Ontology Mapper (n) | MetaHarmonizer (OM) (n) | text2term (t2t) (n) |")
    lines.append("| --- | --- | --- | --- |")
    for benchmark in BENCHMARK_ORDER:
        row = all_methods[benchmark]
        lines.append(
            f"| {benchmark} | {row['ours'].n} | {row['metaharmonizer_om'].n} | {row['text2term'].n} |"
        )
    lines.append("")
    lines.append(
        f"**OLS-EFO (full) denominator caveat.** LLM Ontology Mapper and MetaHarmonizer (OM) "
        f"are both evaluated on n={all_methods['OLS-EFO (full)']['ours'].n} unique queries; "
        f"text2term's published OLS-EFO (full) figure is evaluated on a *different* "
        f"n={all_methods['OLS-EFO (full)']['text2term'].n} (it reports over the "
        "full 7,504-row mapping-pair set rather than the 7,377 deduplicated unique queries "
        "used here). Every OLS-EFO figure and table in this suite that includes text2term "
        "shows both denominators; this is **not** an identical-N comparison and should not "
        "be presented as one."
    )
    lines.append("")
    lines.append(
        "**UKBB-EFO caveat.** UKBB-EFO's gold namespace composition differs from the "
        "100%-EFO-native OLS-EFO and Biomappings-EFO gold sets used elsewhere in this "
        "repository's Scenario 1 evaluation (see the Scenario 1 UKBB run's own "
        "dataset_validation.json / README notes). Top-k/MRR values remain directly comparable "
        "across methods *within* UKBB-EFO (all three methods are scored against the same gold "
        "set), but UKBB-EFO's absolute numbers should not be read as equivalent in difficulty "
        "to OLS-EFO or Biomappings-EFO."
    )
    lines.append("")

    # ── Part 21: source note ────────────────────────────────────────────────
    lines.append("### Source of published baseline values")
    lines.append("")
    lines.append(
        f"OM and text2term values come from a single structured CSV, "
        f"`{baselines_source_path.name}` (schema: benchmark, tool, metric, value, denominator, "
        "source_publication, source_table_or_figure, notes; unit-fraction values), snapshotted "
        f"unchanged into `data/published_baselines_used.csv` on every run of this figure suite. "
        "Publication: the MetaHarmonizer paper's benchmark table, as supplied to this repository "
        "by the project maintainer. The exact table/figure number has **not** been independently "
        "re-verified against the published PDF in this codebase, so `source_table_or_figure` is "
        "recorded honestly as unverified rather than a fabricated citation. OLS-EFO (disease) "
        "rows, if ever added to that CSV, are never consumed by this figure suite -- our "
        "Scenario 1 experiments did not run that subset, so there is no matching three-method "
        "comparison for it."
    )
    lines.append("")

    # ── Part 22: our-method source ──────────────────────────────────────────
    lines.append("### Source of our (LLM Ontology Mapper) values")
    lines.append("")
    lines.append("Our Top-1/Top-3/Top-5/MRR values come from these exact completed Scenario 1 runs' "
                  "`scenario1_metrics.csv`, reconciled against each run's own `predictions.csv` using "
                  "the unmodified `scenario1_metrics.score_prediction`/`aggregate` utilities:")
    lines.append("")
    for benchmark in BENCHMARK_ORDER:
        run = our_runs[benchmark]
        lines.append(f"- **{benchmark}**: `{run.run_dir}`")
    lines.append("")
    lines.append(
        f"Derived from those runs' `experiment_config.json` (not hardcoded): model=`{ours.model}`, "
        f"retrieval_mode=`{ours.retrieval_mode}`, target_ontology=`{ours.target_ontology}`, "
        f"strict_target_ontology=`{ours.strict_target_ontology}`. All three runs share this "
        "configuration; see `data/our_scenario1_metrics_used.csv` for the per-benchmark values."
    )
    lines.append("")

    # ── Outcome-distribution derivation (required standalone section) ───────
    lines.append("## Outcome-distribution derivation")
    lines.append("")
    lines.append(
        "Four mutually-exclusive first-gold-rank bins are reconstructed identically for every "
        "method (LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t)) from cumulative "
        "Top-1/Top-3/Top-5 alone:"
    )
    lines.append("")
    lines.append("- Gold rank 1 = Top-1")
    lines.append("- Gold rank 2-3 = Top-3 − Top-1")
    lines.append("- Gold rank 4-5 = Top-5 − Top-3")
    lines.append("- No gold in Top 5 = 1 − Top-5")
    lines.append("")
    lines.append(
        "These are mutually exclusive categories reconstructed from cumulative Top-k metrics -- "
        "not a figure copied from the MetaHarmonizer paper (which does not report a "
        "first-gold-rank composition directly; see the online source note below) and not "
        "derived from any richer per-row status. The published MetaHarmonizer/text2term "
        "aggregate does not expose execution errors, abstentions, and ordinary no-hit mapped "
        "predictions as separate components, so those cannot be separated out for OM or "
        "text2term. For a fair, symmetric comparison, LLM Ontology Mapper's own richer per-row "
        "outcomes (execution error, abstained, mapped-but-missed) are likewise collapsed into "
        "\"No gold in Top 5\" in **these cross-method figures only** -- the existing internal "
        "Scenario 1/2 figures elsewhere in this repository continue to show that richer "
        "breakdown for our method alone."
    )
    lines.append("")
    lines.append(
        f"Each method's four bins are checked to sum to 1.0 within a "
        f"{OUTCOME_SUM_TOLERANCE:g} tolerance (published Top-k values are reported to one "
        "decimal percentage point, so tiny rounding slack is expected) before any figure is "
        "drawn; a larger discrepancy, or a negative implied bin from non-monotonic Top-k "
        "values, hard-fails rather than silently plotting."
    )
    lines.append("")

    lines.append("### Online source note: MetaHarmonizer vs. original text2term")
    lines.append("")
    lines.append(
        "**MetaHarmonizer paper, Figure 3.** Panel A reports cumulative Top-k performance for "
        "OntologyMapper (OM) and text2term (t2t) under MetaHarmonizer's own controlled "
        "comparison protocol -- this is the source of every Top-1/Top-3/Top-5 value used in "
        "this suite, including as the sole input to the four-bin derivation above. Panel B "
        "reports the composition of *correct* Top-1 OM predictions by pipeline resolving "
        "stage, and Panel C reports confidence distributions for correct vs. incorrect Top-1 "
        "predictions. **None of Figure 3's panels report a first-gold-rank distribution "
        "directly** -- the OM/t2t bars in Figures 10-12 are derived, not copied, from Panel "
        "A's cumulative Top-k values."
    )
    lines.append("")
    lines.append(
        "**Original text2term publication.** Separately, the original text2term paper reports "
        "its own Top-1 mapping-*relationship* distribution (Same / More Specific / More "
        "General / Sibling / Unrelated) for UKBB-EFO, Biomappings, and OLS, with public "
        "evaluation code/data. This is intentionally **not** used for the rank-composition "
        "figures in this suite, because (1) it classifies the *graph relationship* of a Top-1 "
        "prediction, a different quantity than first-gold rank; and (2) it comes from the "
        "original text2term paper's own evaluation protocol, not the MetaHarmonizer-controlled "
        "rerun that this suite uses as its t2t baseline everywhere else (see "
        "`data/published_baselines_used.csv`). Mixing the two would silently compare text2term "
        "under two different protocols inside the same figure -- the original graph-relation "
        "percentages are **not** interchangeable with the controlled rerun used here. A "
        "graph-distance comparison against the *original* text2term paper, if wanted, should "
        "be a separate analysis with its own explicit protocol caveat."
    )
    lines.append("")

    # ── Figure-by-figure ─────────────────────────────────────────────────────
    lines.append("## Figures")
    lines.append("")

    def figure_section(
        number: str, title: str, files: list[str], question: str, datasets: str, methods: str,
        metrics: str, xaxis: str, yaxis: str, interpretation: str, denominators: str,
        caveats: str, source_data: list[str],
    ) -> None:
        lines.append(f"### Figure {number} — {title}")
        lines.append("")
        lines.append("**Files**")
        for f in files:
            lines.append(f"- {f}")
        lines.append("")
        lines.append(f"**Question.** {question}")
        lines.append("")
        lines.append(f"**Datasets.** {datasets}")
        lines.append("")
        lines.append(f"**Methods.** {methods}")
        lines.append("")
        lines.append(f"**Metrics.** {metrics}")
        lines.append("")
        lines.append(f"**Axes.** x: {xaxis}; y: {yaxis}")
        lines.append("")
        lines.append(f"**Interpretation.** {interpretation}")
        lines.append("")
        lines.append(f"**Denominators.** {denominators}")
        lines.append("")
        lines.append(f"**Caveats.** {caveats}")
        lines.append("")
        lines.append("**Source data**")
        for d in source_data:
            lines.append(f"- {d}")
        lines.append("")

    figure_section(
        "1", "All-method Top-k comparison",
        ["main/figure_01_all_methods_topk.png", "main/figure_01_all_methods_topk.svg"],
        "How does LLM Ontology Mapper's Top-1/Top-3/Top-5 accuracy compare to MetaHarmonizer "
        "(OM) and text2term (t2t) on each benchmark?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).",
        "LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t) (three bars per metric group, "
        "this fixed order, never sorted by performance).",
        "Top-1, Top-3, Top-5 accuracy (fraction of queries where an acceptable gold code appears "
        "within the top-k ranked predictions).",
        "Top-1 / Top-3 / Top-5 (grouped within each panel)",
        "Accuracy, 0-100%, shared across all three panels so bar heights are directly comparable.",
        "Higher is better for every bar. Values annotated to one decimal percentage point.",
        "See the denominators table above; the OLS-EFO (full) panel subtitle explicitly shows "
        "both n=7,377 (ours/OM) and n=7,504 (text2term).",
        "Do not read the OLS-EFO (full) text2term bars as scored on the identical query set as "
        "the other two methods in that panel.",
        ["data/all_methods_topk.csv", "data/all_methods_comparison.csv"],
    )

    figure_section(
        "2", "All-method MRR comparison",
        ["main/figure_02_all_methods_mrr.png", "main/figure_02_all_methods_mrr.svg"],
        "How does ranking quality (not just Top-1 hit/miss) compare across methods?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t).",
        "Mean Reciprocal Rank (MRR), a unit-fraction ranking-quality score -- explicitly NOT a "
        "percentage.",
        "Benchmark",
        "MRR, 0-1, shared across the whole chart. Values annotated to three decimal places.",
        "Higher MRR means the correct code tends to rank closer to position 1 on average.",
        "Same as Figure 1 -- see denominators table.",
        "Kept as a separate figure from Top-k on purpose (Part 11) rather than mixed into the "
        "Top-k panels, since MRR is a continuous ranking score, not an accuracy fraction.",
        ["data/all_methods_comparison.csv"],
    )

    figure_section(
        "3", "LLM Ontology Mapper vs. MetaHarmonizer (OM) -- Top-k",
        ["pairwise/figure_03_our_model_vs_metaharmonizer_topk.png",
         "pairwise/figure_03_our_model_vs_metaharmonizer_topk.svg"],
        "Head-to-head: how does our method's Top-k accuracy compare specifically to "
        "MetaHarmonizer's OntologyMapper component?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, MetaHarmonizer (OM) only (text2term omitted from this pair).",
        "Top-1, Top-3, Top-5 accuracy.",
        "Top-1 / Top-3 / Top-5 (grouped within each panel)",
        "Accuracy, 0-100%, shared across panels.",
        "Higher is better. Same method colors as every other figure in this suite.",
        "OLS-EFO (full): both methods share n=7,377 here (this pair has no denominator "
        "mismatch -- the mismatch is specific to text2term).",
        "None beyond the general UKBB-EFO gold-namespace caveat above.",
        ["data/pairwise_vs_metaharmonizer.csv"],
    )

    figure_section(
        "4", "LLM Ontology Mapper vs. MetaHarmonizer (OM) -- MRR",
        ["pairwise/figure_04_our_model_vs_metaharmonizer_mrr.png",
         "pairwise/figure_04_our_model_vs_metaharmonizer_mrr.svg"],
        "Head-to-head ranking quality vs. MetaHarmonizer (OM).",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, MetaHarmonizer (OM).",
        "MRR (unit fraction, not a percentage).",
        "Benchmark",
        "MRR, 0-1.",
        "Higher is better.",
        "n=7,377 shared on OLS-EFO (full) for this pair.",
        "None beyond the general UKBB-EFO caveat.",
        ["data/pairwise_vs_metaharmonizer.csv"],
    )

    figure_section(
        "5", "LLM Ontology Mapper vs. text2term (t2t) -- Top-k",
        ["pairwise/figure_05_our_model_vs_text2term_topk.png",
         "pairwise/figure_05_our_model_vs_text2term_topk.svg"],
        "Head-to-head: how does our method's Top-k accuracy compare specifically to text2term?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, text2term (t2t) only.",
        "Top-1, Top-3, Top-5 accuracy.",
        "Top-1 / Top-3 / Top-5 (grouped within each panel)",
        "Accuracy, 0-100%, shared across panels.",
        "Higher is better.",
        "OLS-EFO (full) panel subtitle explicitly shows n ours=7,377 vs. n text2term=7,504 -- "
        "the one panel in this whole suite where the two bars being compared do not share a "
        "denominator.",
        "Do not read the OLS-EFO (full) panel as an identical-N comparison.",
        ["data/pairwise_vs_text2term.csv"],
    )

    figure_section(
        "6", "LLM Ontology Mapper vs. text2term (t2t) -- MRR",
        ["pairwise/figure_06_our_model_vs_text2term_mrr.png",
         "pairwise/figure_06_our_model_vs_text2term_mrr.svg"],
        "Head-to-head ranking quality vs. text2term.",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, text2term (t2t).",
        "MRR (unit fraction, not a percentage).",
        "Benchmark",
        "MRR, 0-1.",
        "Higher is better.",
        "OLS-EFO (full): n ours=7,377 vs. n text2term=7,504 -- see caveat above.",
        "Same OLS-EFO (full) denominator caveat as Figure 5.",
        ["data/pairwise_vs_text2term.csv"],
    )

    figure_section(
        "7", "Δ vs. MetaHarmonizer (OM)",
        ["pairwise/figure_07_delta_vs_metaharmonizer.png", "pairwise/figure_07_delta_vs_metaharmonizer.svg"],
        "By how many percentage points does LLM Ontology Mapper's Top-k accuracy exceed or "
        "trail MetaHarmonizer (OM), per benchmark and per k?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper minus MetaHarmonizer (OM). Δ = ours − OM.",
        "ΔTop-1, ΔTop-3, ΔTop-5, expressed in **percentage points**, never called a "
        "percentage change.",
        "Percentage-point difference (zero-centered, zero always visible)",
        "One horizontal bar per (benchmark, k) combination.",
        "A bar extending right of zero (positive) means LLM Ontology Mapper is higher on that "
        "metric; a bar extending left (negative) means MetaHarmonizer (OM) is higher. No "
        "statistical test is implied by bar length alone.",
        "n=7,377 shared with MetaHarmonizer (OM) on every benchmark in this chart (no OLS "
        "denominator mismatch here -- that only affects the text2term comparison).",
        "ΔMRR is not plotted here (it is a raw unit-fraction difference, not percentage "
        "points) -- see the per-benchmark ΔMRR list below and "
        "data/delta_vs_metaharmonizer.csv.\n\n" + delta_lines_om,
        ["data/delta_vs_metaharmonizer.csv"],
    )

    figure_section(
        "8", "Δ vs. text2term (t2t)",
        ["pairwise/figure_08_delta_vs_text2term.png", "pairwise/figure_08_delta_vs_text2term.svg"],
        "By how many percentage points does LLM Ontology Mapper's Top-k accuracy exceed or "
        "trail text2term, per benchmark and per k?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper minus text2term (t2t). Δ = ours − t2t.",
        "ΔTop-1, ΔTop-3, ΔTop-5, in percentage points.",
        "Percentage-point difference (zero-centered, zero always visible)",
        "One horizontal bar per (benchmark, k) combination.",
        "Same reading as Figure 7, against text2term instead of MetaHarmonizer (OM).",
        "OLS-EFO (full) bars in this chart compare n=7,377 (ours) against n=7,504 (text2term) "
        "-- the same denominator mismatch as Figures 1 and 5.",
        "ΔMRR is not plotted (unit-fraction difference, not percentage points) -- see below "
        "and data/delta_vs_text2term.csv.\n\n" + delta_lines_t2t,
        ["data/delta_vs_text2term.csv"],
    )

    figure_section(
        "9", "Top-1-only headline summary",
        ["main/figure_09_top1_summary.png", "main/figure_09_top1_summary.svg"],
        "A single, presentation-friendly headline view of Top-1 accuracy across all three "
        "benchmarks and methods, without the Top-3/Top-5 facets of Figure 1.",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t).",
        "Top-1 accuracy only.",
        "Benchmark",
        "Accuracy, 0-100%.",
        "Higher is better. Generated because a single-panel, single-metric chart is easier to "
        "drop into a slide than Figure 1's three-panel Top-1/3/5 facet grid -- it is a genuinely "
        "different layout, not a duplicate.",
        "Same denominator caveats as Figure 1 (OLS-EFO (full) text2term n=7,504 vs. n=7,377).",
        "Read alongside Figure 1 for Top-3/Top-5; this figure intentionally omits them.",
        ["data/all_methods_comparison.csv"],
    )

    figure_section(
        "10", "All-method ranked-outcome distribution",
        ["main/figure_10_all_methods_outcome_distribution.png",
         "main/figure_10_all_methods_outcome_distribution.svg"],
        "How is each method's probability mass distributed across first-gold-rank buckets, "
        "not just collapsed into a single Top-1 hit/miss number?",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).",
        "LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t) -- one 100%-stacked bar per "
        "method, per panel, in this fixed order.",
        "Four mutually-exclusive rank bins -- Gold rank 1, Gold rank 2-3, Gold rank 4-5, No gold "
        "in Top 5 -- see 'Outcome-distribution derivation' above for the exact formulas.",
        "Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly "
        "underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / "
        "MetaHarmonizer (OM) / text2term (t2t) names)",
        "Share of evaluated queries, 0-100%, shared across all three panels.",
        "Segment **color encodes outcome category, not method** -- the same four colors are "
        "reused for every method and every panel in this whole suite; method identity comes "
        "only from the x-axis label. A larger 'Gold rank 1' share is better; a larger 'No gold "
        "in Top 5' share is worse. Segments below 3% are left unlabeled to avoid unreadable "
        "clutter -- exact values are always in the CSV regardless of whether they were "
        "annotated on the chart.",
        "OLS-EFO (full): text2term's bar is explicitly labeled n=7,504 while LLM Ontology "
        "Mapper's and MetaHarmonizer (OM)'s bars are labeled n=7,377, directly under each bar.",
        "This composition is **reconstructed, not measured directly**, for MetaHarmonizer (OM) "
        "and text2term (t2t): their published aggregate does not expose execution errors, "
        "abstentions, or ordinary no-hit predictions separately, so all non-Top-5 outcomes "
        "collapse into 'No gold in Top 5' for every method shown here, including ours (see "
        "'Outcome-distribution derivation' above). Do not read 'No gold in Top 5' as "
        "'abstained' or 'errored' for any method.",
        ["data/outcome_distribution_all_methods.csv", "data/outcome_distribution_all_methods.md"],
    )

    figure_section(
        "11", "LLM Ontology Mapper vs. MetaHarmonizer (OM) — ranked-outcome distribution",
        ["pairwise/figure_11_outcome_distribution_vs_metaharmonizer.png",
         "pairwise/figure_11_outcome_distribution_vs_metaharmonizer.svg"],
        "Head-to-head first-gold-rank composition against MetaHarmonizer (OM) only.",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, MetaHarmonizer (OM) only (text2term omitted from this pair).",
        "Same four rank bins as Figure 10.",
        "Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly "
        "underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / "
        "MetaHarmonizer (OM) / text2term (t2t) names)",
        "Share of evaluated queries, 0-100%, shared across panels.",
        "Same reading as Figure 10, restricted to this pair; same category colors, same "
        "3%-minimum labeling rule.",
        "n=7,377 shared by both methods on OLS-EFO (full) -- no denominator mismatch in this "
        "pair (the mismatch is specific to text2term).",
        "Same reconstruction caveat as Figure 10 -- 'No gold in Top 5' is not 'abstained' or "
        "'errored' for either method.",
        ["data/outcome_distribution_all_methods.csv"],
    )

    figure_section(
        "12", "LLM Ontology Mapper vs. text2term (t2t) — ranked-outcome distribution",
        ["pairwise/figure_12_outcome_distribution_vs_text2term.png",
         "pairwise/figure_12_outcome_distribution_vs_text2term.svg"],
        "Head-to-head first-gold-rank composition against text2term (t2t) only.",
        "UKBB-EFO, Biomappings-EFO, OLS-EFO (full).",
        "LLM Ontology Mapper, text2term (t2t) only.",
        "Same four rank bins as Figure 10.",
        "Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly "
        "underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / "
        "MetaHarmonizer (OM) / text2term (t2t) names)",
        "Share of evaluated queries, 0-100%, shared across panels.",
        "Same reading as Figure 10, restricted to this pair.",
        "OLS-EFO (full): n ours=7,377 vs. n text2term=7,504, labeled directly under each bar -- "
        "the same denominator mismatch as Figures 1, 5, and 8.",
        "Same reconstruction caveat as Figure 10. text2term's bars here use the "
        "MetaHarmonizer-controlled rerun values, NOT the original text2term paper's own "
        "Same/More-Specific/.../Unrelated Top-1 graph-relationship distribution -- see the "
        "online source note above for why those are never mixed.",
        ["data/outcome_distribution_all_methods.csv"],
    )

    lines.append(
        "**On per-benchmark figures 13-15.** A dense per-benchmark breakout "
        "(figure_13_ukbb_outcome_distribution / figure_14_biomappings_outcome_distribution / "
        "figure_15_ols_outcome_distribution) was considered and deliberately **not generated**: "
        "Figure 10's three-panel layout already keeps each panel to three bars of four segments "
        "each, matches the visual density of the existing all-method Top-k/pairwise figures "
        "elsewhere in this suite, and was confirmed legible on inspection (labels not clipped, "
        "small segments cleanly omitted rather than overlapping). Splitting it into three "
        "single-benchmark figures would be redundant with Figure 10 without adding readability."
    )
    lines.append("")

    lines.append("## Metrics not compared to published methods")
    lines.append("")
    lines.append(
        "**Precision/Recall/F1** and the **graph-distance taxonomy** (Same / More Specific / "
        "More General / Sibling / Unrelated) are Scenario 1's own graph-based metrics. The "
        "supplied published baseline table provides only Top-1/Top-3/Top-5/MRR for OM and "
        "text2term -- it does not provide comparable Precision/Recall/F1 or graph-distance "
        "values for those tools, so no cross-method figure is generated for them here. They "
        "remain available, for our method only, in the existing Scenario 1 figure suite."
    )
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "Every comparison above is a direct read of the plotted values (\"LLM Ontology Mapper "
        "has a higher/lower Top-1 than OM on this benchmark\"). No hypothesis test has been run, "
        "so no claim of statistical significance is made or implied, and no causal explanation "
        "(e.g. about *why* one architecture outperforms another) should be drawn from these "
        "figures alone."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    our_runs: dict[str, OurRun]
    all_methods: dict[str, dict[str, MethodMetrics]]


def build_all(
    *, ols_dir: Path, ukbb_dir: Path, biomappings_dir: Path, baselines_path: Path, output_dir: Path,
) -> BuildResult:
    our_runs = load_all_official_runs(ols_dir=ols_dir, ukbb_dir=ukbb_dir, biomappings_dir=biomappings_dir)
    baselines = load_baselines(baselines_path)
    all_methods = build_all_methods(our_runs, baselines)

    data_dir = output_dir / "data"
    write_baselines_snapshot(baselines_path, data_dir / "published_baselines_used.csv")
    write_our_scenario1_metrics_used_csv(our_runs, data_dir / "our_scenario1_metrics_used.csv")
    write_all_methods_topk_csv(all_methods, data_dir / "all_methods_topk.csv")
    write_all_methods_comparison_csv(all_methods, data_dir / "all_methods_comparison.csv")
    write_pairwise_csv(all_methods, "metaharmonizer_om", data_dir / "pairwise_vs_metaharmonizer.csv")
    write_pairwise_csv(all_methods, "text2term", data_dir / "pairwise_vs_text2term.csv")

    delta_om_rows = compute_deltas(all_methods, "metaharmonizer_om")
    delta_t2t_rows = compute_deltas(all_methods, "text2term")
    write_delta_csv(delta_om_rows, data_dir / "delta_vs_metaharmonizer.csv")
    write_delta_csv(delta_t2t_rows, data_dir / "delta_vs_text2term.csv")

    distributions = build_outcome_distributions(all_methods)
    write_outcome_distribution_csv(all_methods, distributions, data_dir / "outcome_distribution_all_methods.csv")
    write_outcome_distribution_md(all_methods, distributions, data_dir / "outcome_distribution_all_methods.md")

    fig_01_all_methods_topk(all_methods, output_dir)
    fig_02_all_methods_mrr(all_methods, output_dir)
    _pairwise_topk_figure(
        all_methods, "metaharmonizer_om", "figure_03_our_model_vs_metaharmonizer_topk", output_dir,
        "LLM Ontology Mapper vs. MetaHarmonizer (OM) — Top-k accuracy",
    )
    _pairwise_mrr_figure(
        all_methods, "metaharmonizer_om", "figure_04_our_model_vs_metaharmonizer_mrr", output_dir,
        "LLM Ontology Mapper vs. MetaHarmonizer (OM) — MRR",
    )
    _pairwise_topk_figure(
        all_methods, "text2term", "figure_05_our_model_vs_text2term_topk", output_dir,
        "LLM Ontology Mapper vs. text2term (t2t) — Top-k accuracy",
    )
    _pairwise_mrr_figure(
        all_methods, "text2term", "figure_06_our_model_vs_text2term_mrr", output_dir,
        "LLM Ontology Mapper vs. text2term (t2t) — MRR",
    )
    _delta_figure(
        delta_om_rows, "figure_07_delta_vs_metaharmonizer", output_dir,
        "Δ Top-k accuracy: LLM Ontology Mapper − MetaHarmonizer (OM)",
    )
    _delta_figure(
        delta_t2t_rows, "figure_08_delta_vs_text2term", output_dir,
        "Δ Top-k accuracy: LLM Ontology Mapper − text2term (t2t)",
    )
    fig_09_top1_summary(all_methods, output_dir)
    fig_10_all_methods_outcome_distribution(all_methods, distributions, output_dir)
    _pairwise_outcome_distribution_figure(
        all_methods, distributions, "metaharmonizer_om", "figure_11_outcome_distribution_vs_metaharmonizer",
        output_dir, "LLM Ontology Mapper vs. MetaHarmonizer (OM) — ranked-outcome distribution",
    )
    _pairwise_outcome_distribution_figure(
        all_methods, distributions, "text2term", "figure_12_outcome_distribution_vs_text2term",
        output_dir, "LLM Ontology Mapper vs. text2term (t2t) — ranked-outcome distribution",
    )

    write_figures_md(
        our_runs=our_runs,
        all_methods=all_methods,
        distributions=distributions,
        delta_om_rows=delta_om_rows,
        delta_t2t_rows=delta_t2t_rows,
        baselines_source_path=baselines_path,
        path=output_dir / "FIGURES.md",
    )

    return BuildResult(output_dir=output_dir, our_runs=our_runs, all_methods=all_methods)
