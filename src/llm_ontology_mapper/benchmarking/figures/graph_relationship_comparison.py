"""
Scenario 1 -- Top-1 ontology graph-relationship comparison: LLM Ontology
Mapper vs. text2term, using the five categories from text2term-evaluation's
`compare_ontology_mappings.compare_mappings()`:

    Same, More Specific, More General, Sibling, Unrelated

======================================================================
THIS IS A DIFFERENT SOURCE THAN THE CONTROLLED TOP-K COMPARISON
======================================================================
The rest of this figure suite (published_comparison.py, Figures 1-12)
compares against the MetaHarmonizer paper's own CONTROLLED RERUN of
text2term (Top-1/Top-3/Top-5/MRR only -- no graph-relationship categories).
THIS module instead compares against the ORIGINAL text2term publication's
own Table 1 (the only public source of these five graph categories for
text2term). These are two different text2term executions under two
different protocols and are never merged into one source or implied to be
the same run -- see FIGURES.md "CONTROLLED TOP-K BASELINE vs.
GRAPH-RELATIONSHIP BASELINE".

Analysis/visualization only. Makes ZERO mapping/LLM/retrieval/validator/
network calls -- every input is a file already on disk (our own persisted
Scenario 1 run artifacts, plus the published Table 1 values stored in
data/text2term_graph_relationship_baseline.csv).

MetaHarmonizer's own OntologyMapper (OM) is deliberately NOT included in
these figures: the published OM table provides cumulative Top-k metrics
only, not a five-category graph-relationship breakdown, so there is no
equivalent OM value to plot here (Part 16).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from llm_ontology_mapper.benchmarking import scenario1_graph_distance as graph_distance
from llm_ontology_mapper.benchmarking import text2term_alignment as align
from llm_ontology_mapper.benchmarking.figures import published_comparison as pc
from llm_ontology_mapper.benchmarking.figures.common import write_csv
from llm_ontology_mapper.benchmarking.figures.style import apply_style, save_figure, style_axis

# Same PNG+SVG-only policy as the rest of this suite -- reused, never
# redefined, so the two modules can never silently drift apart.
FORMATS: tuple[str, ...] = pc.FORMATS

RELATIONSHIP_ORDER: tuple[str, ...] = ("Same", "More Specific", "More General", "Sibling", "Unrelated")
NO_TOP1_CATEGORY = "No Top-1 prediction"
END_TO_END_CATEGORY_ORDER: tuple[str, ...] = (*RELATIONSHIP_ORDER, NO_TOP1_CATEGORY)

# Distinct qualitative palette (ColorBrewer Dark2, colorblind-safe) --
# deliberately NOT reusing METHOD_COLORS or CROSS_METHOD_OUTCOME_COLORS from
# published_comparison.py, since these are a different taxonomy (graph
# relationship, not method identity or first-gold-rank). "No Top-1
# prediction" reuses the same gray-for-no-answer convention as Scenario 2's
# "Abstained" category.
RELATIONSHIP_COLORS: dict[str, str] = {
    "Same": "#1b9e77",
    "More Specific": "#7570b3",
    "More General": "#66a61e",
    "Sibling": "#e6ab02",
    "Unrelated": "#d95f02",
    NO_TOP1_CATEGORY: "#666666",
}

EXPECTED_SOURCE_REPOSITORY = "https://github.com/rsgoncalves/text2term-evaluation"
EXPECTED_EFO_VERSION = "3.62.0"
EXPECTED_TEXT2TERM_VERSION = "4.1.2"

# Segments below this share are left unannotated (Part 7).
_LABEL_MIN_SHARE = 0.03


class GraphComparisonError(RuntimeError):
    """Base class for every hard-fail condition in this module."""


class GraphCompatibilityError(GraphComparisonError):
    pass


class GraphBaselineError(GraphComparisonError):
    pass


class OurGraphLoadError(GraphComparisonError):
    pass


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def _text_color_for_bg(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 140 else "black"


# ─────────────────────────────────────────────────────────────────────────────
# Part 1/13: verify our graph evaluator's priority/version/source is
# demonstrably the same one documented for text2term's compare_mappings --
# never assumed.
# ─────────────────────────────────────────────────────────────────────────────


def verify_graph_evaluator_compatibility(run_dir: Path) -> dict[str, Any]:
    if graph_distance.ALL_RELATIONSHIPS != RELATIONSHIP_ORDER:
        raise GraphCompatibilityError(
            f"scenario1_graph_distance.ALL_RELATIONSHIPS order {graph_distance.ALL_RELATIONSHIPS} does not "
            f"match the expected text2term compare_mappings priority {RELATIONSHIP_ORDER} -- refusing to "
            "plot as if these categories mean the same thing"
        )
    if graph_distance.EFO_VERSION != EXPECTED_EFO_VERSION:
        raise GraphCompatibilityError(
            f"scenario1_graph_distance.EFO_VERSION={graph_distance.EFO_VERSION!r}, expected "
            f"{EXPECTED_EFO_VERSION!r}"
        )
    if graph_distance.SOURCE_REPOSITORY != EXPECTED_SOURCE_REPOSITORY:
        raise GraphCompatibilityError(
            f"scenario1_graph_distance.SOURCE_REPOSITORY={graph_distance.SOURCE_REPOSITORY!r}, expected "
            f"{EXPECTED_SOURCE_REPOSITORY!r}"
        )

    metadata_path = run_dir / "graph_reference_metadata.json"
    if not metadata_path.exists():
        raise GraphCompatibilityError(f"missing graph_reference_metadata.json under {run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("efo_version") != EXPECTED_EFO_VERSION:
        raise GraphCompatibilityError(
            f"{metadata_path}: efo_version={metadata.get('efo_version')!r}, expected {EXPECTED_EFO_VERSION!r}"
        )
    if metadata.get("source_repository") != EXPECTED_SOURCE_REPOSITORY:
        raise GraphCompatibilityError(
            f"{metadata_path}: source_repository={metadata.get('source_repository')!r}, expected "
            f"{EXPECTED_SOURCE_REPOSITORY!r}"
        )
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Our persisted graph-distance data (Part 4/5) -- never manually typed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OurGraphDistribution:
    benchmark: str
    run_dir: Path
    n: int
    mapped_count: int
    unmapped_count: int
    error_count: int
    no_top1_count: int
    counts: dict[str, int]

    def mapped_only_proportions(self) -> dict[str, float]:
        return {rel: self.counts[rel] / self.mapped_count for rel in RELATIONSHIP_ORDER}

    def end_to_end_proportions(self) -> dict[str, float]:
        result = {rel: self.counts[rel] / self.n for rel in RELATIONSHIP_ORDER}
        result[NO_TOP1_CATEGORY] = self.no_top1_count / self.n
        return result


def _read_execution_diagnostics(path: Path) -> dict[str, int]:
    if not path.exists():
        raise OurGraphLoadError(f"missing execution_diagnostics.csv: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return {k: int(row[k]) for k in ("total", "mapped_count", "unmapped_count", "error_count")}


def _read_graph_distance_summary(path: Path) -> dict[str, int]:
    if not path.exists():
        raise OurGraphLoadError(f"missing graph_distance_summary.csv: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {row["relationship"]: int(row["count"]) for row in rows}


def load_our_graph_distribution(benchmark: str, run_dir: Path) -> OurGraphDistribution:
    """Loads and cross-checks our graph-relationship counts (Part 5): the
    five classified relationships must sum to mapped_count, and
    graph_distance_summary.csv's 'Not Applicable' bucket must equal
    unmapped_count + error_count -- both from execution_diagnostics.csv,
    never re-derived independently. Reuses pc.load_official_run for the
    completed/N/source_dataset_path checks rather than re-implementing them.
    """
    pc.load_official_run(benchmark, run_dir)
    verify_graph_evaluator_compatibility(run_dir)

    diagnostics = _read_execution_diagnostics(run_dir / "execution_diagnostics.csv")
    summary_counts = _read_graph_distance_summary(run_dir / "graph_distance_summary.csv")

    counts = {rel: summary_counts.get(rel, 0) for rel in RELATIONSHIP_ORDER}
    not_applicable = summary_counts.get("Not Applicable", 0)
    no_top1 = diagnostics["unmapped_count"] + diagnostics["error_count"]

    if not_applicable != no_top1:
        raise OurGraphLoadError(
            f"{run_dir}: graph_distance_summary.csv 'Not Applicable' count ({not_applicable}) != "
            f"unmapped_count+error_count ({no_top1}) from execution_diagnostics.csv -- refusing to plot an "
            "inconsistent no-prediction bucket"
        )
    classified_sum = sum(counts.values())
    if classified_sum != diagnostics["mapped_count"]:
        raise OurGraphLoadError(
            f"{run_dir}: sum of the five graph-relationship counts ({classified_sum}) != mapped_count "
            f"({diagnostics['mapped_count']}) from execution_diagnostics.csv"
        )
    if classified_sum + not_applicable != diagnostics["total"]:
        raise OurGraphLoadError(
            f"{run_dir}: classified ({classified_sum}) + Not Applicable ({not_applicable}) != total "
            f"({diagnostics['total']})"
        )

    return OurGraphDistribution(
        benchmark=benchmark,
        run_dir=run_dir,
        n=diagnostics["total"],
        mapped_count=diagnostics["mapped_count"],
        unmapped_count=diagnostics["unmapped_count"],
        error_count=diagnostics["error_count"],
        no_top1_count=no_top1,
        counts=counts,
    )


def load_gold_count_distribution(run_dir: Path) -> dict[int, int]:
    """Part 12: audit single- vs. multi-gold queries from our own persisted
    dataset_validation.json -- never re-derived from predictions.csv."""
    path = run_dir / "dataset_validation.json"
    if not path.exists():
        raise OurGraphLoadError(f"missing dataset_validation.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("gold_count_distribution", {})
    return {int(k): int(v) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Published original-text2term Table 1 baseline (Part 2/14) -- ONE
# structured source of truth, never hardcoded a second time below.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Text2termGraphBaseline:
    benchmark: str
    n: int
    text2term_version: str
    efo_version: str
    counts: dict[str, int]

    def proportions(self) -> dict[str, float]:
        return {rel: self.counts[rel] / self.n for rel in RELATIONSHIP_ORDER}


def load_text2term_graph_baseline(path: Path) -> dict[str, Text2termGraphBaseline]:
    if not path.exists():
        raise GraphBaselineError(f"missing text2term graph-relationship baseline CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_benchmark: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        benchmark = row["benchmark"]
        if benchmark not in pc.BENCHMARK_ORDER:
            raise GraphBaselineError(f"unknown benchmark {benchmark!r} in {path}")
        relationship = row["relationship"]
        if relationship not in RELATIONSHIP_ORDER:
            raise GraphBaselineError(f"unknown relationship {relationship!r} in {path}")
        if row.get("source") != "original_text2term_paper":
            raise GraphBaselineError(
                f"unexpected source {row.get('source')!r} in {path} -- expected 'original_text2term_paper' "
                "(this baseline must never be the MetaHarmonizer-controlled rerun)"
            )
        if row.get("efo_version") != EXPECTED_EFO_VERSION:
            raise GraphBaselineError(
                f"{path}: efo_version={row.get('efo_version')!r}, expected {EXPECTED_EFO_VERSION!r}"
            )
        bucket = by_benchmark.setdefault(benchmark, {})
        if relationship in bucket:
            raise GraphBaselineError(f"duplicate relationship row {benchmark}/{relationship} in {path}")
        bucket[relationship] = row

    result: dict[str, Text2termGraphBaseline] = {}
    for benchmark in pc.BENCHMARK_ORDER:
        if benchmark not in by_benchmark:
            raise GraphBaselineError(f"missing benchmark {benchmark!r} in {path}")
        bucket = by_benchmark[benchmark]
        missing = set(RELATIONSHIP_ORDER) - set(bucket)
        if missing:
            raise GraphBaselineError(f"{benchmark}: missing relationship rows {sorted(missing)} in {path}")

        ns = {int(r["n"]) for r in bucket.values()}
        if len(ns) != 1:
            raise GraphBaselineError(f"{benchmark}: inconsistent n across relationship rows in {path}: {ns}")
        n = ns.pop()

        counts = {rel: int(bucket[rel]["count"]) for rel in RELATIONSHIP_ORDER}
        total = sum(counts.values())
        if total != n:
            raise GraphBaselineError(
                f"{benchmark}: published relationship counts sum to {total}, not n={n} -- cannot safely "
                "assume 'No Top-1 prediction'=0 for text2term on this benchmark"
            )

        # Cross-check the CSV's stored 'proportion' column (the paper's own
        # independently-rounded percentage) against count/n: a small gap is
        # expected from the paper's own rounding, but a larger one signals a
        # transcription error in either column.
        for rel in RELATIONSHIP_ORDER:
            stored_proportion = float(bucket[rel]["proportion"])
            derived_proportion = counts[rel] / n
            if abs(stored_proportion - derived_proportion) > 0.005:
                raise GraphBaselineError(
                    f"{benchmark}/{rel}: stored proportion {stored_proportion} disagrees with "
                    f"count/n={derived_proportion:.4f} by more than rounding tolerance in {path} -- "
                    "check for a transcription error in count, n, or proportion"
                )

        result[benchmark] = Text2termGraphBaseline(
            benchmark=benchmark,
            n=n,
            text2term_version=next(iter({r["text2term_version"] for r in bucket.values()})),
            efo_version=next(iter({r["efo_version"] for r in bucket.values()})),
            counts=counts,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Combine + data writers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphComparisonData:
    our: dict[str, OurGraphDistribution]
    text2term: dict[str, Text2termGraphBaseline]


def load_all_graph_data(
    *, ols_dir: Path, ukbb_dir: Path, biomappings_dir: Path, baseline_path: Path
) -> GraphComparisonData:
    dirs = {"UKBB-EFO": ukbb_dir, "Biomappings-EFO": biomappings_dir, "OLS-EFO (full)": ols_dir}
    our = {benchmark: load_our_graph_distribution(benchmark, d) for benchmark, d in dirs.items()}
    text2term = load_text2term_graph_baseline(baseline_path)
    return GraphComparisonData(our=our, text2term=text2term)


def write_graph_relationship_mapped_only_csv(data: GraphComparisonData, path: Path) -> None:
    rows = []
    for benchmark in pc.BENCHMARK_ORDER:
        our = data.our[benchmark]
        t2t = data.text2term[benchmark]
        our_props = our.mapped_only_proportions()
        t2t_props = t2t.proportions()
        rows.append(
            {"benchmark": benchmark, "method": "Our method", "denominator_n": our.mapped_count, "total_n": our.n,
             **{_slug(rel): our_props[rel] for rel in RELATIONSHIP_ORDER}}
        )
        rows.append(
            {"benchmark": benchmark, "method": "text2term", "denominator_n": t2t.n, "total_n": t2t.n,
             **{_slug(rel): t2t_props[rel] for rel in RELATIONSHIP_ORDER}}
        )
    fields = ["benchmark", "method", "denominator_n", "total_n", *[_slug(r) for r in RELATIONSHIP_ORDER]]
    write_csv(rows, fields, path)


def write_graph_relationship_end_to_end_csv(data: GraphComparisonData, path: Path) -> None:
    rows = []
    for benchmark in pc.BENCHMARK_ORDER:
        our = data.our[benchmark]
        t2t = data.text2term[benchmark]
        our_props = our.end_to_end_proportions()
        t2t_props = dict(t2t.proportions())
        t2t_props[NO_TOP1_CATEGORY] = 0.0
        rows.append(
            {"benchmark": benchmark, "method": "Our method", "n": our.n,
             **{_slug(c): our_props[c] for c in END_TO_END_CATEGORY_ORDER}}
        )
        rows.append(
            {"benchmark": benchmark, "method": "text2term", "n": t2t.n,
             **{_slug(c): t2t_props[c] for c in END_TO_END_CATEGORY_ORDER}}
        )
    fields = ["benchmark", "method", "n", *[_slug(c) for c in END_TO_END_CATEGORY_ORDER]]
    write_csv(rows, fields, path)


def write_denominator_comparison_csv(data: GraphComparisonData, path: Path) -> None:
    rows = [
        {
            "benchmark": benchmark,
            "our_n": data.our[benchmark].n,
            "our_mapped_n": data.our[benchmark].mapped_count,
            "our_no_top1_n": data.our[benchmark].no_top1_count,
            "text2term_n": data.text2term[benchmark].n,
            "denominators_match": data.our[benchmark].n == data.text2term[benchmark].n,
        }
        for benchmark in pc.BENCHMARK_ORDER
    ]
    write_csv(rows, ["benchmark", "our_n", "our_mapped_n", "our_no_top1_n", "text2term_n", "denominators_match"], path)


def write_multi_gold_audit_csv(run_dirs: dict[str, Path], path: Path) -> None:
    rows = []
    for benchmark in pc.BENCHMARK_ORDER:
        distribution = load_gold_count_distribution(run_dirs[benchmark])
        for gold_count in sorted(distribution):
            rows.append({"benchmark": benchmark, "gold_count": gold_count, "n_queries": distribution[gold_count]})
    write_csv(rows, ["benchmark", "gold_count", "n_queries"], path)


def compute_descriptive_deltas_mapped_only(data: GraphComparisonData) -> list[dict[str, Any]]:
    """Δ = ours (mapped-only view) - text2term (published), in percentage
    points. Explicitly a DESCRIPTIVE published-protocol difference (Part
    17): the two sides use different, mismatched denominators (no
    common-query alignment was available -- see FIGURES.md), so this is
    never framed as a statistically validated or common-query comparison."""
    rows = []
    for benchmark in pc.BENCHMARK_ORDER:
        our_props = data.our[benchmark].mapped_only_proportions()
        t2t_props = data.text2term[benchmark].proportions()
        row: dict[str, Any] = {"benchmark": benchmark}
        for rel in RELATIONSHIP_ORDER:
            row[f"delta_{_slug(rel)}_pp"] = (our_props[rel] - t2t_props[rel]) * 100.0
        rows.append(row)
    return rows


def write_graph_relationship_delta_csv(rows: list[dict[str, Any]], path: Path) -> None:
    write_csv(rows, ["benchmark", *[f"delta_{_slug(r)}_pp" for r in RELATIONSHIP_ORDER]], path)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 13/14/16
# ─────────────────────────────────────────────────────────────────────────────


def _mapped_only_labels(our: OurGraphDistribution, t2t: Text2termGraphBaseline) -> tuple[str, str]:
    return (f"Our method\nmapped n={our.mapped_count:,}", f"text2term\nn={t2t.n:,} (published)")


def _end_to_end_labels(our: OurGraphDistribution, t2t: Text2termGraphBaseline) -> tuple[str, str]:
    return (f"Our method\nn={our.n:,}", f"text2term\nn={t2t.n:,} (published)")


def _plot_relationship_stack(
    ax, category_order: tuple[str, ...], our_props: dict[str, float], t2t_props: dict[str, float],
    labels: tuple[str, str],
) -> None:
    x = [0, 1]
    bottoms = [0.0, 0.0]
    for category in category_order:
        values = [our_props.get(category, 0.0), t2t_props.get(category, 0.0)]
        color = RELATIONSHIP_COLORS[category]
        bars = ax.bar(x, values, bottom=bottoms, color=color, label=category, width=0.6, edgecolor="white", linewidth=0.6)
        for b, v, bottom in zip(bars, values, bottoms, strict=True):
            if v >= _LABEL_MIN_SHARE:
                ax.text(
                    b.get_x() + b.get_width() / 2, bottom + v / 2, f"{v * 100:.0f}%",
                    ha="center", va="center", fontsize=7.6, color=_text_color_for_bg(color),
                )
        bottoms = [bo + v for bo, v in zip(bottoms, values, strict=True)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)


def fig_13_graph_relationships_mapped_only(data: GraphComparisonData, output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    for ax, benchmark in zip(axes, pc.BENCHMARK_ORDER, strict=True):
        our = data.our[benchmark]
        t2t = data.text2term[benchmark]
        _plot_relationship_stack(
            ax, RELATIONSHIP_ORDER, our.mapped_only_proportions(), t2t.proportions(), _mapped_only_labels(our, t2t)
        )
        ax.set_title(benchmark, fontsize=11.0)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of classifiable Top-1 predictions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Top-1 graph relationship vs. original text2term — classifiable predictions only")
    save_figure(fig, "figure_13_graph_relationships_mapped_only", "pairwise", output_dir, formats=FORMATS)


def fig_14_graph_relationships_end_to_end(data: GraphComparisonData, output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    for ax, benchmark in zip(axes, pc.BENCHMARK_ORDER, strict=True):
        our = data.our[benchmark]
        t2t = data.text2term[benchmark]
        t2t_props = dict(t2t.proportions())
        t2t_props[NO_TOP1_CATEGORY] = 0.0
        _plot_relationship_stack(
            ax, END_TO_END_CATEGORY_ORDER, our.end_to_end_proportions(), t2t_props, _end_to_end_labels(our, t2t)
        )
        ax.set_title(benchmark, fontsize=11.0)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of all evaluated queries")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Top-1 graph relationship vs. original text2term — end-to-end (all evaluated queries)")
    save_figure(fig, "figure_14_graph_relationships_end_to_end", "pairwise", output_dir, formats=FORMATS)


def fig_16_graph_relationship_delta_vs_text2term(delta_rows: list[dict[str, Any]], output_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in delta_rows:
        for rel in RELATIONSHIP_ORDER:
            labels.append(f"{row['benchmark']} — {rel}")
            values.append(float(row[f"delta_{_slug(rel)}_pp"]))
            colors.append(RELATIONSHIP_COLORS[rel])
    y_positions = list(range(len(labels)))[::-1]
    ax.barh(y_positions, values, color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.0)
    max_abs = max((abs(v) for v in values), default=1.0)
    pad = max(max_abs * 0.3, 1.0)
    ax.set_xlim(-(max_abs + pad), max_abs + pad)
    for yi, v in zip(y_positions, values, strict=True):
        ax.annotate(
            f"{v:+.1f} pp", (v, yi), xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
            ha="left" if v >= 0 else "right", va="center", fontsize=7.6,
        )
    ax.set_xlabel("Percentage points (Our method − text2term, mapped-only view)")
    ax.set_title("Descriptive published-protocol difference — NOT a common-query-aligned comparison")
    style_axis(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    save_figure(fig, "figure_16_graph_relationship_delta_vs_text2term", "pairwise", output_dir, formats=FORMATS)


# ─────────────────────────────────────────────────────────────────────────────
# Common-query-aligned figures (15/15b/15c) -- the PRIMARY graph-relationship
# comparison when alignment succeeds: both methods share an IDENTICAL
# denominator (the strict matched N), unlike Figures 13/14/16 above.
# ─────────────────────────────────────────────────────────────────────────────

_ALIGNED_LABEL_MIN_SHARE = 0.03


def _aligned_labels(aligned_n: int) -> tuple[str, str]:
    return (f"LLM Ontology Mapper\naligned n={aligned_n:,}", f"text2term\naligned n={aligned_n:,}")


def fig_15_graph_relationships_common_query_aligned(
    results: dict[str, align.AlignmentResult], output_dir: Path
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    for ax, benchmark in zip(axes, pc.BENCHMARK_ORDER, strict=True):
        r = results[benchmark]
        aligned_n = r.strict_matched_n
        our_n_check = sum(align.outcome_counts(r.aligned_rows, field="ours_recomputed_relationship").values())
        t2t_n_check = sum(align.outcome_counts(r.aligned_rows, field="t2t_recomputed_relationship").values())
        if our_n_check != aligned_n or t2t_n_check != aligned_n:
            raise GraphComparisonError(
                f"{benchmark}: aligned N mismatch between methods (ours={our_n_check}, t2t={t2t_n_check}, "
                f"expected {aligned_n}) -- both bars in a common-query-aligned panel must share the same N"
            )
        our_props = align.outcome_proportions(align.outcome_counts(r.aligned_rows, field="ours_recomputed_relationship"), aligned_n)
        t2t_props = align.outcome_proportions(align.outcome_counts(r.aligned_rows, field="t2t_recomputed_relationship"), aligned_n)
        _plot_relationship_stack(ax, END_TO_END_CATEGORY_ORDER, our_props, t2t_props, _aligned_labels(aligned_n))
        ax.set_title(benchmark, fontsize=11.0)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of common aligned benchmark records")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Common-query-aligned Top-1 graph relationship comparison")
    save_figure(fig, "figure_15_graph_relationships_common_query_aligned", "pairwise", output_dir, formats=FORMATS)


def fig_15b_graph_relationships_common_query_aligned_mapped_only(
    results: dict[str, align.AlignmentResult], output_dir: Path
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    for ax, benchmark in zip(axes, pc.BENCHMARK_ORDER, strict=True):
        r = results[benchmark]

        def _mapped_only_props(field: str, rows: list = r.aligned_rows) -> dict[str, float]:
            mapped_rows = [row for row in rows if getattr(row, field) != NO_TOP1_CATEGORY]
            counts = {rel: sum(1 for row in mapped_rows if getattr(row, field) == rel) for rel in RELATIONSHIP_ORDER}
            n = len(mapped_rows)
            return align.outcome_proportions(counts, n) if n else dict.fromkeys(RELATIONSHIP_ORDER, 0.0)

        our_props = _mapped_only_props("ours_recomputed_relationship")
        t2t_props = _mapped_only_props("t2t_recomputed_relationship")
        our_mapped_n = sum(1 for row in r.aligned_rows if row.ours_recomputed_relationship != NO_TOP1_CATEGORY)
        t2t_mapped_n = sum(1 for row in r.aligned_rows if row.t2t_recomputed_relationship != NO_TOP1_CATEGORY)
        labels = (f"LLM Ontology Mapper\nmapped n={our_mapped_n:,}", f"text2term\nmapped n={t2t_mapped_n:,}")
        _plot_relationship_stack(ax, RELATIONSHIP_ORDER, our_props, t2t_props, labels)
        ax.set_title(benchmark, fontsize=11.0)
        style_axis(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylabel("Share of classifiable Top-1 predictions (aligned subset)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Common-query-aligned Top-1 graph relationship — classifiable predictions only (supplementary)")
    save_figure(fig, "figure_15b_graph_relationships_common_query_aligned_mapped_only", "pairwise", output_dir, formats=FORMATS)


_TRANSITION_CELL_COLORS: dict[str, str] = {
    "both": "#4daf4a",
    "ours_only": "#377eb8",
    "t2t_only": "#984ea3",
    "neither": "#bdbdbd",
}


def fig_15c_exact_match_transitions_vs_text2term(
    transitions: dict[str, align.PairedTransitions], output_dir: Path
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    for ax, benchmark in zip(axes, pc.BENCHMARK_ORDER, strict=True):
        t = transitions[benchmark]
        n = t.total
        grid = [[t.both_exact, t.ours_only_exact], [t.t2t_only_exact, t.neither_exact]]
        colors = [[_TRANSITION_CELL_COLORS["both"], _TRANSITION_CELL_COLORS["ours_only"]],
                  [_TRANSITION_CELL_COLORS["t2t_only"], _TRANSITION_CELL_COLORS["neither"]]]
        for i in range(2):
            for j in range(2):
                count = grid[i][j]
                pct_of_n = count / n if n else 0.0
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=colors[i][j], edgecolor="white", linewidth=1.5))
                ax.text(j + 0.5, 1 - i + 0.5, f"{count:,}\n({pct_of_n * 100:.1f}%)", ha="center", va="center", fontsize=10, color="white")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["text2term exact", "text2term not exact"], fontsize=8.5)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["not exact", "exact"], fontsize=8.5)
        ax.set_ylabel("LLM Ontology Mapper", fontsize=9)
        ax.set_title(f"{benchmark}\n(aligned n={n:,})", fontsize=10.5)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.grid(False)
    fig.suptitle("Paired exact-match (\"Same\") transitions vs. text2term — common-query-aligned")
    save_figure(fig, "figure_15c_exact_match_transitions_vs_text2term", "pairwise", output_dir, formats=FORMATS)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES.md section (appended to the file published_comparison.build_all()
# already wrote -- never overwrites the terminology/denominator/Top-k
# sections it already produced)
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_SECTION_HEADING = "# Top-1 graph-relationship comparison with text2term"


def _build_graph_relationship_markdown_section(
    data: GraphComparisonData, run_dirs: dict[str, Path], *, generate_delta: bool
) -> str:
    lines: list[str] = []
    lines.append(GRAPH_SECTION_HEADING)
    lines.append("")
    lines.append(
        "This section compares **LLM Ontology Mapper**'s Top-1 predictions against the "
        "**original text2term publication**'s own Top-1 graph-relationship distribution -- a "
        "different comparison, with a different text2term source, than the controlled Top-k/MRR "
        "comparison above."
    )
    lines.append("")

    lines.append("### Controlled Top-k baseline vs. graph-relationship baseline")
    lines.append("")
    lines.append(
        "| | Source | What it provides |\n"
        "| --- | --- | --- |\n"
        "| **CONTROLLED TOP-K BASELINE** (Figures 1-12 above) | MetaHarmonizer paper's own controlled "
        "rerun of text2term | Top-1 / Top-3 / Top-5 / MRR only -- no graph-relationship categories |\n"
        "| **GRAPH-RELATIONSHIP BASELINE** (Figures 13-"
        + ("16" if generate_delta else "14")
        + " below) | Original text2term v"
        + EXPECTED_TEXT2TERM_VERSION
        + " publication (Table 1) / text2term-evaluation repository | Same / More Specific / More "
        "General / Sibling / Unrelated |"
    )
    lines.append("")
    lines.append(
        "These are two different text2term executions under two different protocols and are **never** "
        "merged into one source or implied to be the same run. text2term's Top-k figures above and its "
        "graph-relationship figures below should not be read as describing the identical execution."
    )
    lines.append("")

    lines.append("## Category definitions and priority")
    lines.append("")
    lines.append(
        "Reproduced from text2term-evaluation's `compare_ontology_mappings.compare_mappings()` "
        f"(pinned commit `{graph_distance.PINNED_COMMIT}`), and audited line-for-line in "
        "`scenario1_graph_distance.py`'s module docstring:"
    )
    lines.append("")
    lines.append("- **Same** -- predicted mapping equals the benchmark (gold) mapping.")
    lines.append("- **More Specific** -- the predicted term is a subclass/entailed descendant of the "
                  "benchmark term.")
    lines.append("- **More General** -- the predicted term is a superclass/entailed ancestor of the "
                  "benchmark term.")
    lines.append("- **Sibling** -- the predicted and benchmark terms share an asserted direct superclass.")
    lines.append("- **Unrelated** -- none of the above graph relationships apply.")
    lines.append("")
    lines.append(
        "**Priority order** (checked first-match-wins, both here and in the reference "
        "implementation): Same → More Specific → More General → Sibling → Unrelated. This repository's "
        "`scenario1_graph_distance.ALL_RELATIONSHIPS` is verified equal to this exact tuple before any "
        "figure in this section is drawn (`verify_graph_evaluator_compatibility()`); a mismatch would "
        "hard-fail rather than silently plot mismatched semantics as comparable."
    )
    lines.append("")

    lines.append("## Source provenance")
    lines.append("")
    lines.append(
        f"**Our graph evaluator.** `scenario1_graph_distance.py` reimplements (never imports/executes) "
        f"text2term-evaluation's `compare_mappings()` logic, against repository "
        f"`{graph_distance.SOURCE_REPOSITORY}`, file `{graph_distance.SOURCE_FILE}`, pinned commit "
        f"`{graph_distance.PINNED_COMMIT}`, using EFO v{graph_distance.EFO_VERSION} "
        f"(`{graph_distance.EFO_URL}`). Every one of our three official runs' own "
        "`graph_reference_metadata.json` was verified identical and consistent with these constants "
        "before this section was generated."
    )
    lines.append("")
    lines.append(
        f"**text2term's own graph-relationship values.** Sourced from the *original* text2term "
        f"publication's Table 1 (text2term v{EXPECTED_TEXT2TERM_VERSION}, EFO v{EXPECTED_EFO_VERSION}, "
        f"`{EXPECTED_SOURCE_REPOSITORY}`), as supplied to this repository -- stored in "
        "`data/text2term_graph_relationship_baseline.csv` (columns: benchmark, source, "
        "text2term_version, efo_version, n, relationship, count, proportion, publication) and never "
        "hardcoded a second time in any plotting function. The text2term version tag is recorded as "
        "reported; it has not been independently re-verified against a PyPI/source-code artifact in "
        "this repository."
    )
    lines.append("")

    lines.append("## Published original-text2term Table 1 values")
    lines.append("")
    lines.append("| Benchmark | n | Same | More Specific | More General | Sibling | Unrelated |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        t2t = data.text2term[benchmark]
        props = t2t.proportions()
        lines.append(
            f"| {benchmark} | {t2t.n:,} | {props['Same']*100:.1f}% ({t2t.counts['Same']:,}) | "
            f"{props['More Specific']*100:.1f}% ({t2t.counts['More Specific']:,}) | "
            f"{props['More General']*100:.1f}% ({t2t.counts['More General']:,}) | "
            f"{props['Sibling']*100:.1f}% ({t2t.counts['Sibling']:,}) | "
            f"{props['Unrelated']*100:.1f}% ({t2t.counts['Unrelated']:,}) |"
        )
    lines.append("")
    lines.append(
        "Percentages above are computed as count/n at full precision (the same arithmetic used for "
        "every figure and CSV in this section, so bar heights always sum to exactly 100%) and may "
        "therefore differ by up to 0.1 percentage point from the paper's own independently-rounded "
        "percentage for the same count -- the underlying counts are the authoritative published values "
        "and are reproduced exactly."
    )
    lines.append("")

    lines.append("## Two denominator views")
    lines.append("")
    lines.append(
        "Our own graph-distance summary's denominator audit (Part 5) confirmed that "
        "`sum(Same, More Specific, More General, Sibling, Unrelated)` equals `mapped_count` "
        "(the classifiable Top-1 predictions), **not** the full evaluated N -- the gap is exactly "
        "`unmapped_count + error_count`, stored as `Not Applicable` in `graph_distance_summary.csv`. "
        "Two figures make this explicit rather than picking one silently:"
    )
    lines.append("")
    lines.append(
        "- **Figure 13 (mapped-only).** Denominator = our classifiable Top-1 predictions only "
        "(`mapped_count`); text2term's published categories already exhaust its full N. Answers: "
        "\"given a classifiable Top-1 mapping, what was its relationship to gold?\""
    )
    lines.append(
        "- **Figure 14 (end-to-end).** Denominator = the full evaluated N for both methods, with a "
        "sixth category, **No Top-1 prediction** (= unmapped + execution error for us; verified 0 for "
        "text2term because its published counts were confirmed to sum exactly to its own n before being "
        "assigned 0). Answers: \"across the entire evaluated set, what happened at Top-1?\""
    )
    lines.append("")
    lines.append("### Our denominator breakdown (per benchmark)")
    lines.append("")
    lines.append("| Benchmark | Total n | Mapped | Unmapped | Execution error | No Top-1 prediction |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        our = data.our[benchmark]
        lines.append(
            f"| {benchmark} | {our.n:,} | {our.mapped_count:,} | {our.unmapped_count:,} | "
            f"{our.error_count:,} | {our.no_top1_count:,} |"
        )
    lines.append("")

    lines.append("## Comparability limitations")
    lines.append("")
    lines.append(
        "| Benchmark | Our n | text2term (original) n | Denominators match? |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {benchmark} | {data.our[benchmark].n:,} | {data.text2term[benchmark].n:,} | "
            f"{'Yes' if data.our[benchmark].n == data.text2term[benchmark].n else 'No'} |"
            for benchmark in pc.BENCHMARK_ORDER
        )
    )
    lines.append("")
    lines.append(
        f"- **UKBB-EFO denominator mismatch.** Our run evaluates n={data.our['UKBB-EFO'].n:,}; the "
        f"original text2term Table 1 evaluates n={data.text2term['UKBB-EFO'].n:,}. These are NOT "
        "presented as an exact head-to-head comparison."
    )
    lines.append(
        f"- **OLS-EFO denominator mismatch -- THREE different OLS numbers appear across this whole "
        f"figure suite, and they must not be confused:** our controlled-comparison/graph-distance N is "
        f"n={data.our['OLS-EFO (full)'].n:,} unique queries (Figures 1-14); the MetaHarmonizer-"
        f"controlled text2term rerun used for the Top-k comparison (Figures 1, 5, 8, 12) reports "
        f"n=7,504 mapping pairs; and the *original* text2term publication's own OLS-EFO graph-"
        f"relationship evaluation used n={data.text2term['OLS-EFO (full)'].n:,} rows -- a materially "
        "different, larger set than either of the other two. All three are legitimate numbers from "
        "different sources/protocols, never interchangeable."
    )
    lines.append(
        "- **Biomappings-EFO appears most directly comparable**: our n and the original text2term n are "
        f"both {data.our['Biomappings-EFO'].n:,}, though this equality of N alone is not proof of "
        "identical row identity (see common-query alignment below)."
    )
    lines.append(
        "- **Original text2term protocol was single-gold only.** The original text2term paper's "
        "comparison was limited to queries with exactly one benchmark mapping. Our own gold-count "
        "audit (`data/our_multi_gold_audit.csv`):"
    )
    for benchmark in pc.BENCHMARK_ORDER:
        distribution = load_gold_count_distribution(run_dirs[benchmark])
        parts = ", ".join(f"{count:,} queries with {gold_count} gold" for gold_count, count in sorted(distribution.items()))
        lines.append(f"  - {benchmark}: {parts}")
    lines.append(
        "  OLS-EFO (full) includes multi-gold queries (113 with 2 acceptable golds, 7 with 3), so our "
        "full-run OLS-EFO graph distribution is not perfectly protocol-identical to the original "
        "text2term Table 1 distribution even where n happened to align; UKBB-EFO and Biomappings-EFO "
        "are 100% single-gold, matching the original text2term protocol on this dimension."
    )
    lines.append(
        "- **The original text2term run used here is NOT the MetaHarmonizer-controlled t2t rerun** "
        "used everywhere else in this suite -- see the table at the top of this section."
    )
    lines.append("")

    lines.append("## Common-query alignment audit")
    lines.append("")
    lines.append(
        "A stronger, common-query-aligned comparison (Figure 15) was considered: intersecting our "
        "evaluated query/gold records with the raw per-query outputs published in the "
        "`rsgoncalves/text2term-evaluation` repository (`output/*_t2t_mappings.csv`, "
        "`output/*_mappings.tsv`, `output/*_results.tsv`), then re-evaluating both methods on the "
        "identical matched rows with the same graph evaluator."
    )
    lines.append("")
    lines.append(
        "**This alignment was NOT attempted and Figure 15 was NOT generated.** The raw per-query "
        "output files are not present in this repository or vendored under `data/text2term_evaluation/` "
        "(only the EFO edge/entailed-edge reference tables used by our own graph evaluator are vendored "
        "there). Fetching them now would require a live network call to GitHub, which is out of scope "
        "for this analysis-only, zero-network plotting task, and per this suite's reproducibility "
        "policy any such raw files would first need to be explicitly vendored under a reproducible data "
        "directory with their source URL, repository commit, and SHA256 recorded -- not fetched "
        "silently as a side effect of plotting. No fuzzy string matching, row-identity inference, or "
        "ambiguous-duplicate resolution was performed or would be acceptable as a substitute. Figures "
        "13 and 14 above therefore remain descriptive published-protocol comparisons with the explicit "
        "denominator caveats documented in this section, not a common-query-aligned comparison."
    )
    lines.append("")

    figure_files = [
        (
            "13", "Top-1 graph relationship — classifiable predictions only",
            ["pairwise/figure_13_graph_relationships_mapped_only.png",
             "pairwise/figure_13_graph_relationships_mapped_only.svg"],
            "Given a classifiable Top-1 mapping, what was its relationship to the benchmark gold?",
            "Five mutually-exclusive graph-relationship categories (Same, More Specific, More General, "
            "Sibling, Unrelated), each method normalized to its OWN classifiable-predictions "
            "denominator (see 'Two denominator views' above).",
            "data/graph_relationship_mapped_only.csv",
        ),
        (
            "14", "Top-1 graph relationship — end-to-end",
            ["pairwise/figure_14_graph_relationships_end_to_end.png",
             "pairwise/figure_14_graph_relationships_end_to_end.svg"],
            "Across the entire evaluated set, what happened at Top-1 (including no prediction at all)?",
            "Six mutually-exclusive categories -- the five graph relationships plus 'No Top-1 "
            "prediction' -- both methods normalized to their own full evaluated N.",
            "data/graph_relationship_end_to_end.csv",
        ),
    ]
    for number, title, files, question, categories, source_csv in figure_files:
        lines.append(f"### Figure {number} — {title}")
        lines.append("")
        lines.append("**Files**")
        for f in files:
            lines.append(f"- {f}")
        lines.append("")
        lines.append(f"**Question.** {question}")
        lines.append("")
        lines.append("**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).")
        lines.append("**Methods.** Our method, text2term (original publication) only -- MetaHarmonizer (OM) is "
                      "NOT included in this figure (Part 16): the published OM table provides cumulative "
                      "Top-k metrics only, with no equivalent five-category graph-relationship breakdown.")
        lines.append(f"**Categories.** {categories}")
        lines.append("**Colors.** Category colors, not method colors -- the same six hex values are reused "
                      "across Figures 13/14 (and 16, if generated) for the same category.")
        lines.append("**Denominators.** n is shown directly under each bar's label; see the denominator "
                      "table and caveats above.")
        lines.append("**Caveats.** Descriptive published-protocol comparison, not common-query aligned (see "
                      "above); mismatched denominators for UKBB-EFO and OLS-EFO (full).")
        lines.append(f"**Source data.** `{source_csv}`")
        lines.append("")

    if generate_delta:
        lines.append("### Figure 16 — Δ graph relationship vs. text2term (descriptive)")
        lines.append("")
        lines.append("**Files**")
        lines.append("- pairwise/figure_16_graph_relationship_delta_vs_text2term.png")
        lines.append("- pairwise/figure_16_graph_relationship_delta_vs_text2term.svg")
        lines.append("")
        lines.append(
            "**Question.** By how many percentage points does each graph-relationship category differ "
            "between our method (mapped-only view) and the published text2term values?"
        )
        lines.append(
            "**Labeling.** Explicitly titled and documented as a **descriptive published-protocol "
            "difference** -- the two sides use different, mismatched denominators with no common-query "
            "alignment (see above), so this is never framed as, or implying, a statistically validated "
            "or common-query comparison. No hypothesis test has been run."
        )
        lines.append("**Source data.** `data/graph_relationship_delta_vs_text2term.csv`")
        lines.append("")
    else:
        lines.append(
            "**Figure 16 (optional delta view) was not generated** for this run -- see build parameters."
        )
        lines.append("")

    lines.append("## Which comparison is strongest for publication use")
    lines.append("")
    lines.append(
        "**Figures 13/14 are a published-protocol descriptive comparison, not a common-query-aligned "
        "one.** Biomappings-EFO's matching N (795 vs. 795) makes it the most directly comparable of the "
        "three, but even there row-level identity was not verified. UKBB-EFO and OLS-EFO (full) have "
        "materially different denominators from the original text2term evaluation and must be presented "
        "with that caveat. No common-query-aligned analysis (Figure 15) exists in this repository -- if "
        "one is produced in the future by explicitly vendoring and documenting the "
        "`rsgoncalves/text2term-evaluation` raw output files (Part 19), it would be the stronger, "
        "preferred comparison and should supersede Figures 13/14/16 for any claim of near-identical "
        "row-level comparison."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


ALIGNED_SECTION_HEADING = "# Common-query-aligned comparison with original text2term"


def recommend_figure_15_primary(
    results: dict[str, align.AlignmentResult],
) -> tuple[bool, str]:
    """Part 26: promote Figure 15 to PRIMARY only if alignment quality is
    strong/good for every benchmark AND graph-classification reproduction
    (t2t stored-vs-recomputed agreement) is effectively exact everywhere.
    Never promoted automatically just because some rows matched."""
    quality_issues = []
    reclass_issues = []
    for benchmark in pc.BENCHMARK_ORDER:
        r = results[benchmark]
        quality = align.alignment_quality_label(r.match_rate_ours)
        if quality not in ("STRONG", "GOOD"):
            quality_issues.append(f"{benchmark}: alignment quality={quality} (match_rate_ours={r.match_rate_ours:.1%})")
        matched_n = len(r.aligned_rows)
        agreement_n = sum(
            1 for row in r.aligned_rows if row.t2t_original_published_classification == row.t2t_recomputed_relationship
        )
        agreement_rate = agreement_n / matched_n if matched_n else 0.0
        if agreement_rate < 0.99:
            reclass_issues.append(f"{benchmark}: t2t stored-vs-recomputed agreement={agreement_rate:.1%}")

    if not quality_issues and not reclass_issues:
        return True, (
            "Every benchmark reached STRONG or GOOD alignment quality (>=90% of our eligible single-gold "
            "records matched deterministically) and text2term's own stored Classification agreed with our "
            "independently recomputed classification on effectively all matched rows (>=99%) -- both "
            "conditions required by this suite's promotion policy are satisfied."
        )
    reasons = "; ".join(quality_issues + reclass_issues)
    return False, f"Figure 15 was NOT promoted to primary because: {reasons}."


def _build_aligned_markdown_section(
    results: dict[str, align.AlignmentResult],
    transitions: dict[str, align.PairedTransitions],
    mcnemar: dict[str, align.McNemarResult],
    *,
    generate_15c: bool,
) -> str:
    lines: list[str] = []
    lines.append(ALIGNED_SECTION_HEADING)
    lines.append("")
    lines.append(
        "This section supersedes the descriptive comparison above with a STRICT common-query-aligned "
        "comparison: the same benchmark records, evaluated by both LLM Ontology Mapper and the original "
        "text2term evaluation, classified with the identical EFO v3.62.0 graph evaluator."
    )
    lines.append("")

    lines.append("## Source and provenance")
    lines.append("")
    lines.append(
        f"Vendored via `scripts/fetch_text2term_evaluation_outputs.py` from repository "
        f"`{align.graph_distance.SOURCE_REPOSITORY}`, pinned commit `{align.graph_distance.PINNED_COMMIT}` -- "
        "the SAME commit already used as this repository's graph-evaluator reference (audited: that commit's "
        "git tree contains all nine required `output/*.{tsv,csv}` files, so no second commit was ever "
        "consulted). Files: `UKBB-EFO_results.tsv`, `Biomappings_results.tsv`, `OLS-EFO_results.tsv` (the "
        "row-level per-query comparison outputs used for alignment), plus the corresponding `_mappings.tsv` "
        "and `_t2t_mappings.csv` files fetched for completeness. Every file's SHA256 was verified against a "
        "pinned expected value before being accepted; the exact hashes and fetch timestamp are recorded in "
        "`data/text2term_evaluation/original_outputs/provenance.json`. The fetch step is the ONLY part of "
        "this whole workflow that touches the network -- alignment and plotting read only these already-"
        "vendored files."
    )
    lines.append("")

    lines.append("## Alignment identity and normalization")
    lines.append("")
    lines.append(
        "**Alignment key**: `(normalized(source_query), normalized(benchmark_gold_curie))`. This was chosen "
        "after auditing candidate keys: the upstream `Source Term ID` column is 100% unique within each "
        "vendored file but is the ORIGINAL benchmark source's own row identifier (never persisted in our own "
        "records), so it cannot serve as a cross-dataset join key. `(source_term, gold)` pair identity, by "
        "contrast, is directly present and near-uniquely populated on both sides."
    )
    lines.append("")
    lines.append(
        "**Normalization (conservative, documented, no fuzzy matching)**: Unicode NFC normalization + "
        "whitespace trim on source text; the same plus canonicalizing ONE well-known CURIE-prefix alias "
        "(`Orphanet:` <-> `ORDO:`, both naming the Orphanet Rare Disease Ontology identifier space) on gold "
        "codes, uppercasing only the prefix (local codes are preserved verbatim, never case-folded). This "
        "alias was not a guess: all 27 UKBB-EFO rows whose only difference from a text2term row was this "
        "exact prefix spelling resolved cleanly once aliased, taking UKBB-EFO's match rate from 97.0% to "
        "100%. No edit distance, embeddings, lowercasing of arbitrary identifiers, or label-similarity "
        "matching was used anywhere in this alignment."
    )
    lines.append("")
    lines.append(
        "**An incidental but important correctness finding**: our own vendored `efo_edges.tsv` indexes "
        "Orphanet Rare Disease Ontology nodes under the `ORDO:` prefix, not `Orphanet:`. Our OWN persisted "
        "Scenario 1 UKBB-EFO run stores gold codes as `Orphanet:<id>` (unaliased) for 27 rows, meaning "
        "`classify()` against that literal gold code cannot find the node and silently falls through to the "
        "absent-node \"Unrelated\" result for those rows in our *already-persisted* `graph_relationship` "
        "column -- this is a pre-existing data-quality quirk in completed Scenario 1 output, NOT introduced "
        "or fixed here (this task does not modify completed run outputs). This alignment module always "
        "canonicalizes the gold prefix before calling `classify()` for both methods, so the aligned figures "
        "in this section are unaffected; the `our_original_graph_relationship` column in "
        "`text2term_common_query_alignment_rows.csv` preserves the original (potentially prefix-degraded) "
        "persisted value for comparison against `ours_recomputed_relationship`."
    )
    lines.append("")

    lines.append("## Duplicate / ambiguity policy")
    lines.append("")
    lines.append(
        "Exact-duplicate upstream rows (identical source term, gold, t2t prediction, AND classification) are "
        "safely collapsed to one representative row. Duplicate identity keys whose rows DISAGREE on any of "
        "those fields are marked ambiguous and excluded from the strict aligned analysis entirely (never "
        "resolved by picking the first row) -- see `text2term_alignment_unmatched.csv` for every excluded "
        "record and its reason."
    )
    lines.append("")

    lines.append("## OLS-EFO single-gold restriction")
    lines.append("")
    lines.append(
        "The original text2term protocol evaluated each benchmark record against exactly one benchmark "
        "mapping. Our OLS-EFO Scenario 1 run supports multiple acceptable golds per query (7,257 "
        "single-gold, 113 with 2 golds, 7 with 3), which would silently advantage our method if multi-gold "
        "queries were included in a comparison against text2term's single-gold protocol. The PRIMARY strict "
        "OLS-EFO alignment is therefore restricted to our 7,257 single-gold queries only; multi-gold queries "
        "are excluded from this alignment entirely (not scored, not credited, not penalized). UKBB-EFO and "
        "Biomappings-EFO required no such restriction -- both are verified 100% single-gold by their own "
        "`original_mapping_pair_count` field in `unique_queries.csv` (the same field/definition "
        "`dataset_validation.json`'s `gold_count_distribution` uses; naively counting `|` characters in the "
        "`gold_codes` string is NOT equivalent -- a handful of UKBB-EFO rows carry a single canonical gold "
        "whose own composite source-benchmark label happens to already contain literal `|` text)."
    )
    lines.append("")

    lines.append("## Alignment quality")
    lines.append("")
    lines.append("| Benchmark | Our N | t2t N | Our single-gold N | Candidate matches | Strict matched N | Ambiguous | Gold mismatch | Match rate (ours) | Quality |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        r = results[benchmark]
        quality = align.alignment_quality_label(r.match_rate_ours)
        lines.append(
            f"| {benchmark} | {r.ours_total_n:,} | {r.t2t_total_n:,} | {r.ours_single_gold_n:,} | "
            f"{r.candidate_exact_matches:,} | {r.strict_matched_n:,} | {r.ambiguous_n} | {r.gold_mismatch_n} | "
            f"{r.match_rate_ours:.1%} | **{quality}** |"
        )
    lines.append("")
    lines.append(
        "Quality thresholds (reporting heuristics, not statistical rules; never tuned to hit a target): "
        ">=95% STRONG, 90-95% GOOD (disclose exclusions), 75-90% PARTIAL (supplementary only), <75% "
        "insufficient for a primary claim."
    )
    lines.append("")

    lines.append("## Table-1 reproducibility check (Part 4)")
    lines.append("")
    lines.append(
        "Before aligning, the published Table-1 aggregate was recomputed directly from the vendored raw "
        "per-row `Classification` column (never trusted as primary for this analysis -- see Part 4) and "
        "compared against the earlier-recorded published baseline used by Figures 13/14 as a reproducibility "
        "check:"
    )
    lines.append("")
    for benchmark in pc.BENCHMARK_ORDER:
        check = results[benchmark].reproducibility
        if check.agreement:
            lines.append(f"- **{benchmark}**: exact agreement on all five categories.")
        else:
            deltas = ", ".join(f"{rel}: recomputed={rec} vs. published={pub}" for rel, (rec, pub) in check.mismatches.items())
            lines.append(f"- **{benchmark}**: disagreement found and investigated -- {deltas}.")
    lines.append("")
    lines.append(
        "These small discrepancies were investigated (not concealed): the raw per-row data is internally "
        "self-consistent (every benchmark's per-row tally sums exactly to its own n) and matches Biomappings-"
        "EFO's published values exactly; UKBB-EFO differs by 2/899 rows between Same and More Specific, and "
        "OLS-EFO's More Specific/More General counts appear swapped relative to the earlier-recorded "
        "published values (91 and 55 exchanged) -- most consistent with a transcription/column-order slip "
        "when those published values were originally recorded, not a wrong file or wrong graph version. Per "
        "Part 4, this analysis treats the raw vendored per-row data as authoritative rather than blocking on "
        "this small, explained discrepancy; full counts are in `data/text2term_table1_reproducibility_check.csv`."
    )
    lines.append("")

    lines.append("## text2term stored-vs-recomputed classification agreement (Part 15)")
    lines.append("")
    lines.append(
        "For every STRICT matched row, text2term's own stored `Classification` was compared against the "
        "classification our reused `scenario1_graph_distance.classify()` computes for text2term's Top-1 "
        "prediction against the same gold code:"
    )
    lines.append("")
    lines.append("| Benchmark | Matched N | Agreement N | Disagreement N | Agreement rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        r = results[benchmark]
        matched_n = len(r.aligned_rows)
        agreement_n = sum(1 for row in r.aligned_rows if row.t2t_original_published_classification == row.t2t_recomputed_relationship)
        rate = agreement_n / matched_n if matched_n else 0.0
        lines.append(f"| {benchmark} | {matched_n:,} | {agreement_n:,} | {matched_n - agreement_n} | {rate:.2%} |")
    lines.append("")
    lines.append(
        "High agreement here is strong evidence that this repository's from-scratch reimplementation of "
        "`compare_mappings()` faithfully reproduces the original graph-comparison semantics on real data, "
        "not just on the priority-order/EFO-version metadata check."
    )
    lines.append("")

    lines.append("## Aligned graph-relationship distributions")
    lines.append("")
    lines.append("| Benchmark | Method | Same | More Specific | More General | Sibling | Unrelated | No Top-1 prediction |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        r = results[benchmark]
        n = len(r.aligned_rows)
        for label, field in (("LLM Ontology Mapper", "ours_recomputed_relationship"), ("text2term", "t2t_recomputed_relationship")):
            counts = align.outcome_counts(r.aligned_rows, field=field)
            props = align.outcome_proportions(counts, n)
            lines.append(
                f"| {benchmark} | {label} | {props['Same']:.1%} | {props['More Specific']:.1%} | "
                f"{props['More General']:.1%} | {props['Sibling']:.1%} | {props['Unrelated']:.1%} | "
                f"{props[NO_TOP1_CATEGORY]:.1%} |"
            )
    lines.append("")

    lines.append("## Paired exact-match transitions and McNemar's test")
    lines.append("")
    lines.append(
        "Because aligned rows are paired (both methods scored on the identical record), row-level rescue "
        "behavior is quantified directly rather than by subtracting aggregate percentages:"
    )
    lines.append("")
    lines.append("| Benchmark | Both exact | Ours only | text2term only | Neither | Aligned N |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        t = transitions[benchmark]
        lines.append(f"| {benchmark} | {t.both_exact:,} | {t.ours_only_exact:,} | {t.t2t_only_exact:,} | {t.neither_exact:,} | {t.total:,} |")
    lines.append("")
    lines.append(
        "**McNemar's exact test** (binomial, on the discordant pairs -- appropriate specifically because "
        "Top-1-exact correctness is paired on identical records here):"
    )
    lines.append("")
    lines.append("| Benchmark | Ours-only correct | text2term-only correct | Discordant N | p-value |")
    lines.append("| --- | --- | --- | --- | --- |")
    for benchmark in pc.BENCHMARK_ORDER:
        m = mcnemar[benchmark]
        p_display = "1.0 (no discordant pairs)" if m.discordant_n == 0 else f"{m.p_value:.3g}"
        lines.append(f"| {benchmark} | {m.ours_only_correct:,} | {m.t2t_only_correct:,} | {m.discordant_n:,} | {p_display} |")
    lines.append("")
    lines.append(
        "A small p-value indicates the discordant pairs are asymmetric beyond chance -- it does NOT by "
        "itself establish which method is better in any absolute sense, only that the two methods' Top-1-"
        "exact outcomes disagree asymmetrically on this aligned set."
    )
    lines.append("")
    lines.append(
        "**Paired bootstrap CI on the Same-proportion difference**: not implemented in this pass (Part 22 "
        "marks it explicitly optional) -- flagged here rather than silently omitted."
    )
    lines.append("")

    lines.append("### Figure 15 — Common-query-aligned Top-1 graph relationship comparison (PRIMARY)")
    lines.append("")
    lines.append("**Files**")
    lines.append("- pairwise/figure_15_graph_relationships_common_query_aligned.png")
    lines.append("- pairwise/figure_15_graph_relationships_common_query_aligned.svg")
    lines.append("")
    lines.append(
        "**Question.** On the SAME benchmark records, evaluated by both methods, what is the Top-1 "
        "graph-relationship composition, end-to-end (including no-prediction)?"
    )
    lines.append(
        "**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (strict single-gold common subset) -- three panels."
    )
    lines.append("**Methods.** LLM Ontology Mapper, text2term -- two 100%-stacked bars per panel, both bars sharing the IDENTICAL aligned N (enforced by a hard assertion in the plotting code).")
    lines.append("**Categories.** Same, More Specific, More General, Sibling, Unrelated, No Top-1 prediction -- same category colors as Figures 13/14.")
    lines.append("**Denominators.** Aligned n shown under each bar; both bars in a panel always match by construction.")
    lines.append("**Caveats.** OLS-EFO panel is restricted to the single-gold common subset (see above) -- not the full 7,377-query OLS-EFO run.")
    lines.append("**Source data.** `data/text2term_common_query_alignment_rows.csv`, `data/text2term_common_query_alignment_summary.csv`")
    lines.append("")

    lines.append("### Figure 15b — Common-query-aligned, classifiable predictions only (supplementary)")
    lines.append("")
    lines.append("**Files**")
    lines.append("- pairwise/figure_15b_graph_relationships_common_query_aligned_mapped_only.png")
    lines.append("- pairwise/figure_15b_graph_relationships_common_query_aligned_mapped_only.svg")
    lines.append("")
    lines.append(
        "**Question.** Given that a method emitted a classifiable Top-1 mapping on the aligned subset, how "
        "related was it to gold? Each method's \"No Top-1 prediction\" rows are excluded and the remaining "
        "five categories independently renormalized to 100% per method (so the two bars in a panel may have "
        "different denominators here, shown under each bar) -- supplementary to Figure 15, which remains "
        "primary because it does not normalize away abstention."
    )
    lines.append("**Source data.** `data/text2term_common_query_alignment_rows.csv`")
    lines.append("")

    if generate_15c:
        lines.append("### Figure 15c — Paired exact-match transitions (supplementary)")
        lines.append("")
        lines.append("**Files**")
        lines.append("- pairwise/figure_15c_exact_match_transitions_vs_text2term.png")
        lines.append("- pairwise/figure_15c_exact_match_transitions_vs_text2term.svg")
        lines.append("")
        lines.append(
            "**Question.** How many aligned records did each method get exactly right (\"Same\"), broken "
            "down into both-correct / ours-only / text2term-only / neither? Three 2x2 grids (UKBB-EFO, "
            "Biomappings-EFO, OLS-EFO single-gold subset), each cell annotated with count and percentage of "
            "that panel's aligned N."
        )
        lines.append("**Source data.** `data/text2term_aligned_exact_match_transitions.csv`")
        lines.append("")

    promoted, rationale = recommend_figure_15_primary(results)
    lines.append("## Recommendation")
    lines.append("")
    if promoted:
        lines.append(
            f"**Figure 15 is promoted to the PRIMARY graph-relationship comparison for manuscript use.** {rationale}"
        )
    else:
        lines.append(f"**Figure 15 is NOT promoted to primary.** {rationale} Figures 13/14 remain the primary descriptive comparison.")
    lines.append("")

    return "\n".join(lines) + "\n"


def append_graph_relationship_figures_md_section(
    data: GraphComparisonData, run_dirs: dict[str, Path], figures_md_path: Path, *, generate_delta: bool,
    aligned_markdown: str = "",
) -> None:
    """Idempotently append this section (and, if given, the aligned-
    comparison section markdown) to the FIGURES.md that
    published_comparison.build_all() already wrote (Part 15/25). Never
    overwrites the terminology/denominator/Top-k sections above this one;
    re-running this function strips a previous copy of both sections first
    (they always live after GRAPH_SECTION_HEADING) so repeated builds never
    duplicate them."""
    if not figures_md_path.exists():
        raise GraphComparisonError(
            f"{figures_md_path} does not exist -- run published_comparison.build_all() first so the base "
            "FIGURES.md (terminology/denominators/Top-k figures) exists before appending this section"
        )
    existing = figures_md_path.read_text(encoding="utf-8")
    base = existing.split(GRAPH_SECTION_HEADING)[0].rstrip("\n") + "\n\n"
    section = _build_graph_relationship_markdown_section(data, run_dirs, generate_delta=generate_delta)
    figures_md_path.write_text(base + section + aligned_markdown, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphBuildResult:
    output_dir: Path
    data: GraphComparisonData
    alignment_results: dict[str, align.AlignmentResult] | None = None


def build_all(
    *,
    ols_dir: Path,
    ukbb_dir: Path,
    biomappings_dir: Path,
    text2term_baseline_path: Path,
    output_dir: Path,
    figures_md_path: Path,
    generate_delta: bool = True,
    text2term_data_dir: Path | None = None,
    generate_15c: bool = True,
) -> GraphBuildResult:
    data = load_all_graph_data(
        ols_dir=ols_dir, ukbb_dir=ukbb_dir, biomappings_dir=biomappings_dir, baseline_path=text2term_baseline_path
    )
    run_dirs = {"UKBB-EFO": ukbb_dir, "Biomappings-EFO": biomappings_dir, "OLS-EFO (full)": ols_dir}

    data_dir = output_dir / "data"
    write_graph_relationship_mapped_only_csv(data, data_dir / "graph_relationship_mapped_only.csv")
    write_graph_relationship_end_to_end_csv(data, data_dir / "graph_relationship_end_to_end.csv")
    write_denominator_comparison_csv(data, data_dir / "graph_relationship_denominator_comparison.csv")
    write_multi_gold_audit_csv(run_dirs, data_dir / "our_multi_gold_audit.csv")

    fig_13_graph_relationships_mapped_only(data, output_dir)
    fig_14_graph_relationships_end_to_end(data, output_dir)

    if generate_delta:
        delta_rows = compute_descriptive_deltas_mapped_only(data)
        write_graph_relationship_delta_csv(delta_rows, data_dir / "graph_relationship_delta_vs_text2term.csv")
        fig_16_graph_relationship_delta_vs_text2term(delta_rows, output_dir)

    alignment_results: dict[str, align.AlignmentResult] | None = None
    aligned_markdown = ""
    if text2term_data_dir is not None:
        alignment_results = {}
        for benchmark in pc.BENCHMARK_ORDER:
            t2t_results_path = text2term_data_dir / align.T2T_RESULTS_FILENAME[benchmark]
            published_counts = data.text2term[benchmark].counts
            alignment_results[benchmark] = align.align_benchmark(benchmark, run_dirs[benchmark], t2t_results_path, published_counts)

        align.write_alignment_summary_csv(alignment_results, data_dir / "text2term_common_query_alignment_summary.csv")
        align.write_aligned_rows_csv(alignment_results, data_dir / "text2term_common_query_alignment_rows.csv")
        align.write_unmatched_csv(alignment_results, data_dir / "text2term_alignment_unmatched.csv")
        align.write_reclassification_audit_csv(alignment_results, data_dir / "text2term_reclassification_audit.csv")
        align.write_table1_reproducibility_csv(alignment_results, data_dir / "text2term_table1_reproducibility_check.csv")

        transitions = {b: align.compute_paired_transitions(alignment_results[b].aligned_rows) for b in pc.BENCHMARK_ORDER}
        align.write_transitions_csv(transitions, data_dir / "text2term_aligned_exact_match_transitions.csv")

        mcnemar = {b: align.compute_mcnemar(b, transitions[b]) for b in pc.BENCHMARK_ORDER}
        align.write_mcnemar_csv(mcnemar, data_dir / "text2term_aligned_mcnemar.csv")

        fig_15_graph_relationships_common_query_aligned(alignment_results, output_dir)
        fig_15b_graph_relationships_common_query_aligned_mapped_only(alignment_results, output_dir)
        if generate_15c:
            fig_15c_exact_match_transitions_vs_text2term(transitions, output_dir)

        aligned_markdown = _build_aligned_markdown_section(alignment_results, transitions, mcnemar, generate_15c=generate_15c)

    append_graph_relationship_figures_md_section(
        data, run_dirs, figures_md_path, generate_delta=generate_delta, aligned_markdown=aligned_markdown
    )

    return GraphBuildResult(output_dir=output_dir, data=data, alignment_results=alignment_results)
