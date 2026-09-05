"""Figure-level tests for the common-query-aligned comparison
(figure_15/15b/15c) wired into graph_relationship_comparison.build_all().

Filesystem-only (tmp_path), synthetic fixtures reused from
test_scenario1_graph_relationship_comparison.py's helpers plus a small
synthetic vendored text2term results.tsv per benchmark. No network, no
mapper, no LLM calls.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.figures import graph_relationship_comparison as gc
from llm_ontology_mapper.benchmarking.figures import published_comparison as pc

from .test_scenario1_graph_relationship_comparison import (
    _OUR_GRAPH,
    _make_all_graph_run_dirs,
    _write_text2term_graph_baseline_csv,
)

pytestmark = pytest.mark.unit

_T2T_FIELDS = ["Source Term ID", "Source Term", "t2t.Mapping", "t2t.MappingLabel",
               "Benchmark.Mapping", "Benchmark.MappingLabel", "Classification"]

# Matches the synthetic our-side counts in test_scenario1_graph_relationship_comparison
# (_OUR_GRAPH) so every row deterministically joins with n=our n for a clean,
# fully-aligned synthetic fixture.
_SOURCE_TERMS = {
    "UKBB-EFO": [f"ukbb_term_{i}" for i in range(_OUR_GRAPH["UKBB-EFO"]["n"])],
    "Biomappings-EFO": [f"bio_term_{i}" for i in range(_OUR_GRAPH["Biomappings-EFO"]["n"])],
    "OLS-EFO (full)": [f"ols_term_{i}" for i in range(_OUR_GRAPH["OLS-EFO (full)"]["n"])],
}


def _make_t2t_results_tsv(path: Path, benchmark: str) -> None:
    """A synthetic results.tsv whose Source Term / Benchmark.Mapping exactly
    matches this benchmark's synthetic unique_queries.csv / mapping_pair_expanded_predictions.csv
    fixture (see test_scenario1_graph_relationship_comparison._make_graph_run_dir),
    so every row aligns 1:1 with n = the fixture's own n."""
    spec = _OUR_GRAPH[benchmark]
    n = spec["n"]
    mapped = spec["mapped"]
    terms = _SOURCE_TERMS[benchmark]
    rows = []
    # Build n rows: `mapped` of them "Same" (t2t predicts the gold exactly),
    # remaining (unmapped+error) rows still need a row on the t2t side with
    # SOME prediction (t2t itself always predicts something in the real
    # vendored data) so the aligned figure has a full n-row aligned set.
    for i in range(n):
        term = terms[i]
        gold = f"EFO:SYN{i:05d}"
        is_same = i < mapped
        prediction = gold if is_same else f"EFO:OTHER{i:05d}"
        classification = "Same" if is_same else "Unrelated"
        rows.append({
            "Source Term ID": f"id{i}", "Source Term": term, "t2t.Mapping": prediction,
            "t2t.MappingLabel": "", "Benchmark.Mapping": gold, "Benchmark.MappingLabel": "",
            "Classification": classification,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_T2T_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _patch_our_run_dir_for_alignment(run_dir: Path, benchmark: str) -> None:
    """The graph_relationship_comparison fixtures build a run dir with
    graph_distance_summary.csv/execution_diagnostics.csv (Figures 13/14) but
    not the unique_queries.csv/mapping_pair_expanded_predictions.csv that
    the alignment path also needs -- add those here, matching the same
    synthetic per-row structure as _make_t2t_results_tsv above so every row
    exact-matches."""
    spec = _OUR_GRAPH[benchmark]
    n = spec["n"]
    mapped = spec["mapped"]
    terms = _SOURCE_TERMS[benchmark]

    uq_fields = ["query_id", "source_query", "gold_codes", "gold_labels", "original_row_indices", "original_mapping_pair_count"]
    exp_fields = ["query_id", "source_query", "gold_code", "gold_label", "raw_row_index", "rank_1_code",
                  "rank_2_code", "rank_3_code", "rank_4_code", "rank_5_code", "first_gold_rank", "top1_hit",
                  "top3_hit", "top5_hit", "reciprocal_rank", "status"]
    pred_fields = ["query_id", "graph_relationship"]

    uq_rows, exp_rows, pred_rows = [], [], []
    for i in range(n):
        term = terms[i]
        gold = f"EFO:SYN{i:05d}"
        is_mapped = i < mapped
        our_pred = gold if is_mapped else ""
        status = "mapped" if is_mapped else "unmapped"
        relationship = "Same" if is_mapped else "Not Applicable"
        uq_rows.append({"query_id": str(i), "source_query": term, "gold_codes": gold, "gold_labels": "",
                         "original_row_indices": str(i), "original_mapping_pair_count": "1"})
        exp_rows.append({"query_id": str(i), "source_query": term, "gold_code": gold, "gold_label": "",
                          "raw_row_index": str(i), "rank_1_code": our_pred, "rank_2_code": "", "rank_3_code": "",
                          "rank_4_code": "", "rank_5_code": "", "first_gold_rank": "1" if is_mapped else "",
                          "top1_hit": str(is_mapped), "top3_hit": str(is_mapped), "top5_hit": str(is_mapped),
                          "reciprocal_rank": "1.0" if is_mapped else "0.0", "status": status})
        pred_rows.append({"query_id": str(i), "graph_relationship": relationship})

    for fname, fields, rows in (
        ("unique_queries.csv", uq_fields, uq_rows),
        ("mapping_pair_expanded_predictions.csv", exp_fields, exp_rows),
        ("predictions.csv", pred_fields, pred_rows),
    ):
        with (run_dir / fname).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _full_aligned_setup(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    run_dirs = _make_all_graph_run_dirs(
        tmp_path, **{"OLS-EFO (full)": {"gold_count_distribution": {1: _OUR_GRAPH["OLS-EFO (full)"]["n"]}}}
    )
    for benchmark, run_dir in run_dirs.items():
        _patch_our_run_dir_for_alignment(run_dir, benchmark)

    baseline_csv = tmp_path / "text2term_graph_baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)

    t2t_data_dir = tmp_path / "t2t_data"
    for benchmark, filename in gc.align.T2T_RESULTS_FILENAME.items():
        _make_t2t_results_tsv(t2t_data_dir / filename, benchmark)

    return run_dirs, baseline_csv, t2t_data_dir


def test_aligned_figures_created_png_svg_no_pdf(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    result = gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        text2term_data_dir=t2t_data_dir,
    )
    assert result.alignment_results is not None

    for name in (
        "figure_15_graph_relationships_common_query_aligned",
        "figure_15b_graph_relationships_common_query_aligned_mapped_only",
        "figure_15c_exact_match_transitions_vs_text2term",
    ):
        png = output_dir / "pairwise" / f"{name}.png"
        svg = output_dir / "pairwise" / f"{name}.svg"
        assert png.exists() and png.stat().st_size > 0, f"missing {png}"
        assert svg.exists() and svg.stat().st_size > 0, f"missing {svg}"

    assert list(output_dir.rglob("*.pdf")) == []


def test_aligned_n_identical_between_both_bars(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    result = gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        text2term_data_dir=t2t_data_dir,
    )
    for benchmark in pc.BENCHMARK_ORDER:
        r = result.alignment_results[benchmark]
        our_n = sum(gc.align.outcome_counts(r.aligned_rows, field="ours_recomputed_relationship").values())
        t2t_n = sum(gc.align.outcome_counts(r.aligned_rows, field="t2t_recomputed_relationship").values())
        assert our_n == t2t_n == len(r.aligned_rows)


def test_skip_transitions_omits_figure_15c(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        text2term_data_dir=t2t_data_dir, generate_15c=False,
    )
    assert not (output_dir / "pairwise" / "figure_15c_exact_match_transitions_vs_text2term.png").exists()
    assert (output_dir / "pairwise" / "figure_15_graph_relationships_common_query_aligned.png").exists()


def test_no_text2term_data_dir_skips_aligned_figures_entirely(tmp_path: Path) -> None:
    from .test_scenario1_graph_relationship_comparison import _make_full_setup

    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    result = gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    assert result.alignment_results is None
    assert not (output_dir / "pairwise" / "figure_15_graph_relationships_common_query_aligned.png").exists()
    text = figures_md.read_text(encoding="utf-8")
    assert gc.ALIGNED_SECTION_HEADING not in text


def test_figures_md_includes_aligned_section_and_recommendation(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base heading\n\nsome pre-existing descriptive content\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        text2term_data_dir=t2t_data_dir,
    )
    text = figures_md.read_text(encoding="utf-8")
    assert "some pre-existing descriptive content" in text
    assert gc.GRAPH_SECTION_HEADING in text
    assert gc.ALIGNED_SECTION_HEADING in text
    assert "## Recommendation" in text
    # descriptive section must appear BEFORE the aligned section
    assert text.index(gc.GRAPH_SECTION_HEADING) < text.index(gc.ALIGNED_SECTION_HEADING)


def test_rerunning_aligned_build_is_idempotent(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    for _ in range(2):
        gc.build_all(
            ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
            text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
            text2term_data_dir=t2t_data_dir,
        )
    text = figures_md.read_text(encoding="utf-8")
    assert text.count(gc.GRAPH_SECTION_HEADING) == 1
    assert text.count(gc.ALIGNED_SECTION_HEADING) == 1


def test_recommend_figure_15_primary_when_quality_strong(tmp_path: Path) -> None:
    run_dirs, baseline_csv, t2t_data_dir = _full_aligned_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    result = gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        text2term_data_dir=t2t_data_dir,
    )
    promoted, _rationale = gc.recommend_figure_15_primary(result.alignment_results)
    assert promoted is True  # synthetic fixture is 100% aligned with perfect reclassification agreement


def test_recommend_not_primary_when_match_rate_low() -> None:
    # Construct a minimal fake AlignmentResult with a poor match rate.
    fake = gc.align.AlignmentResult(
        benchmark="UKBB-EFO", ours_total_n=100, t2t_total_n=100, ours_single_gold_n=100,
        candidate_exact_matches=50, aligned_rows=[], unmatched=[], ambiguous_n=0, gold_mismatch_n=0,
        reproducibility=gc.align.ReproducibilityCheckResult("UKBB-EFO", {}, {}, True, {}),
    )
    # match_rate_ours = strict_matched_n(0)/ours_single_gold_n(100) = 0.0
    results = {"UKBB-EFO": fake, "Biomappings-EFO": fake, "OLS-EFO (full)": fake}
    promoted, rationale = gc.recommend_figure_15_primary(results)
    assert promoted is False
    assert "NOT promoted" in rationale
