"""Scenario 1 Top-1 graph-relationship comparison vs. original text2term
(figures/graph_relationship_comparison.py + the graph-relationship section of
scripts/plot_scenario1_published_comparison.py).

Filesystem-only (tmp_path), synthetic fixtures -- no network, no mapper, no
LLM calls, and the real completed Scenario 1 run directories under
outputs/evaluation/ are never touched by this suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking import scenario1_graph_distance as graph_distance
from llm_ontology_mapper.benchmarking.figures import graph_relationship_comparison as gc
from llm_ontology_mapper.benchmarking.figures import published_comparison as pc

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Published original-text2term Table 1 values (task spec, Part 2)
# ─────────────────────────────────────────────────────────────────────────────

_PUBLISHED = {
    "UKBB-EFO": {"n": 899, "counts": {"Same": 660, "More Specific": 34, "More General": 20, "Sibling": 13, "Unrelated": 172},
                 "proportions": {"Same": 0.733, "More Specific": 0.040, "More General": 0.022, "Sibling": 0.014, "Unrelated": 0.191}},
    "Biomappings-EFO": {"n": 795, "counts": {"Same": 626, "More Specific": 0, "More General": 2, "Sibling": 47, "Unrelated": 120},
                         "proportions": {"Same": 0.787, "More Specific": 0.000, "More General": 0.003, "Sibling": 0.059, "Unrelated": 0.151}},
    "OLS-EFO (full)": {"n": 8143, "counts": {"Same": 6588, "More Specific": 91, "More General": 55, "Sibling": 89, "Unrelated": 1320},
                        "proportions": {"Same": 0.809, "More Specific": 0.011, "More General": 0.007, "Sibling": 0.011, "Unrelated": 0.162}},
}

_OUR_GRAPH = {
    "UKBB-EFO": {"n": 888, "mapped": 884, "unmapped": 4, "error": 0,
                 "counts": {"Same": 702, "More Specific": 15, "More General": 24, "Sibling": 12, "Unrelated": 131}},
    "Biomappings-EFO": {"n": 795, "mapped": 781, "unmapped": 14, "error": 0,
                         "counts": {"Same": 758, "More Specific": 0, "More General": 1, "Sibling": 4, "Unrelated": 18}},
    "OLS-EFO (full)": {"n": 7377, "mapped": 7262, "unmapped": 115, "error": 0,
                        "counts": {"Same": 6154, "More Specific": 53, "More General": 69, "Sibling": 43, "Unrelated": 943}},
}

_SOURCE_DATASET_PATH = {
    "UKBB-EFO": "UKBB-EFO.csv",
    "Biomappings-EFO": "Biomappings-EFO.csv",
    "OLS-EFO (full)": "OLS-EFO_full.csv",
}

_VALID_METADATA = {
    "source_repository": gc.EXPECTED_SOURCE_REPOSITORY,
    "source_file": "compare_ontology_mappings.py (compare_mappings)",
    "pinned_commit": "b999dbb670fa13c9ceb1ba631a7abc7557f3293b",
    "efo_version": gc.EXPECTED_EFO_VERSION,
    "efo_url": "http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl",
}


def _write_metrics_csv(path: Path, n: int) -> None:
    fields = ("metric", "value", "numerator", "denominator", "evaluation_unit", "status")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for metric in ("Top-1", "Top-3", "Top-5", "MRR"):
            writer.writerow({"metric": metric, "value": 0.5, "numerator": "", "denominator": n,
                              "evaluation_unit": "unique_query", "status": "OK"})


def _make_graph_run_dir(
    tmp_path: Path, benchmark: str, *, gold_count_distribution: dict[int, int] | None = None,
    metadata_overrides: dict | None = None, completed: bool = True,
) -> Path:
    out = tmp_path / benchmark.replace(" ", "_").replace("(", "").replace(")", "")
    out.mkdir()
    spec = _OUR_GRAPH[benchmark]
    n, mapped, unmapped, error = spec["n"], spec["mapped"], spec["unmapped"], spec["error"]

    _write_metrics_csv(out / "scenario1_metrics.csv", n)
    config = {
        "experiment_name": "scenario1_ols_efo",
        "source_dataset_path": _SOURCE_DATASET_PATH[benchmark],
        "completed": completed,
        "rows_completed": n,
        "model": "gpt-5.6-luna",
        "retrieval_mode": "local",
        "target_ontology": "EFO",
        "strict_target_ontology": False,
    }
    (out / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")

    metadata = dict(_VALID_METADATA)
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (out / "graph_reference_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with (out / "execution_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("total", "mapped_count", "unmapped_count", "error_count"))
        writer.writeheader()
        writer.writerow({"total": n, "mapped_count": mapped, "unmapped_count": unmapped, "error_count": error})

    not_applicable = unmapped + error
    with (out / "graph_distance_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("relationship", "count", "percentage", "denominator"))
        writer.writeheader()
        for rel, count in spec["counts"].items():
            writer.writerow({"relationship": rel, "count": count, "percentage": count / n, "denominator": n})
        writer.writerow({"relationship": "Not Applicable", "count": not_applicable, "percentage": not_applicable / n, "denominator": n})

    gold_dist = gold_count_distribution if gold_count_distribution is not None else {1: n}
    (out / "dataset_validation.json").write_text(
        json.dumps({"gold_count_distribution": {str(k): v for k, v in gold_dist.items()}}), encoding="utf-8"
    )
    return out


def _make_all_graph_run_dirs(tmp_path: Path, **overrides) -> dict[str, Path]:
    return {
        benchmark: _make_graph_run_dir(tmp_path, benchmark, **overrides.get(benchmark, {}))
        for benchmark in pc.BENCHMARK_ORDER
    }


def _write_text2term_graph_baseline_csv(path: Path, *, benchmark_overrides: dict | None = None) -> None:
    fields = ("benchmark", "source", "text2term_version", "efo_version", "n", "relationship", "count", "proportion", "publication")
    rows = []
    for benchmark, spec in _PUBLISHED.items():
        n = spec["n"]
        counts = dict(spec["counts"])
        if benchmark_overrides and benchmark in benchmark_overrides:
            counts.update(benchmark_overrides[benchmark].get("counts", {}))
            n = benchmark_overrides[benchmark].get("n", n)
        for rel in gc.RELATIONSHIP_ORDER:
            rows.append({
                "benchmark": benchmark, "source": "original_text2term_paper", "text2term_version": "4.1.2",
                "efo_version": "3.62.0", "n": n, "relationship": rel, "count": counts[rel],
                "proportion": spec["proportions"][rel], "publication": "test fixture",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_full_setup(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    run_dirs = _make_all_graph_run_dirs(
        tmp_path, **{"OLS-EFO (full)": {"gold_count_distribution": {1: 7257, 2: 113, 3: 7}}}
    )
    baseline_csv = tmp_path / "text2term_graph_baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    return run_dirs, baseline_csv


# ─────────────────────────────────────────────────────────────────────────────
# 1-2. category definitions + priority compatibility
# ─────────────────────────────────────────────────────────────────────────────


def test_relationship_order_matches_text2term_compare_mappings_priority() -> None:
    assert gc.RELATIONSHIP_ORDER == ("Same", "More Specific", "More General", "Sibling", "Unrelated")
    assert gc.RELATIONSHIP_ORDER == graph_distance.ALL_RELATIONSHIPS


def test_verify_graph_evaluator_compatibility_passes_for_valid_run(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    metadata = gc.verify_graph_evaluator_compatibility(run_dir)
    assert metadata["efo_version"] == "3.62.0"


def test_verify_graph_evaluator_compatibility_rejects_wrong_efo_version(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO", metadata_overrides={"efo_version": "3.50.0"})
    with pytest.raises(gc.GraphCompatibilityError):
        gc.verify_graph_evaluator_compatibility(run_dir)


def test_verify_graph_evaluator_compatibility_rejects_wrong_source_repository(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO", metadata_overrides={"source_repository": "https://example.com/other"})
    with pytest.raises(gc.GraphCompatibilityError):
        gc.verify_graph_evaluator_compatibility(run_dir)


def test_compatibility_check_would_reject_mismatched_priority_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(graph_distance, "ALL_RELATIONSHIPS", ("Same", "Sibling", "More Specific", "More General", "Unrelated"))
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    with pytest.raises(gc.GraphCompatibilityError):
        gc.verify_graph_evaluator_compatibility(run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 3. text2term EFO version = 3.62.0 / 4. published categories sum to n
# ─────────────────────────────────────────────────────────────────────────────


def test_text2term_baseline_efo_version_is_3_62_0(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    baselines = gc.load_text2term_graph_baseline(baseline_csv)
    assert all(b.efo_version == "3.62.0" for b in baselines.values())
    assert gc.EXPECTED_EFO_VERSION == "3.62.0"


def test_published_categories_must_sum_to_published_n(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv, benchmark_overrides={"UKBB-EFO": {"counts": {"Same": 1}}})
    with pytest.raises(gc.GraphBaselineError, match="sum to"):
        gc.load_text2term_graph_baseline(baseline_csv)


def test_stored_proportion_disagreeing_with_count_over_n_hard_fails(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    text = baseline_csv.read_text(encoding="utf-8").replace(",0.733,", ",0.999,")
    baseline_csv.write_text(text, encoding="utf-8")
    with pytest.raises(gc.GraphBaselineError, match="disagrees"):
        gc.load_text2term_graph_baseline(baseline_csv)


# ─────────────────────────────────────────────────────────────────────────────
# 5-7. UKBB / Biomappings / OLS published values
# ─────────────────────────────────────────────────────────────────────────────


def test_ukbb_published_values(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    b = gc.load_text2term_graph_baseline(baseline_csv)["UKBB-EFO"]
    assert b.n == 899
    assert b.counts == {"Same": 660, "More Specific": 34, "More General": 20, "Sibling": 13, "Unrelated": 172}


def test_biomappings_published_values(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    b = gc.load_text2term_graph_baseline(baseline_csv)["Biomappings-EFO"]
    assert b.n == 795
    assert b.counts == {"Same": 626, "More Specific": 0, "More General": 2, "Sibling": 47, "Unrelated": 120}


def test_ols_published_values(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    _write_text2term_graph_baseline_csv(baseline_csv)
    b = gc.load_text2term_graph_baseline(baseline_csv)["OLS-EFO (full)"]
    assert b.n == 8143
    assert b.counts == {"Same": 6588, "More Specific": 91, "More General": 55, "Sibling": 89, "Unrelated": 1320}


# ─────────────────────────────────────────────────────────────────────────────
# 8-10. our graph rows load correctly / denominators / no-prediction count
# ─────────────────────────────────────────────────────────────────────────────


def test_our_graph_rows_load_correctly(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "OLS-EFO (full)")
    d = gc.load_our_graph_distribution("OLS-EFO (full)", run_dir)
    assert d.n == 7377
    assert d.mapped_count == 7262
    assert d.counts == {"Same": 6154, "More Specific": 53, "More General": 69, "Sibling": 43, "Unrelated": 943}


def test_mapped_only_denominator_calculated_correctly(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "OLS-EFO (full)")
    d = gc.load_our_graph_distribution("OLS-EFO (full)", run_dir)
    props = d.mapped_only_proportions()
    assert props["Same"] == pytest.approx(6154 / 7262)


def test_no_prediction_count_calculated_correctly(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "OLS-EFO (full)")
    d = gc.load_our_graph_distribution("OLS-EFO (full)", run_dir)
    assert d.no_top1_count == 115 == d.unmapped_count + d.error_count


def test_our_graph_load_rejects_inconsistent_not_applicable_bucket(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    # Tamper: change execution_diagnostics unmapped_count so it no longer
    # matches graph_distance_summary's "Not Applicable" count.
    text = (run_dir / "execution_diagnostics.csv").read_text(encoding="utf-8")
    text = text.replace("884,4,0", "883,5,0")  # mapped_count/unmapped_count shifted by 1
    (run_dir / "execution_diagnostics.csv").write_text(text, encoding="utf-8")
    with pytest.raises(gc.OurGraphLoadError):
        gc.load_our_graph_distribution("UKBB-EFO", run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 11-12. mapped-only sums to 1 / end-to-end sums to 1
# ─────────────────────────────────────────────────────────────────────────────


def test_mapped_only_categories_sum_to_one(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    d = gc.load_our_graph_distribution("UKBB-EFO", run_dir)
    assert sum(d.mapped_only_proportions().values()) == pytest.approx(1.0)


def test_end_to_end_categories_sum_to_one(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    d = gc.load_our_graph_distribution("UKBB-EFO", run_dir)
    assert sum(d.end_to_end_proportions().values()) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 13-15. denominator equality/mismatch detection
# ─────────────────────────────────────────────────────────────────────────────


def test_biomappings_denominator_equality_detected(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    data = gc.load_all_graph_data(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        baseline_path=baseline_csv,
    )
    assert data.our["Biomappings-EFO"].n == data.text2term["Biomappings-EFO"].n == 795


def test_ukbb_denominator_mismatch_preserved(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    data = gc.load_all_graph_data(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        baseline_path=baseline_csv,
    )
    assert data.our["UKBB-EFO"].n == 888
    assert data.text2term["UKBB-EFO"].n == 899
    assert data.our["UKBB-EFO"].n != data.text2term["UKBB-EFO"].n


def test_ols_denominator_mismatch_preserved(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    data = gc.load_all_graph_data(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        baseline_path=baseline_csv,
    )
    assert data.our["OLS-EFO (full)"].n == 7377
    assert data.text2term["OLS-EFO (full)"].n == 8143


# ─────────────────────────────────────────────────────────────────────────────
# 16. OLS multi-gold caveat derived from our dataset metadata
# ─────────────────────────────────────────────────────────────────────────────


def test_ols_multi_gold_caveat_derived_from_dataset_metadata(tmp_path: Path) -> None:
    run_dir = _make_graph_run_dir(tmp_path, "OLS-EFO (full)", gold_count_distribution={1: 7257, 2: 113, 3: 7})
    distribution = gc.load_gold_count_distribution(run_dir)
    assert distribution == {1: 7257, 2: 113, 3: 7}
    assert sum(distribution.values()) == 7377


def test_ukbb_and_biomappings_are_single_gold_only(tmp_path: Path) -> None:
    ukbb_dir = _make_graph_run_dir(tmp_path, "UKBB-EFO")
    biomappings_dir = _make_graph_run_dir(tmp_path, "Biomappings-EFO")
    assert gc.load_gold_count_distribution(ukbb_dir) == {1: 888}
    assert gc.load_gold_count_distribution(biomappings_dir) == {1: 795}


# ─────────────────────────────────────────────────────────────────────────────
# 17-20. no raw-file parsing, no fuzzy matching, no forced alignment, no Figure 15
# ─────────────────────────────────────────────────────────────────────────────


def test_module_never_fuzzy_matches_or_parses_raw_text2term_output_files() -> None:
    source = Path(gc.__file__).read_text(encoding="utf-8")
    # No fuzzy-matching library or technique is actually USED anywhere in this
    # module -- "fuzzy" itself appears in prose explicitly stating none was
    # performed, so check for real library usage, not the bare word.
    forbidden_usages = [
        "import difflib", "import rapidfuzz", "import fuzzywuzzy", "Levenshtein.",
        "difflib.SequenceMatcher", "get_close_matches(", "process.extract(",
    ]
    for token in forbidden_usages:
        assert token not in source, f"{token!r} (fuzzy-matching) must never appear in {gc.__file__}"
    assert "no fuzzy" in source.lower(), "expected an explicit statement that no fuzzy matching was performed"
    # The raw text2term-evaluation "output/*" files are named in documentation
    # prose (explaining what was intentionally NOT fetched/parsed) but no
    # code in this module ever opens or reads one.
    for fragment in ("_t2t_mappings.csv", "_mappings.tsv", "_results.tsv"):
        assert fragment in source, f"expected documentation reference to {fragment!r} explaining the skip"
    for opener in ("open(", ".read_csv(", "DictReader("):
        for line in source.splitlines():
            if opener in line:
                assert "_mappings" not in line and "_results" not in line and "t2t_mappings" not in line, (
                    f"line appears to open a raw text2term-evaluation output file: {line!r}"
                )


def test_figure_15_is_never_generated(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")
    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    matches = list(output_dir.rglob("*figure_15*"))
    assert matches == []
    text = figures_md.read_text(encoding="utf-8")
    assert "NOT attempted" in text or "not attempted" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 21-23. PNG/SVG created, PDF NOT created
# ─────────────────────────────────────────────────────────────────────────────


def test_figures_13_14_16_png_and_svg_created_no_pdf(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    result = gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    assert result.output_dir == output_dir

    for name in (
        "figure_13_graph_relationships_mapped_only",
        "figure_14_graph_relationships_end_to_end",
        "figure_16_graph_relationship_delta_vs_text2term",
    ):
        png = output_dir / "pairwise" / f"{name}.png"
        svg = output_dir / "pairwise" / f"{name}.svg"
        assert png.exists() and png.stat().st_size > 0, f"missing {png}"
        assert svg.exists() and svg.stat().st_size > 0, f"missing {svg}"

    assert list(output_dir.rglob("*.pdf")) == []


def test_skip_delta_omits_figure_16(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md, generate_delta=False,
    )
    assert not (output_dir / "pairwise" / "figure_16_graph_relationship_delta_vs_text2term.png").exists()
    assert (output_dir / "pairwise" / "figure_13_graph_relationships_mapped_only.png").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 24-25. FIGURES.md caveats + OM exclusion
# ─────────────────────────────────────────────────────────────────────────────


def test_figures_md_section_includes_source_and_protocol_caveats(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base heading\n\nsome pre-existing content\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    text = figures_md.read_text(encoding="utf-8")

    # The pre-existing base content must be preserved (append, not overwrite).
    assert "some pre-existing content" in text
    assert gc.GRAPH_SECTION_HEADING in text
    assert "CONTROLLED TOP-K BASELINE" in text
    assert "GRAPH-RELATIONSHIP BASELINE" in text
    assert "single-gold" in text.lower()
    assert "multi-gold" in text.lower() or "multi-gold" in text
    assert "8,143" in text  # OLS original text2term n
    assert "7,504" in text  # MetaHarmonizer-controlled rerun n, called out for disambiguation


def test_appending_twice_does_not_duplicate_section(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    for _ in range(2):
        gc.build_all(
            ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
            text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
        )
    text = figures_md.read_text(encoding="utf-8")
    assert text.count(gc.GRAPH_SECTION_HEADING) == 1


def test_metaharmonizer_om_not_included_in_graph_figures(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    mapped_only_csv = (output_dir / "data" / "graph_relationship_mapped_only.csv").read_text(encoding="utf-8")
    assert "MetaHarmonizer" not in mapped_only_csv
    assert "metaharmonizer_om" not in mapped_only_csv.lower()

    text = figures_md.read_text(encoding="utf-8")
    section = text.split(gc.GRAPH_SECTION_HEADING)[1]
    assert "MetaHarmonizer (OM) is NOT included" in section


# ─────────────────────────────────────────────────────────────────────────────
# 26. zero OpenAI/SapBERT/mapper/network calls
# ─────────────────────────────────────────────────────────────────────────────


def test_module_has_no_network_or_mapper_imports() -> None:
    source = Path(gc.__file__).read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator",
                 "import openai", "import requests", "import httpx", "urllib.request", "SapBert"]
    for token in forbidden:
        assert token not in source, f"{token!r} must never appear in {gc.__file__}"


def test_cli_script_still_has_no_network_or_mapper_imports() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_scenario1_published_comparison.py"
    source = script_path.read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator", "import openai"]
    for token in forbidden:
        assert token not in source


# ─────────────────────────────────────────────────────────────────────────────
# Data CSV sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_our_multi_gold_audit_csv_written(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    with (output_dir / "data" / "our_multi_gold_audit.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ols_rows = {int(r["gold_count"]): int(r["n_queries"]) for r in rows if r["benchmark"] == "OLS-EFO (full)"}
    assert ols_rows == {1: 7257, 2: 113, 3: 7}


def test_denominator_comparison_csv_written(tmp_path: Path) -> None:
    run_dirs, baseline_csv = _make_full_setup(tmp_path)
    output_dir = tmp_path / "figures_out"
    figures_md = output_dir / "FIGURES.md"
    figures_md.parent.mkdir(parents=True, exist_ok=True)
    figures_md.write_text("# base\n", encoding="utf-8")

    gc.build_all(
        ols_dir=run_dirs["OLS-EFO (full)"], ukbb_dir=run_dirs["UKBB-EFO"], biomappings_dir=run_dirs["Biomappings-EFO"],
        text2term_baseline_path=baseline_csv, output_dir=output_dir, figures_md_path=figures_md,
    )
    with (output_dir / "data" / "graph_relationship_denominator_comparison.csv").open(newline="", encoding="utf-8") as fh:
        rows = {r["benchmark"]: r for r in csv.DictReader(fh)}
    assert rows["Biomappings-EFO"]["denominators_match"] == "True"
    assert rows["UKBB-EFO"]["denominators_match"] == "False"
    assert rows["OLS-EFO (full)"]["denominators_match"] == "False"
