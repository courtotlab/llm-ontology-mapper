"""Scenario 1 published-baseline comparison figure-suite tests
(figures/published_comparison.py + scripts/plot_scenario1_published_comparison.py).

Filesystem-only (tmp_path), synthetic fixtures -- no network, no mapper, no
LLM calls, and the real completed Scenario 1 run directories under
outputs/evaluation/ are never touched by this suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.figures import published_comparison as pc

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic official-run fixtures
# ─────────────────────────────────────────────────────────────────────────────

_OFFICIAL_METRICS = {
    "UKBB-EFO": {"n": 888, "top1": 0.7905405405405406, "top3": 0.8873873873873874, "top5": 0.8975225225225225, "mrr": 0.8395833333333332},
    "Biomappings-EFO": {"n": 795, "top1": 0.9534591194968554, "top3": 0.9660377358490566, "top5": 0.9660377358490566, "mrr": 0.959538784067086},
    "OLS-EFO (full)": {"n": 7377, "top1": 0.8342144503185577, "top3": 0.8582079436085129, "top5": 0.8598346211196963, "mrr": 0.8457841941168497},
}

_SOURCE_DATASET_PATH = {
    "UKBB-EFO": "UKBB-EFO.csv",
    "Biomappings-EFO": "Biomappings-EFO.csv",
    "OLS-EFO (full)": "OLS-EFO_full.csv",
}

_BASELINE_ROWS = {
    ("UKBB-EFO", "metaharmonizer_ontology_mapper"): {"n": 888, "top1": 0.779, "top3": 0.870, "top5": 0.894, "mrr": 0.826},
    ("UKBB-EFO", "text2term"): {"n": 888, "top1": 0.716, "top3": 0.810, "top5": 0.833, "mrr": 0.765},
    ("Biomappings-EFO", "metaharmonizer_ontology_mapper"): {"n": 795, "top1": 0.955, "top3": 0.982, "top5": 0.987, "mrr": 0.969},
    ("Biomappings-EFO", "text2term"): {"n": 795, "top1": 0.791, "top3": 0.897, "top5": 0.932, "mrr": 0.848},
    ("OLS-EFO (full)", "metaharmonizer_ontology_mapper"): {"n": 7377, "top1": 0.891, "top3": 0.915, "top5": 0.921, "mrr": 0.903},
    ("OLS-EFO (full)", "text2term"): {"n": 7504, "top1": 0.792, "top3": 0.835, "top5": 0.842, "mrr": 0.813},
}


def _make_predictions_csv(path: Path, n: int, top1: float, top3: float, top5: float, mrr: float) -> None:
    """Build a predictions.csv whose recomputed Top-1/3/5/MRR exactly match
    the given targets, using single-gold-code rows scored via
    first_gold_rank/top_k_hit/reciprocal_rank semantics: a hit at rank 1
    counts toward top1/top3/top5/mrr(=1.0); a hit at rank 3 counts toward
    top3/top5 only (mrr=1/3); a hit nowhere counts toward none (mrr=0).
    n is chosen small (a handful of rows) purely for readability -- the
    reconciliation logic being tested is dimension-agnostic.
    """
    n_top1 = round(top1 * n)
    n_top3_only = round((top3 - top1) * n)
    n_top5_only = round((top5 - top3) * n)
    n_miss = n - n_top1 - n_top3_only - n_top5_only
    assert n_miss >= 0, "fixture construction produced a negative miss count"

    fieldnames = [
        "query_id", "query", "gold_codes", "gold_labels", "gold_count", "status",
        "mapped_code", "mapped_term", "mapped_ontology", "confidence",
        "rank_1_code", "rank_1_label", "rank_1_ontology",
        "rank_2_code", "rank_2_label", "rank_2_ontology",
        "rank_3_code", "rank_3_label", "rank_3_ontology",
        "rank_4_code", "rank_4_label", "rank_4_ontology",
        "rank_5_code", "rank_5_label", "rank_5_ontology",
        "first_gold_rank", "top1_hit", "top3_hit", "top5_hit", "reciprocal_rank", "recall_at_gt",
        "graph_relationship", "graph_matched_gold_code", "execution_error", "error_type",
        "error_stage", "error_message", "end_to_end_seconds", "query_planner_seconds",
        "retrieval_seconds", "reranker_seconds", "llm_seconds", "total_input_tokens",
        "total_cached_input_tokens", "total_output_tokens", "total_reasoning_tokens",
        "api_cost_usd", "retrieval_request_count", "retrieval_retry_count",
        "retrieval_recovered_error_count", "retrieval_final_error_count",
        "retrieval_error_sources", "retrieval_error_types",
    ]

    def _blank_row(qid: int) -> dict[str, str]:
        row = dict.fromkeys(fieldnames, "")
        row.update({"query_id": qid, "query": f"q{qid}", "gold_codes": "EFO:GOLD", "status": "mapped"})
        return row

    rows = []
    qid = 0
    for _ in range(n_top1):
        row = _blank_row(qid)
        row["rank_1_code"] = "EFO:GOLD"
        rows.append(row)
        qid += 1
    for _ in range(n_top3_only):
        row = _blank_row(qid)
        row["rank_1_code"] = "EFO:OTHER"
        row["rank_3_code"] = "EFO:GOLD"
        rows.append(row)
        qid += 1
    for _ in range(n_top5_only):
        row = _blank_row(qid)
        row["rank_1_code"] = "EFO:OTHER"
        row["rank_5_code"] = "EFO:GOLD"
        rows.append(row)
        qid += 1
    for _ in range(n_miss):
        row = _blank_row(qid)
        row["rank_1_code"] = "EFO:OTHER"
        rows.append(row)
        qid += 1

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _recompute_expected(n: int, top1: float, top3: float, top5: float) -> tuple[float, float, float, float]:
    """Mirror _make_predictions_csv's rounding so tests assert against the
    actual achievable fixture values, not the (possibly unrounded) inputs."""
    n_top1 = round(top1 * n)
    n_top3_only = round((top3 - top1) * n)
    n_top5_only = round((top5 - top3) * n)
    recomputed_top1 = n_top1 / n
    recomputed_top3 = (n_top1 + n_top3_only) / n
    recomputed_top5 = (n_top1 + n_top3_only + n_top5_only) / n
    recomputed_mrr = (n_top1 * 1.0 + n_top3_only * (1 / 3) + n_top5_only * (1 / 5)) / n
    return recomputed_top1, recomputed_top3, recomputed_top5, recomputed_mrr


def _write_metrics_csv(path: Path, n: int, top1: float, top3: float, top5: float, mrr: float) -> None:
    fields = ("metric", "value", "numerator", "denominator", "evaluation_unit", "status")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for metric, value in (("Top-1", top1), ("Top-3", top3), ("Top-5", top5), ("MRR", mrr)):
            writer.writerow({"metric": metric, "value": value, "numerator": "", "denominator": n,
                              "evaluation_unit": "unique_query", "status": "OK"})


def _make_official_run_dir(
    tmp_path: Path, benchmark: str, *, completed: bool = True, n_override: int | None = None,
    corrupt_metrics: bool = False,
) -> Path:
    out = tmp_path / benchmark.replace(" ", "_").replace("(", "").replace(")", "")
    out.mkdir()
    spec = _OFFICIAL_METRICS[benchmark]
    n = n_override if n_override is not None else spec["n"]

    top1, top3, top5, mrr = _recompute_expected(n, spec["top1"], spec["top3"], spec["top5"])
    _make_predictions_csv(out / "predictions.csv", n, spec["top1"], spec["top3"], spec["top5"], spec["mrr"])
    if corrupt_metrics:
        top1 = 0.0123456  # deliberately wrong vs. predictions.csv
    _write_metrics_csv(out / "scenario1_metrics.csv", n, top1, top3, top5, mrr)

    config = {
        "experiment_name": "scenario1_ols_efo",  # the repo's own runner stamps this literal name for every EFO variant
        "source_dataset_path": _SOURCE_DATASET_PATH[benchmark],
        "completed": completed,
        "rows_completed": n,
        "model": "gpt-5.6-luna",
        "retrieval_mode": "local",
        "target_ontology": "EFO",
        "strict_target_ontology": False,
    }
    (out / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    return out


def _make_all_official_runs(tmp_path: Path, **overrides) -> dict[str, Path]:
    return {
        "UKBB-EFO": _make_official_run_dir(tmp_path, "UKBB-EFO", **overrides.get("UKBB-EFO", {})),
        "Biomappings-EFO": _make_official_run_dir(tmp_path, "Biomappings-EFO", **overrides.get("Biomappings-EFO", {})),
        "OLS-EFO (full)": _make_official_run_dir(tmp_path, "OLS-EFO (full)", **overrides.get("OLS-EFO (full)", {})),
    }


def _write_baselines_csv(path: Path, *, include_disease: bool = False, extra_rows: list[dict] | None = None) -> None:
    fields = ("benchmark", "tool", "metric", "value", "denominator", "source_publication",
              "source_table_or_figure", "notes")
    rows = []
    for (benchmark, tool), spec in _BASELINE_ROWS.items():
        for metric_key, metric_label in (("top1", "Top-1"), ("top3", "Top-3"), ("top5", "Top-5"), ("mrr", "MRR")):
            rows.append({
                "benchmark": benchmark, "tool": tool, "metric": metric_label, "value": spec[metric_key],
                "denominator": spec["n"], "source_publication": "MetaHarmonizer paper (test fixture)",
                "source_table_or_figure": "unverified", "notes": "",
            })
    if include_disease:
        rows.append({
            "benchmark": "OLS-EFO (disease)", "tool": "metaharmonizer_ontology_mapper", "metric": "Top-1",
            "value": 0.5, "denominator": 100, "source_publication": "MetaHarmonizer paper (test fixture)",
            "source_table_or_figure": "unverified", "notes": "should never be consumed",
        })
    if extra_rows:
        rows.extend(extra_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1-2. display labels / 10-11. fixed method+benchmark order
# ─────────────────────────────────────────────────────────────────────────────


def test_metaharmonizer_om_display_label() -> None:
    assert pc.METHOD_DISPLAY["metaharmonizer_om"] == "MetaHarmonizer (OM)"


def test_text2term_display_label() -> None:
    assert pc.METHOD_DISPLAY["text2term"] == "text2term (t2t)"


def test_ours_display_label_is_formal_system_name_not_bare_model() -> None:
    assert pc.METHOD_DISPLAY["ours"] == "LLM Ontology Mapper"
    assert pc.METHOD_DISPLAY["ours"].lower() != "model"
    assert pc.METHOD_DISPLAY["ours"].lower() != "our model"


def test_method_order_fixed_and_never_alphabetical_or_performance_sorted() -> None:
    assert pc.METHOD_ORDER == ("ours", "metaharmonizer_om", "text2term")


def test_benchmark_order_fixed() -> None:
    assert pc.BENCHMARK_ORDER == ("UKBB-EFO", "Biomappings-EFO", "OLS-EFO (full)")


def test_method_colors_stable_across_the_whole_module() -> None:
    # Same colors must be reused everywhere a method appears -- there is only
    # one METHOD_COLORS dict, so this is really testing there's no second
    # (possibly divergent) color table anywhere in the module.
    assert len(pc.METHOD_COLORS) == 3
    assert len(set(pc.METHOD_COLORS.values())) == 3  # no two methods share a color


# ─────────────────────────────────────────────────────────────────────────────
# 3-6. published baseline table loading
# ─────────────────────────────────────────────────────────────────────────────


def test_baseline_table_loads_ukbb_values_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path)
    baselines = pc.load_baselines(csv_path)
    om = baselines[("UKBB-EFO", "metaharmonizer_om")]
    assert (om.n, om.top1, om.top3, om.top5, om.mrr) == (888, 0.779, 0.870, 0.894, 0.826)
    t2t = baselines[("UKBB-EFO", "text2term")]
    assert (t2t.n, t2t.top1) == (888, 0.716)


def test_baseline_table_loads_biomappings_values_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path)
    baselines = pc.load_baselines(csv_path)
    om = baselines[("Biomappings-EFO", "metaharmonizer_om")]
    assert (om.n, om.top1, om.mrr) == (795, 0.955, 0.969)


def test_baseline_table_loads_ols_full_values_correctly_with_differing_denominators(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path)
    baselines = pc.load_baselines(csv_path)
    om = baselines[("OLS-EFO (full)", "metaharmonizer_om")]
    t2t = baselines[("OLS-EFO (full)", "text2term")]
    assert om.n == 7377
    assert t2t.n == 7504
    assert om.n != t2t.n


def test_ols_disease_rows_excluded_not_hard_failed(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path, include_disease=True)
    baselines = pc.load_baselines(csv_path)
    assert all(b != "OLS-EFO (disease)" for (b, _tool) in baselines)


def test_truly_unknown_benchmark_hard_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path, extra_rows=[{
        "benchmark": "SomeOtherBenchmark", "tool": "text2term", "metric": "Top-1", "value": 0.5,
        "denominator": 10, "source_publication": "", "source_table_or_figure": "", "notes": "",
    }])
    with pytest.raises(pc.BaselineTableError):
        pc.load_baselines(csv_path)


def test_duplicate_baseline_row_hard_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    dup = {
        "benchmark": "UKBB-EFO", "tool": "text2term", "metric": "Top-1", "value": 0.716,
        "denominator": 888, "source_publication": "", "source_table_or_figure": "", "notes": "",
    }
    _write_baselines_csv(csv_path, extra_rows=[dup])
    with pytest.raises(pc.BaselineTableError, match="duplicate"):
        pc.load_baselines(csv_path)


def test_malformed_baseline_value_hard_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path)
    text = csv_path.read_text(encoding="utf-8").replace("0.716", "not-a-number")
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(pc.BaselineTableError):
        pc.load_baselines(csv_path)


def test_missing_baseline_metric_hard_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    # Drop the UKBB-EFO/text2term/Top-1 row entirely.
    filtered = [ln for ln in lines if not ln.startswith("UKBB-EFO,text2term,Top-1,")]
    assert len(filtered) == len(lines) - 1
    csv_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    with pytest.raises(pc.BaselineTableError):
        pc.load_baselines(csv_path)


def test_unknown_tool_hard_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "baselines.csv"
    _write_baselines_csv(csv_path, extra_rows=[{
        "benchmark": "UKBB-EFO", "tool": "some_other_tool", "metric": "Top-1", "value": 0.5,
        "denominator": 888, "source_publication": "", "source_table_or_figure": "", "notes": "",
    }])
    with pytest.raises(pc.BaselineTableError):
        pc.load_baselines(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7-9. our-run loading: saved metrics, stale run rejected, incomplete rejected
# ─────────────────────────────────────────────────────────────────────────────


def test_our_run_values_loaded_from_saved_scenario1_metrics_csv(tmp_path: Path) -> None:
    run_dir = _make_official_run_dir(tmp_path, "UKBB-EFO")
    run = pc.load_official_run("UKBB-EFO", run_dir)
    assert run.metrics.n == 888
    assert run.metrics.top1 == pytest.approx(_OFFICIAL_METRICS["UKBB-EFO"]["top1"], abs=1e-3)


def test_stale_partial_run_with_wrong_n_rejected(tmp_path: Path) -> None:
    # A "stale" OLS run that only completed a partial N (e.g. an earlier,
    # superseded attempt) must never be silently accepted.
    run_dir = _make_official_run_dir(tmp_path, "OLS-EFO (full)", n_override=100)
    with pytest.raises(pc.RunLoadError, match="stale/partial"):
        pc.load_official_run("OLS-EFO (full)", run_dir)


def test_incomplete_run_rejected(tmp_path: Path) -> None:
    run_dir = _make_official_run_dir(tmp_path, "UKBB-EFO", completed=False)
    with pytest.raises(pc.RunLoadError, match="completed"):
        pc.load_official_run("UKBB-EFO", run_dir)


def test_reconciliation_passes_for_consistent_fixture(tmp_path: Path) -> None:
    run_dir = _make_official_run_dir(tmp_path, "Biomappings-EFO")
    run = pc.load_official_run("Biomappings-EFO", run_dir)
    pc.reconcile_official_run(run)  # must not raise


def test_reconciliation_fails_for_tampered_metrics_csv(tmp_path: Path) -> None:
    run_dir = _make_official_run_dir(tmp_path, "Biomappings-EFO", corrupt_metrics=True)
    run = pc.load_official_run("Biomappings-EFO", run_dir)
    with pytest.raises(pc.ReconciliationError):
        pc.reconcile_official_run(run)


def test_load_all_official_runs_reconciles_every_run(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    assert set(runs) == {"UKBB-EFO", "Biomappings-EFO", "OLS-EFO (full)"}
    assert runs["OLS-EFO (full)"].metrics.n == 7377


# ─────────────────────────────────────────────────────────────────────────────
# 12-13. all-methods extraction (Top-k + MRR)
# ─────────────────────────────────────────────────────────────────────────────


def test_all_methods_topk_extraction(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines_csv = tmp_path / "baselines.csv"
    _write_baselines_csv(baselines_csv)
    baselines = pc.load_baselines(baselines_csv)
    all_methods = pc.build_all_methods(runs, baselines)

    assert all_methods["UKBB-EFO"]["metaharmonizer_om"].top1 == 0.779
    assert all_methods["UKBB-EFO"]["text2term"].top3 == 0.810
    assert all_methods["OLS-EFO (full)"]["ours"].n == 7377
    assert all_methods["OLS-EFO (full)"]["text2term"].n == 7504


def test_all_methods_mrr_extraction(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines_csv = tmp_path / "baselines.csv"
    _write_baselines_csv(baselines_csv)
    baselines = pc.load_baselines(baselines_csv)
    all_methods = pc.build_all_methods(runs, baselines)
    assert all_methods["Biomappings-EFO"]["metaharmonizer_om"].mrr == 0.969


# ─────────────────────────────────────────────────────────────────────────────
# 14-15. pairwise data
# ─────────────────────────────────────────────────────────────────────────────


def test_pairwise_ours_vs_om_data(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines = pc.load_baselines(_write_and_return(tmp_path))
    all_methods = pc.build_all_methods(runs, baselines)
    out_csv = tmp_path / "pairwise_om.csv"
    pc.write_pairwise_csv(all_methods, "metaharmonizer_om", out_csv)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    methods_seen = {r["method"] for r in rows}
    assert methods_seen == {"LLM Ontology Mapper", "MetaHarmonizer (OM)"}
    assert "text2term (t2t)" not in methods_seen


def test_pairwise_ours_vs_t2t_data(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines = pc.load_baselines(_write_and_return(tmp_path))
    all_methods = pc.build_all_methods(runs, baselines)
    out_csv = tmp_path / "pairwise_t2t.csv"
    pc.write_pairwise_csv(all_methods, "text2term", out_csv)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    methods_seen = {r["method"] for r in rows}
    assert methods_seen == {"LLM Ontology Mapper", "text2term (t2t)"}


def _write_and_return(tmp_path: Path) -> Path:
    p = tmp_path / "baselines_shared.csv"
    if not p.exists():
        _write_baselines_csv(p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 16-19. delta calculation + sign convention
# ─────────────────────────────────────────────────────────────────────────────


def test_delta_top1_top3_top5_calculation_and_sign_convention(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines = pc.load_baselines(_write_and_return(tmp_path))
    all_methods = pc.build_all_methods(runs, baselines)

    delta_rows = pc.compute_deltas(all_methods, "metaharmonizer_om")
    ukbb_row = next(r for r in delta_rows if r["benchmark"] == "UKBB-EFO")

    ours = all_methods["UKBB-EFO"]["ours"]
    om = all_methods["UKBB-EFO"]["metaharmonizer_om"]
    # Sign convention: delta = ours - baseline (never baseline - ours).
    assert ukbb_row["delta_top1_pp"] == pytest.approx((ours.top1 - om.top1) * 100.0)
    assert ukbb_row["delta_top3_pp"] == pytest.approx((ours.top3 - om.top3) * 100.0)
    assert ukbb_row["delta_top5_pp"] == pytest.approx((ours.top5 - om.top5) * 100.0)
    assert ukbb_row["delta_mrr"] == pytest.approx(ours.mrr - om.mrr)
    # UKBB-EFO: our fixture Top-1 (~0.7905) is ABOVE the OM baseline (0.779)
    # -> delta must be positive.
    assert ukbb_row["delta_top1_pp"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 20-21. OLS denominator difference preserved / n shown correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_ols_denominator_difference_preserved_in_delta_rows(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines = pc.load_baselines(_write_and_return(tmp_path))
    all_methods = pc.build_all_methods(runs, baselines)
    delta_rows = pc.compute_deltas(all_methods, "text2term")
    ols_row = next(r for r in delta_rows if r["benchmark"] == "OLS-EFO (full)")
    assert ols_row["ours_n"] == 7377
    assert ols_row["baseline_n"] == 7504
    assert ols_row["ours_n"] != ols_row["baseline_n"]


def test_n_shown_correctly_for_all_tools_in_our_metrics_csv(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    out_csv = tmp_path / "our_metrics.csv"
    pc.write_our_scenario1_metrics_used_csv(runs, out_csv)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = {r["benchmark"]: r for r in csv.DictReader(fh)}
    assert rows["UKBB-EFO"]["n"] == "888"
    assert rows["Biomappings-EFO"]["n"] == "795"
    assert rows["OLS-EFO (full)"]["n"] == "7377"


# ─────────────────────────────────────────────────────────────────────────────
# 22-24. PNG/SVG created, PDF NOT created
# ─────────────────────────────────────────────────────────────────────────────


def test_build_all_creates_png_and_svg_never_pdf(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"

    result = pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    assert result.output_dir == output_dir

    expected_basenames = [
        ("main", "figure_01_all_methods_topk"),
        ("main", "figure_02_all_methods_mrr"),
        ("pairwise", "figure_03_our_model_vs_metaharmonizer_topk"),
        ("pairwise", "figure_04_our_model_vs_metaharmonizer_mrr"),
        ("pairwise", "figure_05_our_model_vs_text2term_topk"),
        ("pairwise", "figure_06_our_model_vs_text2term_mrr"),
        ("pairwise", "figure_07_delta_vs_metaharmonizer"),
        ("pairwise", "figure_08_delta_vs_text2term"),
        ("main", "figure_09_top1_summary"),
    ]
    for subdir, name in expected_basenames:
        png = output_dir / subdir / f"{name}.png"
        svg = output_dir / subdir / f"{name}.svg"
        assert png.exists() and png.stat().st_size > 0, f"missing {png}"
        assert svg.exists() and svg.stat().st_size > 0, f"missing {svg}"

    pdf_files = list(output_dir.rglob("*.pdf"))
    assert pdf_files == [], f"PDF files must never be produced by this suite: {pdf_files}"


def test_save_figure_default_formats_unchanged_for_other_suites() -> None:
    # style.save_figure's default (used by scenario1/scenario2 pre-existing
    # figure suites) must still include pdf -- only this module opts out.
    import inspect

    from llm_ontology_mapper.benchmarking.figures.style import save_figure

    default_formats = inspect.signature(save_figure).parameters["formats"].default
    assert "pdf" in default_formats


def test_published_comparison_requests_png_and_svg_only() -> None:
    assert pc.FORMATS == ("png", "svg")
    assert "pdf" not in pc.FORMATS


# ─────────────────────────────────────────────────────────────────────────────
# 25-28. FIGURES.md content
# ─────────────────────────────────────────────────────────────────────────────


def test_figures_md_generated_and_mentions_required_terminology(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    figures_md = output_dir / "FIGURES.md"
    assert figures_md.exists()
    text = figures_md.read_text(encoding="utf-8")

    assert "OntologyMapper" in text
    assert "MetaHarmonizer (OM)" in text
    assert "text2term (t2t)" in text
    assert "7377" in text and "7504" in text
    assert "figure_01_all_methods_topk" in text
    assert "figure_09_top1_summary" in text


# ─────────────────────────────────────────────────────────────────────────────
# 29-30. no network/provider/mapper calls; source runs never modified
# ─────────────────────────────────────────────────────────────────────────────


def test_published_comparison_module_has_no_llm_mapper_or_network_imports() -> None:
    import llm_ontology_mapper.benchmarking.figures.published_comparison as module

    forbidden = [
        "OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator",
        "import openai", "import requests", "import httpx", "urllib.request", "SapBert",
    ]
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, f"{token!r} must never appear in {module.__name__}"


def test_cli_script_has_no_llm_mapper_or_network_imports() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_scenario1_published_comparison.py"
    source = script_path.read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator", "import openai"]
    for token in forbidden:
        assert token not in source, f"{token!r} must never appear in the zero-LLM-call plotting CLI"


def test_build_all_does_not_modify_original_run_artifacts(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    before = {
        b: (d / "predictions.csv").read_text(encoding="utf-8") for b, d in dirs.items()
    }
    before_metrics = {
        b: (d / "scenario1_metrics.csv").read_text(encoding="utf-8") for b, d in dirs.items()
    }
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=tmp_path / "figures_out",
    )
    for b, d in dirs.items():
        assert (d / "predictions.csv").read_text(encoding="utf-8") == before[b]
        assert (d / "scenario1_metrics.csv").read_text(encoding="utf-8") == before_metrics[b]


# ═════════════════════════════════════════════════════════════════════════════
# Cross-method ranked-outcome distribution (Figures 10/11/12)
# ═════════════════════════════════════════════════════════════════════════════

# 1-4: formula checks, with an easy-to-hand-verify synthetic MethodMetrics.
_SAMPLE_METRICS = pc.MethodMetrics(n=1000, top1=0.70, top3=0.85, top5=0.92, mrr=0.77)


def test_rank1_equals_top1() -> None:
    d = pc.derive_outcome_distribution(_SAMPLE_METRICS)
    assert d.gold_rank_1 == pytest.approx(0.70)


def test_rank2_3_equals_top3_minus_top1() -> None:
    d = pc.derive_outcome_distribution(_SAMPLE_METRICS)
    assert d.gold_rank_2_3 == pytest.approx(0.85 - 0.70)


def test_rank4_5_equals_top5_minus_top3() -> None:
    d = pc.derive_outcome_distribution(_SAMPLE_METRICS)
    assert d.gold_rank_4_5 == pytest.approx(0.92 - 0.85)


def test_no_gold_top5_equals_one_minus_top5() -> None:
    d = pc.derive_outcome_distribution(_SAMPLE_METRICS)
    assert d.no_gold_top5 == pytest.approx(1.0 - 0.92)


def test_four_categories_sum_to_approximately_one() -> None:
    d = pc.derive_outcome_distribution(_SAMPLE_METRICS)
    total = d.gold_rank_1 + d.gold_rank_2_3 + d.gold_rank_4_5 + d.no_gold_top5
    assert total == pytest.approx(1.0, abs=1e-9)


# 6-11: exact published validation values from the task spec, loaded through
# the same baseline CSV (never a second hardcoded copy of these numbers).


def _load_baseline_distribution(tmp_path: Path, benchmark: str, tool_key: str) -> pc.OutcomeDistribution:
    csv_path = tmp_path / "baselines.csv"
    if not csv_path.exists():
        _write_baselines_csv(csv_path)
    baselines = pc.load_baselines(csv_path)
    return pc.derive_outcome_distribution(baselines[(benchmark, tool_key)])


def test_ukbb_om_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "UKBB-EFO", "metaharmonizer_om")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.779, 0.091, 0.024, 0.106)


def test_ukbb_t2t_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "UKBB-EFO", "text2term")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.716, 0.094, 0.023, 0.167)


def test_biomappings_om_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "Biomappings-EFO", "metaharmonizer_om")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.955, 0.027, 0.005, 0.013)


def test_biomappings_t2t_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "Biomappings-EFO", "text2term")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.791, 0.106, 0.035, 0.068)


def test_ols_om_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "OLS-EFO (full)", "metaharmonizer_om")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.891, 0.024, 0.006, 0.079)


def test_ols_t2t_expected_distribution(tmp_path: Path) -> None:
    d = _load_baseline_distribution(tmp_path, "OLS-EFO (full)", "text2term")
    assert (round(d.gold_rank_1, 3), round(d.gold_rank_2_3, 3), round(d.gold_rank_4_5, 3), round(d.no_gold_top5, 3)) == (0.792, 0.043, 0.007, 0.158)


# 12-13: our method uses the identical derivation, and its richer per-row
# outcome vocabulary (execution error / abstained) never appears cross-method.


def test_our_method_uses_same_derivation_as_baselines(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    ours = runs["UKBB-EFO"].metrics
    d = pc.derive_outcome_distribution(ours)
    assert d.gold_rank_1 == pytest.approx(ours.top1)
    assert d.gold_rank_2_3 == pytest.approx(ours.top3 - ours.top1)
    assert d.gold_rank_4_5 == pytest.approx(ours.top5 - ours.top3)
    assert d.no_gold_top5 == pytest.approx(1.0 - ours.top5)


def test_cross_method_categories_exclude_execution_error_and_abstained() -> None:
    assert "Execution error" not in pc.CROSS_METHOD_OUTCOME_CATEGORIES
    assert "Abstained" not in pc.CROSS_METHOD_OUTCOME_CATEGORIES
    assert pc.CROSS_METHOD_OUTCOME_CATEGORIES == ("Gold rank 1", "Gold rank 2-3", "Gold rank 4-5", "No gold in Top 5")


def test_negative_bin_from_non_monotonic_topk_hard_fails() -> None:
    # top3 < top1 is impossible for real cumulative accuracy but must still
    # be caught rather than silently plotted as a negative-height segment.
    bad = pc.MethodMetrics(n=100, top1=0.90, top3=0.80, top5=0.95, mrr=0.85)
    with pytest.raises(pc.PublishedComparisonError):
        pc.derive_outcome_distribution(bad)


# 14-16: all three outcome-distribution figures created


def test_outcome_distribution_figures_all_created(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    expected = [
        ("main", "figure_10_all_methods_outcome_distribution"),
        ("pairwise", "figure_11_outcome_distribution_vs_metaharmonizer"),
        ("pairwise", "figure_12_outcome_distribution_vs_text2term"),
    ]
    for subdir, name in expected:
        png = output_dir / subdir / f"{name}.png"
        svg = output_dir / subdir / f"{name}.svg"
        assert png.exists() and png.stat().st_size > 0, f"missing {png}"
        assert svg.exists() and svg.stat().st_size > 0, f"missing {svg}"


# 17-19: PNG/SVG generated, no PDF, for the outcome-distribution figures specifically


def test_outcome_distribution_figures_no_pdf(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    pdf_files = list(output_dir.rglob("*outcome_distribution*.pdf"))
    assert pdf_files == []


# 20: OLS n difference retained in the outcome-distribution CSV


def test_outcome_distribution_csv_retains_ols_denominator_difference(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    runs = pc.load_all_official_runs(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
    )
    baselines = pc.load_baselines(_write_and_return(tmp_path))
    all_methods = pc.build_all_methods(runs, baselines)
    distributions = pc.build_outcome_distributions(all_methods)
    out_csv = tmp_path / "outcome_distribution.csv"
    pc.write_outcome_distribution_csv(all_methods, distributions, out_csv)
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = {(r["benchmark"], r["method"]): r for r in csv.DictReader(fh)}
    assert rows[("OLS-EFO (full)", "LLM Ontology Mapper")]["n"] == "7377"
    assert rows[("OLS-EFO (full)", "MetaHarmonizer (OM)")]["n"] == "7377"
    assert rows[("OLS-EFO (full)", "text2term (t2t)")]["n"] == "7504"


# 21-22: FIGURES.md derivation formulas + baseline-abstention-inability note


def test_figures_md_includes_derivation_formulas_and_limitation_note(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    text = (output_dir / "FIGURES.md").read_text(encoding="utf-8")

    assert "Gold rank 1 = Top-1" in text
    assert "Top-3" in text and "Top-1" in text  # rank2-3 formula components present
    assert "Outcome-distribution derivation" in text
    assert "does not expose execution errors, abstentions" in text
    assert "figure_10_all_methods_outcome_distribution" in text
    assert "figure_11_outcome_distribution_vs_metaharmonizer" in text
    assert "figure_12_outcome_distribution_vs_text2term" in text


# 23: original text2term graph-relationship distribution never used here


def test_original_text2term_graph_distribution_not_used_as_controlled_source() -> None:
    # The Same/More-Specific/.../Unrelated vocabulary (original text2term
    # paper's own Top-1 graph-relationship taxonomy) must never appear as a
    # cross-method outcome category or color key here -- it may only appear
    # in FIGURES.md prose explaining why it is excluded (checked separately
    # in test_figures_md_documents_online_source_distinction).
    forbidden = {"Same", "More Specific", "More General", "Sibling", "Unrelated"}
    assert forbidden.isdisjoint(pc.CROSS_METHOD_OUTCOME_CATEGORIES)
    assert forbidden.isdisjoint(pc.CROSS_METHOD_OUTCOME_COLORS)


def test_figures_md_documents_online_source_distinction(tmp_path: Path) -> None:
    dirs = _make_all_official_runs(tmp_path)
    baselines_csv = _write_and_return(tmp_path)
    output_dir = tmp_path / "figures_out"
    pc.build_all(
        ols_dir=dirs["OLS-EFO (full)"], ukbb_dir=dirs["UKBB-EFO"], biomappings_dir=dirs["Biomappings-EFO"],
        baselines_path=baselines_csv, output_dir=output_dir,
    )
    text = (output_dir / "FIGURES.md").read_text(encoding="utf-8")
    assert "original text2term" in text.lower()
    assert "Same / More Specific / More General / Sibling / Unrelated" in text
    assert "MetaHarmonizer-controlled rerun" in text


# 24: zero network/provider/mapper calls (already broadly covered above, but
# re-asserted specifically against the new outcome-distribution code paths)


def test_outcome_distribution_code_has_no_network_or_mapper_imports() -> None:
    import llm_ontology_mapper.benchmarking.figures.published_comparison as module

    forbidden = ["OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator",
                 "import openai", "import requests", "import httpx", "urllib.request", "SapBert"]
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source
