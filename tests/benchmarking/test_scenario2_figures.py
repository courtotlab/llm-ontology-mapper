"""Scenario 2 figure-suite tests (figures/scenario2.py + figures/style.py +
figures/common.py + scripts/plot_scenario2_results.py).

Filesystem-only (tmp_path), synthetic fixtures -- no network, no mapper, no
LLM calls, no real Scenario 2 run directories are read. Figure builders are
exercised against small hand-computed fixtures rather than the full 218-row
official runs, so this suite stays fast.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.figures import scenario2 as s2figs
from llm_ontology_mapper.benchmarking.figures.style import MODE_DISPLAY, MODE_ORDER
from llm_ontology_mapper.benchmarking.scenario2_compare import (
    CompareConfigMismatchError,
    CompareDatasetMismatchError,
)
from llm_ontology_mapper.benchmarking.scenario2_output import PREDICTIONS_CSV_FIELDS

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture: 6 rows (3x HPO, 3x MONDO), single gold/rank per row so
# top1 == top3 == top5 == mrr == recall_at_gt by construction, which keeps the
# hand-computed mode_summary.csv values simple and unambiguous.
# ─────────────────────────────────────────────────────────────────────────────

_BASE_CONFIG = {
    "source_dataset_sha256": "fixture-sha",
    "dataset_row_count": 6,
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "reasoning_effort": "low",
    "temperature": None,
    "temperature_mode": "provider_default",
    "seed": 42,
    "max_alternatives": 4,
    "strict_target_ontology": False,
    "completed": True,
}

_ROW_META = {
    1: ("v1", "HPO", "HP:0001"),
    2: ("v2", "HPO", "HP:0002"),
    3: ("v3", "HPO", "HP:0003"),
    4: ("v4", "MONDO", "MONDO:0001"),
    5: ("v5", "MONDO", "MONDO:0002"),
    6: ("v6", "MONDO", "MONDO:0003"),
}

# mode -> {row_id: outcome}; "correct" / "incorrect" / "unmapped" / "error"
_OUTCOMES: dict[str, dict[int, str]] = {
    "public": {1: "correct", 2: "correct", 3: "incorrect", 4: "correct", 5: "incorrect", 6: "incorrect"},
    "local": {1: "correct", 2: "incorrect", 3: "correct", 4: "correct", 5: "correct", 6: "incorrect"},
    "disabled": {1: "error", 2: "unmapped", 3: "correct", 4: "correct", 5: "incorrect", 6: "incorrect"},
}


def _blank_row() -> dict[str, str]:
    return dict.fromkeys(PREDICTIONS_CSV_FIELDS, "")


def _make_prediction_row(row_id: int, mode: str, outcome: str) -> dict[str, str]:
    source_variable, ontology, gold_code = _ROW_META.get(row_id, (f"v{row_id}", "HPO", f"HP:{row_id:04d}"))
    row = _blank_row()
    row.update(
        {
            "row_id": row_id,
            "source_variable": source_variable,
            "source_label": source_variable,
            "source_description": "",
            "target_ontology": ontology,
            "gold_codes": gold_code,
            "gold_terms": "",
            "retrieval_mode": mode,
        }
    )
    if outcome == "error":
        row.update(
            {
                "status": "error",
                "execution_error": "True",
                "error_stage": "planner",
                "error_type": "TestError",
                # score_prediction() always scores an error row's ranks as
                # all-None -> top1_hit/semantic_correctness=False, never blank
                # (matches row_result_to_csv_dict in production).
                "top1_hit": "False",
                "top3_hit": "False",
                "top5_hit": "False",
                "reciprocal_rank": 0.0,
                "recall_at_gt": 0.0,
                "semantic_correctness": "False",
            }
        )
        return row
    if outcome == "unmapped":
        row.update(
            {
                "status": "unmapped",
                "execution_error": "False",
                "top1_hit": "False",
                "top3_hit": "False",
                "top5_hit": "False",
                "reciprocal_rank": 0.0,
                "recall_at_gt": 0.0,
                "semantic_correctness": "False",
            }
        )
        return row

    is_correct = outcome == "correct"
    mapped_code = gold_code if is_correct else "OTHER:9999"
    confidence = 0.9 if is_correct else 0.4
    row.update(
        {
            "status": "mapped",
            "mapped_code": mapped_code,
            "mapped_code_normalized": mapped_code,
            "mapped_term": "x",
            "mapped_ontology": ontology,
            "confidence": confidence,
            "rank_1_code": mapped_code,
            "rank_1_term": "x",
            "first_gold_rank": 1 if is_correct else "",
            "top1_hit": str(is_correct),
            "top3_hit": str(is_correct),
            "top5_hit": str(is_correct),
            "reciprocal_rank": 1.0 if is_correct else 0.0,
            "recall_at_gt": 1.0 if is_correct else 0.0,
            "semantic_correctness": str(is_correct),
            "is_grounded": "True",
            "grounding_source": "retrieved" if mode != "disabled" else "",
            "selected_code_was_retrieved": "True" if mode != "disabled" else "False",
            "retrieval_skipped": "False",
            "validation_status": "VALID",
            "validation_source": "OLS4",
            "execution_error": "False",
            "planner_seconds": 1.0,
            "retrieval_seconds": 0.5 if mode != "disabled" else "",
            "reranker_seconds": 0.5 if mode != "disabled" else "",
            "llm_seconds": 1.5,
            "end_to_end_seconds": 3.0,
        }
    )
    return row


def _mode_metrics(outcomes: dict[int, str]) -> dict[str, float]:
    n = len(outcomes)
    correct = sum(1 for o in outcomes.values() if o == "correct")
    errors = sum(1 for o in outcomes.values() if o == "error")
    abstained = sum(1 for o in outcomes.values() if o == "unmapped")
    rate = correct / n
    return {
        "n": n,
        "top1_accuracy": rate,
        "top3_accuracy": rate,
        "top5_accuracy": rate,
        "mrr": rate,
        "recall_at_gt": rate,
        "abstention_rate": abstained / n,
        "execution_error_count": errors,
        "hallucination_rate": 0.1,
        "validation_coverage": 1.0,
        "grounding_rate": 1.0,  # overwritten to 0.0 for mode="disabled" in _write_mode_summary
        "roc_auc": 0.75,
        "brier_score": 0.2,
        "ece": 0.15,
        "cohens_d": 0.5,
        "execution_error_rate": errors / n,
        "mean_end_to_end_seconds": 3.0,
        "mean_llm_seconds": 1.5,
        "mean_api_cost_per_row_usd": 0.001,
        "total_api_cost_usd": 0.006,
    }


def _write_mode_summary(path: Path, mode: str, metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerow({"metric": "mode", "value": mode})
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})
    # grounding_rate for disabled must mechanically be 0 -- fix the sentinel
    # placeholder above with a real, mode-aware value.
    if mode == "disabled":
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        for r in rows:
            if r["metric"] == "grounding_rate":
                r["value"] = "0.0"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(rows)


def _write_calibration_bins(path: Path, mode: str) -> None:
    fields = ("mode", "bin_lower", "bin_upper", "count", "mean_confidence", "empirical_accuracy", "calibration_gap")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {"mode": mode, "bin_lower": 0.8, "bin_upper": 0.9, "count": 2, "mean_confidence": 0.85, "empirical_accuracy": 0.9, "calibration_gap": 0.05}
        )


def _make_run_dir(
    tmp_path: Path, mode: str, *, completed: bool = True, config_overrides: dict | None = None, outcomes: dict[int, str] | None = None
) -> Path:
    out = tmp_path / mode
    out.mkdir()
    config = dict(_BASE_CONFIG)
    config["retrieval_mode"] = mode
    config["completed"] = completed
    if config_overrides:
        config.update(config_overrides)
    (out / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")

    resolved_outcomes = outcomes if outcomes is not None else _OUTCOMES[mode]
    rows = [_make_prediction_row(row_id, mode, outcome) for row_id, outcome in sorted(resolved_outcomes.items())]
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_mode_summary(out / "mode_summary.csv", mode, _mode_metrics(resolved_outcomes))
    _write_calibration_bins(out / "calibration_bins.csv", mode)
    return out


def _make_all_runs(tmp_path: Path, **kwargs) -> tuple[Path, Path, Path]:
    return (
        _make_run_dir(tmp_path, "public", **kwargs.get("public", {})),
        _make_run_dir(tmp_path, "local", **kwargs.get("local", {})),
        _make_run_dir(tmp_path, "disabled", **kwargs.get("disabled", {})),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full run compatibility accepted / 3. incomplete run rejected /
# 4. row-id mismatch rejected (propagated from scenario2_compare, not
#    reimplemented here)
# ─────────────────────────────────────────────────────────────────────────────


def test_compatible_completed_runs_accepted(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    assert set(runs) == {"public", "local", "disabled"}


def test_incomplete_run_rejected(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path, disabled={"completed": False})
    with pytest.raises(s2figs.ScenarioCompatibilityError):
        s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


def test_row_id_mismatch_rejected(tmp_path: Path) -> None:
    mismatched_outcomes = dict(_OUTCOMES["disabled"])
    mismatched_outcomes[99] = mismatched_outcomes.pop(6)  # different row_id set
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path, disabled={"outcomes": mismatched_outcomes})
    with pytest.raises(CompareDatasetMismatchError):
        s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


def test_config_mismatch_rejected(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path, disabled={"config_overrides": {"model": "gpt-5.4-mini"}})
    with pytest.raises(CompareConfigMismatchError):
        s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 5. mode order fixed Public/Local/Disabled
# ─────────────────────────────────────────────────────────────────────────────


def test_mode_order_and_display_names_fixed() -> None:
    assert s2figs.MODES == ("public", "local", "disabled") == MODE_ORDER
    assert MODE_DISPLAY == {"public": "Public", "local": "Local", "disabled": "Disabled"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Top-1 (and other) source values reconcile with saved mode summary
# ─────────────────────────────────────────────────────────────────────────────


def test_reconciliation_passes_for_consistent_fixture(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    mode_summaries = s2figs.load_mode_summaries(runs)
    s2figs.reconcile_all(runs, mode_summaries)  # must not raise


def test_reconciliation_fails_for_tampered_mode_summary(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    mode_summaries = s2figs.load_mode_summaries(runs)
    mode_summaries["public"]["top1_accuracy"] = "0.999999"  # deliberately wrong
    with pytest.raises(s2figs.ReconciliationError):
        s2figs.reconcile_all(runs, mode_summaries)


# ─────────────────────────────────────────────────────────────────────────────
# 7-8. ontology Top-1 aggregation + N preserved
# ─────────────────────────────────────────────────────────────────────────────


def test_ontology_top1_aggregation_and_n(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    rows, order = s2figs.build_ontology_top1(runs)

    assert order == ["HPO", "MONDO"]
    by_key = {(r["mode"], r["ontology"]): r for r in rows}
    assert by_key[("public", "HPO")]["n"] == 3
    assert by_key[("public", "MONDO")]["n"] == 3
    # public: rows 1,2 correct (HPO), row 3 incorrect (HPO) -> HPO top1 = 2/3
    assert by_key[("public", "HPO")]["top1_accuracy"] == pytest.approx(2 / 3)
    # public: row 4 correct (MONDO), rows 5,6 incorrect -> MONDO top1 = 1/3
    assert by_key[("public", "MONDO")]["top1_accuracy"] == pytest.approx(1 / 3)

    total_n_per_mode = {}
    for mode in s2figs.MODES:
        total_n_per_mode[mode] = sum(r["n"] for r in rows if r["mode"] == mode)
    assert all(n == 6 for n in total_n_per_mode.values())


# ─────────────────────────────────────────────────────────────────────────────
# 9-10. paired correctness transitions + sum to paired N
# ─────────────────────────────────────────────────────────────────────────────


def test_paired_transition_matrix_matches_fixture(tmp_path: Path) -> None:
    from llm_ontology_mapper.benchmarking.scenario2_compare import build_paired_predictions

    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    paired = build_paired_predictions(runs)
    matrices = s2figs.build_full_transition_matrices(paired)

    # public: {1,2,4} correct; local: {1,3,4,5} correct.
    # both correct: 1,4 -> 2. public correct/local wrong: 2 -> 1.
    # public wrong/local correct: 3,5 -> 2. both wrong: 6 -> 1.
    m = matrices[("public", "local")]
    assert m == {"both_correct": 2, "a_correct_b_wrong": 1, "a_wrong_b_correct": 2, "both_wrong": 1}


def test_transition_counts_sum_to_paired_n(tmp_path: Path) -> None:
    from llm_ontology_mapper.benchmarking.scenario2_compare import build_paired_predictions

    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    paired = build_paired_predictions(runs)
    matrices = s2figs.build_full_transition_matrices(paired)
    for m in matrices.values():
        assert sum(m.values()) == len(paired) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 11-13. rank/outcome categories: mutually exclusive, sum to N, error before
# abstention
# ─────────────────────────────────────────────────────────────────────────────


def test_outcome_categories_mutually_exclusive_and_sum_to_n(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    rows = s2figs.build_outcome_distribution(runs)
    for mode in s2figs.MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        assert {r["outcome"] for r in mode_rows} == set(s2figs.OUTCOME_CATEGORIES)
        assert sum(r["count"] for r in mode_rows) == 6


def test_error_row_classified_before_abstention() -> None:
    error_row = {"status": "error", "mapped_code": "UNKNOWN:UNMAPPED", "first_gold_rank": ""}
    assert s2figs.classify_row_outcome(error_row) == "Execution error"

    unmapped_row = {"status": "unmapped", "mapped_code": "", "first_gold_rank": ""}
    assert s2figs.classify_row_outcome(unmapped_row) == "Abstained"


def test_disabled_outcome_distribution_has_error_and_abstained(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    rows = s2figs.build_outcome_distribution(runs)
    disabled_rows = {r["outcome"]: r["count"] for r in rows if r["mode"] == "disabled"}
    assert disabled_rows["Execution error"] == 1
    assert disabled_rows["Abstained"] == 1
    assert disabled_rows["Gold rank 1"] == 2  # rows 3, 4


# ─────────────────────────────────────────────────────────────────────────────
# 14-15. reliability diagram consumes existing bins, never recomputes them
# ─────────────────────────────────────────────────────────────────────────────


def test_reliability_diagram_uses_persisted_bins(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    output_dir = tmp_path / "out"
    s2figs.fig_s2d_reliability_diagram(runs, output_dir)
    for ext in ("png", "svg", "pdf"):
        f = output_dir / "main" / f"figure_04_reliability_diagram.{ext}"
        assert f.exists() and f.stat().st_size > 0


def test_scenario2_figures_module_never_recomputes_calibration_bins() -> None:
    import llm_ontology_mapper.benchmarking.figures.scenario2 as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden = ["expected_calibration_error(", "roc_auc(", "brier_score(", "confidence_separation_stats("]
    for token in forbidden:
        assert token not in source, f"{token!r} must not be recomputed -- reuse mode_summary.csv/calibration_bins.csv instead"


# ─────────────────────────────────────────────────────────────────────────────
# 16. validation coverage loaded correctly (never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_coverage_loaded_not_hardcoded(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    mode_summaries = s2figs.load_mode_summaries(runs)
    mode_summaries["local"]["validation_coverage"] = "0.42"  # not 1.0

    output_dir = tmp_path / "out"
    s2figs.fig_s2b_retrieval_behavior(mode_summaries, output_dir)
    with (output_dir / "data" / "scenario2_retrieval_behavior.csv").open(newline="", encoding="utf-8") as fh:
        rows = {r["mode"]: r for r in csv.DictReader(fh)}
    assert rows["local"]["validation_coverage"] == "0.42"
    assert rows["public"]["validation_coverage"] == "1.0"


def test_disabled_grounding_rate_read_not_hardcoded(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    runs = s2figs.load_completed_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    mode_summaries = s2figs.load_mode_summaries(runs)
    assert mode_summaries["disabled"]["grounding_rate"] == "0.0"


# ─────────────────────────────────────────────────────────────────────────────
# 17-19. full build_all smoke test: figures saved PNG/SVG/PDF, data CSVs
# written, captions written
# ─────────────────────────────────────────────────────────────────────────────


def test_build_all_writes_every_figure_data_file_and_caption(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    output_dir = tmp_path / "figures_out"

    result = s2figs.build_all(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir, output_dir=output_dir)
    assert result.output_dir == output_dir

    main_figures = [
        "figure_01_mapping_performance",
        "figure_02_retrieval_behavior",
        "figure_03_calibration_metrics",
        "figure_04_reliability_diagram",
        "figure_05_ontology_top1_heatmap",
        "figure_06_paired_correctness_transitions",
    ]
    supp_figures = [
        "supp_figure_01_rank_outcome_distribution",
        "supp_figure_02_confidence_by_correctness",
        "supp_figure_03_latency_breakdown",
    ]
    for name in main_figures:
        for ext in ("png", "svg", "pdf"):
            f = output_dir / "main" / f"{name}.{ext}"
            assert f.exists() and f.stat().st_size > 0, f"missing {f}"
    for name in supp_figures:
        for ext in ("png", "svg", "pdf"):
            f = output_dir / "supplementary" / f"{name}.{ext}"
            assert f.exists() and f.stat().st_size > 0, f"missing {f}"

    data_files = [
        "scenario2_mapping_performance.csv",
        "scenario2_retrieval_behavior.csv",
        "scenario2_calibration_metrics.csv",
        "ontology_top1_by_mode.csv",
        "paired_correctness_transitions.csv",
        "paired_predictions.csv",
        "scenario2_comparison.csv",
        "scenario2_comparison.md",
        "scenario2_outcome_distribution.csv",
        "scenario2_latency_stage_breakdown_public_local.csv",
        "scenario2_summary_table.csv",
        "scenario2_summary_table.md",
    ]
    for name in data_files:
        f = output_dir / "data" / name
        assert f.exists() and f.stat().st_size > 0, f"missing {f}"

    captions = output_dir / "figure_captions.md"
    assert captions.exists()
    text = captions.read_text(encoding="utf-8")
    for name in main_figures:
        assert name in text


def test_build_all_does_not_modify_original_predictions_csv(tmp_path: Path) -> None:
    public_dir, local_dir, disabled_dir = _make_all_runs(tmp_path)
    before = (public_dir / "predictions.csv").read_text(encoding="utf-8")
    s2figs.build_all(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir, output_dir=tmp_path / "figures_out")
    after = (public_dir / "predictions.csv").read_text(encoding="utf-8")
    assert before == after


# ─────────────────────────────────────────────────────────────────────────────
# 20. zero external/network/LLM calls anywhere in the plotting path
# ─────────────────────────────────────────────────────────────────────────────


def test_figures_modules_have_no_llm_mapper_or_network_imports() -> None:
    import llm_ontology_mapper.benchmarking.figures.common as common_module
    import llm_ontology_mapper.benchmarking.figures.scenario2 as scenario2_module
    import llm_ontology_mapper.benchmarking.figures.style as style_module

    forbidden = [
        "OpenAIProvider",
        "OntologyMapper",
        "PlannedPipeline",
        "OntologyValidator",
        "import openai",
        "import requests",
        "import httpx",
        "urllib.request",
    ]
    for module in (common_module, scenario2_module, style_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token!r} must never appear in {module.__name__}"


def test_cli_script_has_no_llm_mapper_or_network_imports() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_scenario2_results.py"
    source = script_path.read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper", "PlannedPipeline", "OntologyValidator", "import openai"]
    for token in forbidden:
        assert token not in source, f"{token!r} must never appear in the zero-LLM-call plotting CLI"
