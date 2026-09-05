"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_output.

Run with:  pytest tests/benchmarking/test_scenario1_output.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.scenario1_metrics import PredictionRecord, score_prediction
from llm_ontology_mapper.benchmarking.scenario1_output import (
    IncrementalPredictionsCsvWriter,
    ResumeConfigMismatchError,
    csv_row_to_prediction_record,
    read_existing_predictions,
    read_manual_review_decisions,
    read_published_baselines,
    row_to_csv_dict,
    validate_resume,
    write_manual_review_required_csv,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import RankSlot, Scenario1RowResult

pytestmark = pytest.mark.unit


def _row(**overrides) -> Scenario1RowResult:
    defaults = dict(
        query_id=1,
        query="headache disorder",
        gold_codes=["EFO:0000001"],
        gold_labels=["Headache disorder"],
        gold_count=1,
        status="mapped",
        mapped_code="EFO:0000001",
        mapped_term="Headache disorder",
        mapped_ontology="EFO",
        confidence=0.9,
        ranks=[RankSlot("EFO:0000001", "Headache disorder", "EFO")] + [RankSlot() for _ in range(4)],
    )
    defaults.update(overrides)
    return Scenario1RowResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 30. incremental persistence
# ─────────────────────────────────────────────────────────────────────────────


def test_incremental_writer_flushes_each_row(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    row = _row()
    rm = score_prediction(
        PredictionRecord(
            query_id=row.query_id, query=row.query, gold_codes=tuple(row.gold_codes), status=row.status, ranks=row.rank_codes
        )
    )
    with IncrementalPredictionsCsvWriter(path) as writer:
        writer.write_row(row_to_csv_dict(row, row_metrics=rm, graph=None))
        # File must already be readable (flushed) before the writer closes.
        assert path.exists()
        rows_mid_write = read_existing_predictions(path)
        assert len(rows_mid_write) == 1

    rows = read_existing_predictions(path)
    assert len(rows) == 1
    assert rows[0]["query_id"] == "1"
    assert rows[0]["rank_1_code"] == "EFO:0000001"


# ─────────────────────────────────────────────────────────────────────────────
# 37. raw predictions sufficient to recompute every metric, no more LLM calls
# ─────────────────────────────────────────────────────────────────────────────


def test_csv_round_trip_preserves_enough_to_rescore(tmp_path: Path) -> None:
    from llm_ontology_mapper.benchmarking.scenario1_metrics import PredictionRecord

    row = _row()
    rm = score_prediction(
        PredictionRecord(
            query_id=row.query_id, query=row.query, gold_codes=tuple(row.gold_codes), status=row.status, ranks=row.rank_codes
        )
    )
    csv_dict = row_to_csv_dict(row, row_metrics=rm, graph=None)

    path = tmp_path / "predictions.csv"
    with IncrementalPredictionsCsvWriter(path) as writer:
        writer.write_row(csv_dict)

    reloaded = read_existing_predictions(path)[0]
    record = csv_row_to_prediction_record(reloaded)
    assert record.query_id == 1
    assert record.gold_codes == ("EFO:0000001",)
    assert record.ranks[0] == "EFO:0000001"

    rescored = score_prediction(record)
    assert rescored.top1_hit is True


# ─────────────────────────────────────────────────────────────────────────────
# csv_row_to_prediction_record()'s gold_codes split strips whitespace
# (UKBB gold-parsing-bug fix, PART 5): a stored gold_codes cell that still
# carries the old compound " | "-joined text (e.g. a pre-fix predictions.csv,
# or the already-completed UKBB query-872 rerun) must reload as clean,
# individually-trimmed codes rather than tokens with stray leading/trailing
# whitespace that can never equal a clean rank code.
# ─────────────────────────────────────────────────────────────────────────────


def test_csv_row_to_prediction_record_strips_whitespace_around_split_gold_codes() -> None:
    record = csv_row_to_prediction_record(
        {
            "query_id": "872",
            "query": "Paraplegia and tetraplegia",
            "gold_codes": "EFO:0009679 | EFO:0009684",
            "status": "unmapped",
            "rank_1_code": "",
            "rank_2_code": "EFO:0009679",
            "rank_3_code": "EFO:0009684",
            "rank_4_code": "HP:0003470",
            "rank_5_code": "HP:0002385",
        }
    )
    assert record.gold_codes == ("EFO:0009679", "EFO:0009684")

    row_metrics = score_prediction(record)
    assert row_metrics.gold_rank == 2
    assert row_metrics.top3_hit is True
    assert row_metrics.top5_hit is True


def test_csv_row_to_prediction_record_gold_codes_no_delimiter_unchanged() -> None:
    record = csv_row_to_prediction_record(
        {"query_id": "1", "query": "q", "gold_codes": "EFO:0000001", "status": "mapped", "rank_1_code": "EFO:0000001"}
    )
    assert record.gold_codes == ("EFO:0000001",)


# ─────────────────────────────────────────────────────────────────────────────
# 31/32/33. resume with identical config / rejected on mismatch / SHA recorded
# ─────────────────────────────────────────────────────────────────────────────


def _fingerprint(**overrides) -> dict:
    base = dict(
        source_dataset_sha256="abc123",
        provider="openai",
        model="gpt-5.6-luna",
        reasoning_effort="low",
        temperature_mode="provider_default",
        temperature=None,
        seed=42,
        target_ontology="EFO",
        retrieval_mode="local",
        strict_target_ontology=False,
        max_alternatives=4,
        sapbert_url="http://localhost:8765",
        llm_ontology_mapper_git_commit="deadbeef",
    )
    base.update(overrides)
    return base


def test_resume_accepts_identical_config() -> None:
    validate_resume(_fingerprint(), _fingerprint())  # must not raise


def test_resume_rejected_on_dataset_sha_mismatch() -> None:
    with pytest.raises(ResumeConfigMismatchError, match="source_dataset_sha256"):
        validate_resume(_fingerprint(), _fingerprint(source_dataset_sha256="different"))


def test_resume_rejected_on_model_mismatch() -> None:
    with pytest.raises(ResumeConfigMismatchError, match="model"):
        validate_resume(_fingerprint(), _fingerprint(model="gpt-5-mini"))


def test_resume_rejected_on_strict_target_ontology_mismatch() -> None:
    with pytest.raises(ResumeConfigMismatchError, match="strict_target_ontology"):
        validate_resume(_fingerprint(), _fingerprint(strict_target_ontology=True))


def test_resume_rejected_on_retrieval_mode_mismatch() -> None:
    with pytest.raises(ResumeConfigMismatchError, match="retrieval_mode"):
        validate_resume(_fingerprint(), _fingerprint(retrieval_mode="public"))


# ─────────────────────────────────────────────────────────────────────────────
# 36. published baseline values are never fabricated
# ─────────────────────────────────────────────────────────────────────────────


def test_read_published_baselines_empty_file_returns_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "published_baselines.csv"
    path.write_text(
        "benchmark,tool,metric,value,denominator,source_publication,source_table_or_figure,notes\n"
    )
    assert read_published_baselines(path) == []


def test_read_published_baselines_missing_file_returns_no_rows(tmp_path: Path) -> None:
    assert read_published_baselines(tmp_path / "nope.csv") == []


# ─────────────────────────────────────────────────────────────────────────────
# 26. manual-review queue: reviewer_decision/reviewer_notes left blank
# ─────────────────────────────────────────────────────────────────────────────


def test_manual_review_csv_leaves_decision_blank(tmp_path: Path) -> None:
    path = tmp_path / "manual_review_required.csv"
    write_manual_review_required_csv(
        [
            {
                "query_id": 1,
                "query": "q",
                "predicted_code": "MONDO:1",
                "predicted_label": "term",
                "predicted_ontology": "MONDO",
                "gold_codes": "EFO:1",
                "gold_labels": "label",
                "graph_relationship": "More Specific",
                "graph_matched_gold_code": "EFO:1",
            }
        ],
        path,
    )
    rows = read_existing_predictions(path)
    assert rows[0]["reviewer_decision"] == ""
    assert rows[0]["reviewer_notes"] == ""


def test_read_manual_review_decisions_ignores_blank_decisions(tmp_path: Path) -> None:
    path = tmp_path / "manual_review_required.csv"
    write_manual_review_required_csv(
        [
            {"query_id": 1, "query": "q1", "predicted_code": "MONDO:1", "predicted_label": "t",
             "predicted_ontology": "MONDO", "gold_codes": "EFO:1", "gold_labels": "l",
             "graph_relationship": "More Specific", "graph_matched_gold_code": "EFO:1"},
            {"query_id": 2, "query": "q2", "predicted_code": "MONDO:2", "predicted_label": "t",
             "predicted_ontology": "MONDO", "gold_codes": "EFO:2", "gold_labels": "l",
             "graph_relationship": "Sibling", "graph_matched_gold_code": "EFO:2",
             "reviewer_decision": "TP-Related"},
        ],
        path,
    )
    decisions = read_manual_review_decisions(path)
    assert decisions == {(2, "MONDO:2"): "TP-Related"}
