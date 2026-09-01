"""
Scenario 2 resume/checkpoint reliability tests (Part 30, items 42-45).

Filesystem-only (tmp_path) -- no network, no mapper, no LLM calls.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.scenario1_runner import (
    ERROR_STAGE_LOCAL_RETRIEVAL as SCENARIO1_ERROR_STAGE_LOCAL_RETRIEVAL,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    SapBertHealthError as Scenario1SapBertHealthError,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    check_sapbert_health as scenario1_check_sapbert_health,
)
from llm_ontology_mapper.benchmarking.scenario2_output import (
    PREDICTIONS_CSV_FIELDS,
    quarantine_error_rows_for_resume,
    read_existing_predictions,
)
from llm_ontology_mapper.benchmarking.scenario2_runner import (
    ERROR_STAGE_LOCAL_RETRIEVAL,
    SapBertHealthError,
    check_sapbert_health,
)

pytestmark = pytest.mark.unit


def _write_predictions_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PREDICTIONS_CSV_FIELDS})


def _row(row_id: int, status: str, **overrides) -> dict:
    base = {"row_id": row_id, "status": status, "source_variable": f"var_{row_id}"}
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 42. mapped/unmapped rows skipped on resume
# ─────────────────────────────────────────────────────────────────────────────


def test_mapped_and_unmapped_rows_are_skipped_on_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run1"
    output_dir.mkdir()
    _write_predictions_csv(
        output_dir / "predictions.csv",
        [_row(1, "mapped"), _row(2, "unmapped"), _row(3, "mapped")],
    )

    skip_ids = quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-01-01T00:00:00Z", provider="openai", model="gpt-5.6-luna"
    )
    assert skip_ids == {1, 2, 3}


# ─────────────────────────────────────────────────────────────────────────────
# 43. error rows quarantined and retried (removed from predictions.csv,
# appended to retry_error_history.csv, and NOT in the skip set)
# ─────────────────────────────────────────────────────────────────────────────


def test_error_rows_quarantined_and_excluded_from_skip_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "run1"
    output_dir.mkdir()
    _write_predictions_csv(
        output_dir / "predictions.csv",
        [
            _row(1, "mapped"),
            _row(2, "error", error_type="RuntimeError", error_stage="pipeline", error_message="boom"),
        ],
    )

    skip_ids = quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-01-01T00:00:00Z", provider="openai", model="gpt-5.6-luna"
    )
    assert skip_ids == {1}  # row 2 (error) must be retried, not skipped

    remaining_rows = read_existing_predictions(output_dir / "predictions.csv")
    assert {int(r["row_id"]) for r in remaining_rows} == {1}

    history_path = output_dir / "retry_error_history.csv"
    assert history_path.exists()
    with history_path.open(newline="", encoding="utf-8") as fh:
        history_rows = list(csv.DictReader(fh))
    assert len(history_rows) == 1
    assert history_rows[0]["row_id"] == "2"
    assert history_rows[0]["previous_status"] == "error"


# ─────────────────────────────────────────────────────────────────────────────
# 44. duplicate row IDs prevented -- after quarantine, row_id 2 is fully
# removed from predictions.csv exactly once (no duplicate rows survive).
# ─────────────────────────────────────────────────────────────────────────────


def test_no_duplicate_row_ids_after_quarantine(tmp_path: Path) -> None:
    output_dir = tmp_path / "run1"
    output_dir.mkdir()
    _write_predictions_csv(
        output_dir / "predictions.csv",
        [_row(1, "mapped"), _row(2, "error"), _row(2, "mapped")],  # simulate a stale duplicate error row
    )
    quarantine_error_rows_for_resume(output_dir, resume_timestamp="t", provider="openai", model="m")
    remaining = read_existing_predictions(output_dir / "predictions.csv")
    row_ids = [int(r["row_id"]) for r in remaining]
    assert row_ids.count(2) == 1  # only the mapped duplicate survives, the error copy is gone
    assert sorted(row_ids) == [1, 2]


def test_quarantine_is_idempotent_with_no_prior_predictions(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty_run"
    output_dir.mkdir()
    skip_ids = quarantine_error_rows_for_resume(output_dir, resume_timestamp="t", provider="openai", model="m")
    assert skip_ids == set()


# ─────────────────────────────────────────────────────────────────────────────
# 45. local outage guard preserved -- Scenario 2 reuses (does not
# reimplement) Scenario 1's SapBERT health-check/error-stage machinery.
# ─────────────────────────────────────────────────────────────────────────────


def test_scenario2_reuses_scenario1_sapbert_health_machinery() -> None:
    assert check_sapbert_health is scenario1_check_sapbert_health
    assert SapBertHealthError is Scenario1SapBertHealthError
    assert ERROR_STAGE_LOCAL_RETRIEVAL == SCENARIO1_ERROR_STAGE_LOCAL_RETRIEVAL == "local_retrieval"
