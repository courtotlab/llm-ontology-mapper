"""
Unit tests for llm_ontology_mapper.benchmarking.output metadata: proves the
benchmark never falsely claims a specific temperature (e.g. 0.0) was used
when the request actually omitted temperature for provider default.

Run with:  pytest tests/benchmarking/test_output.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.output import (
    MODEL_SUMMARY_FIELDS,
    ROW_CSV_FIELDS,
    RUN_SUMMARY_FIELDS,
    build_benchmark_config,
    build_model_summary,
    build_reproducibility_summary,
    build_run_summary,
    row_to_csv_dict,
)
from llm_ontology_mapper.benchmarking.pricing import get_pricing
from llm_ontology_mapper.benchmarking.runner import AlternativeSlot, RowResult

pytestmark = pytest.mark.unit


def _row_result(**overrides: object) -> RowResult:
    defaults: dict[str, object] = dict(
        input_row=1,
        source_variable="var1",
        source_label="label",
        source_description=None,
        target_ontology="HPO",
        gold_code_raw="HP:0000001",
        gold_codes_normalized=["HP:0000001"],
        gold_target_term="term",
        mapped_status="mapped",
        mapped_code="HP:0000001",
        mapped_code_normalized="HP:0000001",
        mapped_term="term",
        mapped_ontology="HPO",
        confidence=0.9,
        logic_type="rag",
        alternatives=[AlternativeSlot(None, None, None, None) for _ in range(4)],
        model="gpt-5.6-luna",
        requested_reasoning_effort="low",
        temperature=None,
        temperature_mode="provider_default",
        seed=42,
        run_number=1,
    )
    defaults.update(overrides)
    return RowResult(**defaults)  # type: ignore[arg-type]


def test_build_benchmark_config_records_provider_default_temperature_not_zero() -> None:
    config = build_benchmark_config(
        input_path=Path("diverse.xlsx"),
        input_file_hash="deadbeef",
        model="gpt-5.6-luna",
        provider="openai",
        runs=2,
        temperature=None,
        seed=42,
        reasoning_effort="low",
        retrieval_mode="public",
        max_alternatives=4,
        pricing=get_pricing("gpt-5.6-luna"),
        repo_dir=Path("."),
        start_timestamp="2026-08-20T00:00:00+00:00",
    )

    assert config["temperature"] is None
    assert config["temperature_mode"] == "provider_default"
    assert config["seed"] == 42
    assert config["reasoning_effort"] == "low"


def test_build_benchmark_config_records_explicit_temperature() -> None:
    config = build_benchmark_config(
        input_path=Path("diverse.xlsx"),
        input_file_hash="deadbeef",
        model="gpt-4.1-mini",
        provider="openai",
        runs=2,
        temperature=0.2,
        seed=42,
        reasoning_effort=None,
        retrieval_mode="public",
        max_alternatives=4,
        pricing=get_pricing("gpt-4.1-mini"),
        repo_dir=Path("."),
        start_timestamp="2026-08-20T00:00:00+00:00",
    )

    assert config["temperature"] == 0.2
    assert config["temperature_mode"] == "explicit"
    assert config["reasoning_effort"] == "N/A"


def test_row_to_csv_dict_includes_temperature_mode_and_never_fabricates_temperature() -> None:
    row = _row_result()
    d = row_to_csv_dict(row)

    assert "temperature_mode" in ROW_CSV_FIELDS
    assert d["temperature"] is None
    assert d["temperature_mode"] == "provider_default"
    assert d["seed"] == 42


def test_row_to_csv_dict_includes_retrieval_diagnostics_fields() -> None:
    retrieval_fields = (
        "retrieval_request_count",
        "retrieval_retry_count",
        "retrieval_recovered_error_count",
        "retrieval_final_error_count",
        "retrieval_error_sources",
        "retrieval_error_types",
    )
    for field in retrieval_fields:
        assert field in ROW_CSV_FIELDS

    row = _row_result(
        retrieval_request_count=3,
        retrieval_retry_count=2,
        retrieval_recovered_error_count=1,
        retrieval_final_error_count=1,
        retrieval_error_sources="OLS:HPO",
        retrieval_error_types="timeout",
    )
    d = row_to_csv_dict(row)

    assert d["retrieval_request_count"] == 3
    assert d["retrieval_retry_count"] == 2
    assert d["retrieval_recovered_error_count"] == 1
    assert d["retrieval_final_error_count"] == 1
    assert d["retrieval_error_sources"] == "OLS:HPO"
    assert d["retrieval_error_types"] == "timeout"


def test_row_to_csv_dict_retrieval_fields_default_to_none() -> None:
    row = _row_result()
    d = row_to_csv_dict(row)

    assert d["retrieval_request_count"] is None
    assert d["retrieval_error_sources"] is None


def test_build_run_summary_includes_retrieval_retry_counters() -> None:
    rows = [
        _row_result(input_row=1, retrieval_retry_count=2, retrieval_final_error_count=0),
        _row_result(input_row=2, retrieval_retry_count=0, retrieval_final_error_count=1),
        _row_result(input_row=3, retrieval_retry_count=None, retrieval_final_error_count=None),
    ]

    summary = build_run_summary(model="gpt-4.1-mini", run_number=1, run_complete=True, rows=rows)

    assert "total_retrieval_retries" in RUN_SUMMARY_FIELDS
    assert summary["total_retrieval_retries"] == 2
    assert summary["rows_with_retrieval_retries"] == 1
    assert summary["rows_with_final_retrieval_errors"] == 1


def test_build_model_summary_sums_retrieval_counters_across_runs() -> None:
    rows1 = [_row_result(input_row=1, retrieval_retry_count=2, retrieval_final_error_count=1)]
    rows2 = [_row_result(input_row=1, retrieval_retry_count=1, retrieval_final_error_count=0)]
    run1_summary = build_run_summary(
        model="gpt-4.1-mini", run_number=1, run_complete=True, rows=rows1
    )
    run1_summary["total_run_seconds"] = 1.0
    run2_summary = build_run_summary(
        model="gpt-4.1-mini", run_number=2, run_complete=True, rows=rows2
    )
    run2_summary["total_run_seconds"] = 1.0
    reproducibility = build_reproducibility_summary(
        model="gpt-4.1-mini",
        rows_run1=rows1,
        rows_run2=rows2,
        run1_summary=run1_summary,
        run2_summary=run2_summary,
    )

    model_summary = build_model_summary(
        model="gpt-4.1-mini",
        run1_summary=run1_summary,
        run2_summary=run2_summary,
        reproducibility=reproducibility,
    )

    assert "total_retrieval_retries" in MODEL_SUMMARY_FIELDS
    assert model_summary["total_retrieval_retries"] == 3
    assert model_summary["rows_with_retrieval_retries"] == 2
    assert model_summary["rows_with_final_retrieval_errors"] == 1
