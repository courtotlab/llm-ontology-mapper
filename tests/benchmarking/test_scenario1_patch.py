"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_patch (the targeted
Scenario 1 EFO UNMAPPED rerun-and-patch workflow).

No network calls, no real OpenAI/SapBERT calls -- rerun tests use a stub
mapper (MagicMock) exactly like tests/benchmarking/test_scenario1_runner.py.
Patch tests operate entirely on synthetic predictions.csv fixtures in
tmp_path, never on the real outputs/evaluation/ directories.

Run with:  pytest tests/benchmarking/test_scenario1_patch.py -v -m unit
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.benchmarking.scenario1_dataset import CanonicalQuery
from llm_ontology_mapper.benchmarking.scenario1_output import (
    PREDICTIONS_CSV_FIELDS,
    read_existing_predictions,
)
from llm_ontology_mapper.benchmarking import scenario1_patch as scenario1_patch_module
from llm_ontology_mapper.benchmarking.scenario1_patch import (
    DATASET_SPECS,
    DERIVED_SCORING_FIELDS,
    IMMUTABLE_FIELDS,
    MAPPER_OUTPUT_FIELDS,
    PINNED_MAX_ALTERNATIVES,
    PINNED_MAX_CANDIDATES,
    PINNED_MAX_RESULTS_PER_QUERY,
    GoldCorrectionResult,
    PatchResult,
    Scenario1DatasetSpec,
    Scenario1PatchError,
    Scenario1RunConfig,
    build_gold_corrected_predictions,
    build_patched_predictions,
    build_pinned_run_config,
    execute_targeted_rerun,
    extract_unmapped_subset,
    pinned_model_config,
    select_unmapped_rows,
    summarize_patch,
    write_gold_correction_validation_json,
    write_patch_validation_json,
)
from llm_ontology_mapper.benchmarking.scenario1_metrics import classify_tp_taxonomy_row, TP_IDENTICAL, FN
from llm_ontology_mapper.models import AlternativeMapping, LogicType, MappingMetadata, MappingResult

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────


def _base_row(query_id: int, status: str, **overrides: str) -> dict[str, str]:
    row = {f: "" for f in PREDICTIONS_CSV_FIELDS}
    row.update(
        {
            "query_id": str(query_id),
            "query": f"query {query_id}",
            "gold_codes": "EFO:0000001",
            "gold_labels": "Some disorder",
            "gold_count": "1",
            "status": status,
        }
    )
    row.update(overrides)
    return row


def _write_predictions_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_original_run_dir(tmp_path: Path, rows: list[dict[str, str]], name: str = "original_run") -> Path:
    d = tmp_path / name
    d.mkdir()
    _write_predictions_csv(d / "predictions.csv", rows)
    config = {
        "source_dataset_path": "dummy.csv",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "temperature_mode": "provider_default",
        "temperature": None,
        "seed": 42,
        "target_ontology": "EFO",
        "retrieval_mode": "local",
        "strict_target_ontology": False,
        "max_alternatives": 4,
        "sapbert_url": "http://localhost:8765",
        "llm_ontology_mapper_git_commit": "deadbeef",
    }
    (d / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


def _spec(tmp_path: Path, original_run_dir: Path, key: str = "test-efo") -> Scenario1DatasetSpec:
    return Scenario1DatasetSpec(
        key=key,
        label="TEST-EFO",
        original_run_dir=original_run_dir,
        dataset_path=tmp_path / "dummy.csv",
        rerun_output_root=tmp_path / "rerun_root",
        patched_output_root=tmp_path / "patched_root",
        stability_output_root=tmp_path / "stability_root",
        gold_corrected_output_root=tmp_path / "gold_corrected_root",
    )


def _cq(query_id: int, gold_codes: list[str]) -> CanonicalQuery:
    return CanonicalQuery(
        query_id=query_id,
        source_query=f"query {query_id}",
        gold_codes=list(gold_codes),
        gold_labels=[None for _ in gold_codes],
        gold_first_row_indices=[0 for _ in gold_codes],
        original_row_indices=[0],
    )


def _alt(code: str, ontology: str, confidence: float) -> AlternativeMapping:
    return AlternativeMapping(code=code, term=f"term for {code}", ontology=ontology, confidence=confidence)


def _mapped_result(source_term: str, code: str, ontology: str = "EFO") -> MappingResult:
    return MappingResult(
        source_term=source_term,
        target_code=code,
        target_term=f"term for {code}",
        ontology=ontology,
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )


def _unmapped_result_with_alternatives(source_term: str, alternatives: list[AlternativeMapping]) -> MappingResult:
    return MappingResult(
        source_term=source_term,
        target_code="UNKNOWN:UNMAPPED",
        target_term="",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.RAG,
        alternatives=alternatives,
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1/2. status=="unmapped" selection only; mapped/error rows excluded
# ─────────────────────────────────────────────────────────────────────────────


def test_select_unmapped_rows_selects_only_status_unmapped(tmp_path: Path) -> None:
    rows = [_base_row(1, "mapped"), _base_row(2, "unmapped"), _base_row(3, "error"), _base_row(4, "unmapped")]
    d = _make_original_run_dir(tmp_path, rows)
    selected = select_unmapped_rows(d)
    assert {r["query_id"] for r in selected} == {"2", "4"}


def test_extract_unmapped_subset_excludes_mapped_and_error_and_never_touches_predictions_csv(tmp_path: Path) -> None:
    rows = [_base_row(1, "mapped"), _base_row(2, "unmapped"), _base_row(3, "error")]
    d = _make_original_run_dir(tmp_path, rows)
    predictions_path = d / "predictions.csv"
    before = predictions_path.read_bytes()

    spec = _spec(tmp_path, d)
    result = extract_unmapped_subset(spec)

    assert result.query_ids == (2,)
    assert predictions_path.read_bytes() == before  # untouched

    with (d / "unmapped_subset.csv").open(newline="", encoding="utf-8") as fh:
        subset_rows = list(csv.DictReader(fh))
    assert [r["query_id"] for r in subset_rows] == ["2"]


# ─────────────────────────────────────────────────────────────────────────────
# 3/4. query_id uniqueness validation; exact expected subset IDs
# ─────────────────────────────────────────────────────────────────────────────


def test_select_unmapped_rows_rejects_duplicate_query_id(tmp_path: Path) -> None:
    rows = [_base_row(1, "unmapped"), _base_row(1, "unmapped")]
    d = _make_original_run_dir(tmp_path, rows)
    with pytest.raises(Scenario1PatchError, match="Duplicate query_id"):
        select_unmapped_rows(d)


def test_select_unmapped_rows_rejects_empty_query_id(tmp_path: Path) -> None:
    rows = [_base_row(1, "unmapped")]
    rows[0]["query_id"] = ""
    d = _make_original_run_dir(tmp_path, rows)
    with pytest.raises(Scenario1PatchError, match="empty/missing query_id"):
        select_unmapped_rows(d)


def test_extract_unmapped_subset_exact_expected_ids(tmp_path: Path) -> None:
    rows = [_base_row(i, "mapped") for i in range(1, 6)] + [_base_row(6, "unmapped"), _base_row(9, "unmapped")]
    d = _make_original_run_dir(tmp_path, rows)
    spec = _spec(tmp_path, d)
    result = extract_unmapped_subset(spec, expected_count=2)
    assert result.query_ids == (6, 9)


def test_extract_unmapped_subset_count_mismatch_raises_without_override(tmp_path: Path) -> None:
    rows = [_base_row(1, "unmapped")]
    d = _make_original_run_dir(tmp_path, rows)
    spec = _spec(tmp_path, d)
    with pytest.raises(Scenario1PatchError, match="expected 5"):
        extract_unmapped_subset(spec, expected_count=5)
    result = extract_unmapped_subset(spec, expected_count=5, allow_count_mismatch=True)
    assert result.subset_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. rerun restriction uses only targeted IDs
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_targeted_rerun_only_executes_targeted_query_ids(tmp_path: Path) -> None:
    canonical_queries = [_cq(1, ["EFO:0000001"]), _cq(2, ["EFO:0000002"]), _cq(3, ["EFO:0000003"])]
    calls: list[str] = []

    def fake_map_term(*, source_term: str, strict_target_ontology: bool) -> MappingResult:
        calls.append(source_term)
        return _mapped_result(source_term, "EFO:0000002")

    mapper = MagicMock()
    mapper.map_term.side_effect = fake_map_term
    output_dir = tmp_path / "rerun"
    output_dir.mkdir()
    graph_index = MagicMock()
    graph_index.classify.return_value = MagicMock(graph_relationship="Same", graph_matched_gold_code="EFO:0000002")

    outcome = execute_targeted_rerun(
        label="TEST",
        mapper=mapper,
        canonical_queries=canonical_queries,
        targeted_query_ids={2},
        pricing=None,
        graph_index=graph_index,
        output_dir=output_dir,
        append=False,
    )

    assert calls == ["query 2"]
    assert outcome.rows_completed_total == 1
    rows = read_existing_predictions(output_dir / "predictions.csv")
    assert [r["query_id"] for r in rows] == ["2"]


def test_execute_targeted_rerun_rejects_unknown_targeted_id(tmp_path: Path) -> None:
    canonical_queries = [_cq(1, ["EFO:0000001"])]
    mapper = MagicMock()
    output_dir = tmp_path / "rerun"
    output_dir.mkdir()
    with pytest.raises(Scenario1PatchError, match="not present in the current dataset"):
        execute_targeted_rerun(
            label="TEST",
            mapper=mapper,
            canonical_queries=canonical_queries,
            targeted_query_ids={999},
            pricing=None,
            graph_index=MagicMock(),
            output_dir=output_dir,
            append=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 23. resume skips already-completed targeted query_ids
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_targeted_rerun_resume_skips_already_completed_targeted_ids(tmp_path: Path) -> None:
    canonical_queries = [_cq(1, ["EFO:0000001"]), _cq(2, ["EFO:0000002"])]
    calls: list[str] = []

    def fake_map_term(*, source_term: str, strict_target_ontology: bool) -> MappingResult:
        calls.append(source_term)
        return _mapped_result(source_term, "EFO:0000002")

    mapper = MagicMock()
    mapper.map_term.side_effect = fake_map_term
    output_dir = tmp_path / "rerun"
    output_dir.mkdir()

    # query_id=1 already has a row on disk from a prior (interrupted) attempt.
    _write_predictions_csv(output_dir / "predictions.csv", [_base_row(1, "mapped", mapped_code="EFO:0000001")])

    outcome = execute_targeted_rerun(
        label="TEST",
        mapper=mapper,
        canonical_queries=canonical_queries,
        targeted_query_ids={1, 2},
        pricing=None,
        graph_index=MagicMock(),
        output_dir=output_dir,
        append=True,
        already_completed_query_ids={1},
    )

    assert calls == ["query 2"]  # query_id=1 was never re-executed
    assert outcome.rows_completed_this_call == 1
    assert outcome.rows_completed_total == 2
    rows = read_existing_predictions(output_dir / "predictions.csv")
    assert {r["query_id"] for r in rows} == {"1", "2"}


# ─────────────────────────────────────────────────────────────────────────────
# 6/7/8. Scenario1RunConfig pin: 10/10/4, immune to current 15/20 defaults;
# explicit model/retrieval/EFO settings preserved
# ─────────────────────────────────────────────────────────────────────────────


def test_build_pinned_run_config_pins_10_10_4() -> None:
    model_config = pinned_model_config()
    run_config = build_pinned_run_config(model_config=model_config, sapbert_url="http://x")
    assert run_config.max_results_per_query == PINNED_MAX_RESULTS_PER_QUERY == 10
    assert run_config.max_candidates == PINNED_MAX_CANDIDATES == 10
    assert run_config.max_alternatives == PINNED_MAX_ALTERNATIVES == 4


def test_pinned_run_config_does_not_leak_current_application_default() -> None:
    model_config = pinned_model_config()
    # Whatever Scenario1RunConfig's own (currently 15/20) field defaults are,
    # the pinned config must never inherit them.
    default_config = Scenario1RunConfig(model_config=model_config, sapbert_url="http://x")
    pinned_config = build_pinned_run_config(model_config=model_config, sapbert_url="http://x")
    assert pinned_config.max_results_per_query != default_config.max_results_per_query or default_config.max_results_per_query == 10
    assert pinned_config.max_results_per_query == 10
    assert pinned_config.max_candidates == 10


def test_pinned_model_config_preserves_explicit_model_and_reasoning_effort() -> None:
    model_config = pinned_model_config()
    assert model_config.model == "gpt-5.6-luna"
    assert model_config.reasoning_effort == "low"
    assert model_config.is_reasoning is True


def test_execute_targeted_rerun_uses_non_strict_efo_local_pipeline_via_execute_query(tmp_path: Path) -> None:
    """execute_targeted_rerun delegates to scenario1_runner.execute_query()
    (via iter_predictions), which hardcodes STRICT_TARGET_ONTOLOGY=False --
    verify the mapper actually receives strict_target_ontology=False, not a
    value this module invents."""
    canonical_queries = [_cq(1, ["EFO:0000001"])]
    mapper = MagicMock()
    mapper.map_term.return_value = _mapped_result("query 1", "EFO:0000001")
    output_dir = tmp_path / "rerun"
    output_dir.mkdir()

    execute_targeted_rerun(
        label="TEST",
        mapper=mapper,
        canonical_queries=canonical_queries,
        targeted_query_ids={1},
        pricing=None,
        graph_index=MagicMock(),
        output_dir=output_dir,
        append=False,
    )
    _, kwargs = mapper.map_term.call_args
    assert kwargs["strict_target_ontology"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 19/20. Top-5 detects gold in alternatives for status=unmapped; TP-taxonomy
# still treats status=unmapped as FN (scoring contract untouched)
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_row_with_gold_in_alternatives_gets_top5_credit(tmp_path: Path) -> None:
    canonical_queries = [_cq(1, ["EFO:0000042"])]
    mapper = MagicMock()
    mapper.map_term.return_value = _unmapped_result_with_alternatives(
        "query 1",
        [_alt("EFO:0000042", "EFO", 0.4), _alt("EFO:0000099", "EFO", 0.3)],
    )
    output_dir = tmp_path / "rerun"
    output_dir.mkdir()

    execute_targeted_rerun(
        label="TEST",
        mapper=mapper,
        canonical_queries=canonical_queries,
        targeted_query_ids={1},
        pricing=None,
        graph_index=MagicMock(),
        output_dir=output_dir,
        append=False,
    )
    rows = read_existing_predictions(output_dir / "predictions.csv")
    row = rows[0]
    assert row["status"] == "unmapped"
    assert row["rank_2_code"] == "EFO:0000042"  # gold landed in the retained alternatives
    assert row["top5_hit"] == "True"
    assert row["top1_hit"] == "False"


# ─────────────────────────────────────────────────────────────────────────────
# 9/10/11/12. rerun count mismatch / missing / extra / duplicate query_id
# aborts the patch
# ─────────────────────────────────────────────────────────────────────────────


def _default_original_rows() -> list[dict[str, str]]:
    return [
        _base_row(1, "mapped", mapped_code="EFO:0000001", rank_1_code="EFO:0000001", rank_1_ontology="EFO", top1_hit="True"),
        _base_row(2, "unmapped"),
        _base_row(3, "unmapped"),
    ]


def test_patch_aborts_on_missing_rerun_query_id(tmp_path: Path) -> None:
    original_dir = _make_original_run_dir(tmp_path, _default_original_rows())
    spec = _spec(tmp_path, original_dir)
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    # Only reruns query_id=2, missing query_id=3.
    _write_predictions_csv(rerun_dir / "predictions.csv", [_base_row(2, "unmapped")])

    with pytest.raises(Scenario1PatchError, match="missing="):
        build_patched_predictions(spec, rerun_dir, tmp_path / "patched")


def test_patch_aborts_on_unexpected_rerun_query_id(tmp_path: Path) -> None:
    original_dir = _make_original_run_dir(tmp_path, _default_original_rows())
    spec = _spec(tmp_path, original_dir)
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    # Reruns query_id=1, which was NOT originally unmapped.
    _write_predictions_csv(
        rerun_dir / "predictions.csv",
        [_base_row(2, "mapped"), _base_row(3, "mapped"), _base_row(1, "mapped")],
    )

    with pytest.raises(Scenario1PatchError, match="unexpected="):
        build_patched_predictions(spec, rerun_dir, tmp_path / "patched")


def test_patch_aborts_on_duplicate_rerun_query_id(tmp_path: Path) -> None:
    original_dir = _make_original_run_dir(tmp_path, _default_original_rows())
    spec = _spec(tmp_path, original_dir)
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    _write_predictions_csv(
        rerun_dir / "predictions.csv",
        [_base_row(2, "mapped"), _base_row(2, "mapped"), _base_row(3, "mapped")],
    )

    with pytest.raises(Scenario1PatchError, match="duplicates="):
        build_patched_predictions(spec, rerun_dir, tmp_path / "patched")


# ─────────────────────────────────────────────────────────────────────────────
# 13/14/15/16/17/18/21/24. patch replaces by query_id, immutable columns
# untouched, mapper-output replaced for targets only, non-target rows
# unchanged, row count preserved, scoring recomputed, originals never
# modified, no gold-dependent keep/revert logic
# ─────────────────────────────────────────────────────────────────────────────


def _full_patch_fixture(tmp_path: Path) -> tuple[Scenario1DatasetSpec, Path, Path, bytes]:
    original_rows = [
        _base_row(
            1, "mapped", mapped_code="EFO:0000001", mapped_ontology="EFO",
            rank_1_code="EFO:0000001", rank_1_ontology="EFO",
            first_gold_rank="1", top1_hit="True", top3_hit="True", top5_hit="True", reciprocal_rank="1.0",
        ),
        _base_row(2, "unmapped"),  # will be rerun -> gold in alt rank 3
        _base_row(3, "unmapped", gold_codes="EFO:0000999"),  # will be rerun -> stays unmapped, no gold anywhere
    ]
    original_dir = _make_original_run_dir(tmp_path, original_rows)
    predictions_before = (original_dir / "predictions.csv").read_bytes()
    spec = _spec(tmp_path, original_dir)

    rerun_rows = [
        _base_row(
            2, "unmapped", gold_codes="EFO:0000001",
            rank_2_code="EFO:0000001", rank_2_ontology="EFO",
            api_cost_usd="0.001", confidence="0.0",
        ),
        _base_row(3, "unmapped", gold_codes="EFO:0000999", api_cost_usd="0.002"),
    ]
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    _write_predictions_csv(rerun_dir / "predictions.csv", rerun_rows)

    return spec, original_dir, rerun_dir, predictions_before


def test_patch_replaces_by_query_id_not_row_position(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    rows = {r["query_id"]: r for r in read_existing_predictions(result.patched_dir / "predictions.csv")}
    assert rows["2"]["rank_2_code"] == "EFO:0000001"
    assert rows["3"]["status"] == "unmapped"


def test_patch_immutable_columns_never_change_for_non_target_rows(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    original_row = {r["query_id"]: r for r in read_existing_predictions(original_dir / "predictions.csv")}["1"]
    patched_row = {r["query_id"]: r for r in read_existing_predictions(result.patched_dir / "predictions.csv")}["1"]
    for f in IMMUTABLE_FIELDS:
        assert patched_row[f] == original_row[f], f"immutable field {f} changed for non-target row"


def test_patch_targeted_mapper_output_columns_replaced(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    patched_row = {r["query_id"]: r for r in read_existing_predictions(result.patched_dir / "predictions.csv")}["2"]
    assert patched_row["rank_2_code"] == "EFO:0000001"
    assert patched_row["api_cost_usd"] == "0.001"
    # gold_codes came along in the rerun row for query_id=2 too, but since it
    # is an IMMUTABLE field it must be sourced from the ORIGINAL, not the rerun.
    assert patched_row["gold_codes"] == "EFO:0000001"  # matches original row's gold, not coincidentally different


def test_patch_non_target_mapper_output_rows_byte_identical(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    original_row = {r["query_id"]: r for r in read_existing_predictions(original_dir / "predictions.csv")}["1"]
    patched_row = {r["query_id"]: r for r in read_existing_predictions(result.patched_dir / "predictions.csv")}["1"]
    for f in MAPPER_OUTPUT_FIELDS:
        assert patched_row[f] == original_row[f], f"mapper-output field {f} changed for non-target row 1"
    assert result.non_target_row_mismatches == ()


def test_patch_row_count_equals_original_row_count(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    assert result.patched_row_count == result.original_row_count == 3
    assert result.replaced_query_ids_count == result.targeted_row_count == 2


def test_patch_recomputes_derived_scoring_from_patched_mapper_output(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    rows = {r["query_id"]: r for r in read_existing_predictions(result.patched_dir / "predictions.csv")}
    # query_id=2's rerun put gold at rank_2 -> derived scoring must reflect that.
    assert rows["2"]["first_gold_rank"] == "2"
    assert rows["2"]["top1_hit"] == "False"
    assert rows["2"]["top3_hit"] == "True"
    assert rows["2"]["top5_hit"] == "True"
    # query_id=3 has no gold anywhere in ranks -> stays a miss.
    assert rows["3"]["first_gold_rank"] == ""
    assert rows["3"]["top5_hit"] == "False"


def test_patch_never_modifies_original_run_directory(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, predictions_before = _full_patch_fixture(tmp_path)
    build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    assert (original_dir / "predictions.csv").read_bytes() == predictions_before


def test_patch_result_passed_true_for_a_clean_patch(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    assert result.passed is True
    assert result.missing_query_ids == ()
    assert result.unexpected_query_ids == ()
    assert result.duplicate_query_ids == ()
    assert result.immutable_column_mismatches == ()
    assert result.non_target_row_mismatches == ()


def test_no_gold_dependent_keep_or_revert_logic(tmp_path: Path) -> None:
    """Rerun rows are written into the patch unconditionally, regardless of
    whether the new result contains the gold code, matches it, or is worse
    than the original -- build_patched_predictions never reads gold_codes to
    decide replace-vs-keep, only to (re)compute reporting-only derived
    scoring after the unconditional replacement."""
    original_rows = [_base_row(1, "unmapped", gold_codes="EFO:0000001")]
    original_dir = _make_original_run_dir(tmp_path, original_rows)
    spec = _spec(tmp_path, original_dir)
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    # Rerun result is WORSE (still unmapped, gold nowhere in ranks) -- must
    # still unconditionally replace, never silently keep the original row.
    _write_predictions_csv(
        rerun_dir / "predictions.csv", [_base_row(1, "unmapped", gold_codes="EFO:0000001", api_cost_usd="0.005")]
    )
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    patched_row = read_existing_predictions(result.patched_dir / "predictions.csv")[0]
    assert patched_row["api_cost_usd"] == "0.005"  # rerun value written, not silently reverted
    assert result.replaced_query_ids_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 22. patch_validation.json correctly reports validation
# ─────────────────────────────────────────────────────────────────────────────


def test_patch_validation_json_reports_pass(tmp_path: Path) -> None:
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    result = build_patched_predictions(spec, rerun_dir, tmp_path / "patched")
    path = write_patch_validation_json(result, spec, rerun_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["validation_passed"] is True
    assert payload["original_row_count"] == 3
    assert payload["patched_row_count"] == 3
    assert payload["replaced_query_ids_count"] == 2
    assert payload["missing_query_ids"] == []
    assert payload["unexpected_query_ids"] == []
    assert payload["duplicate_query_ids"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Column-group partition sanity (schema audit, not hand-maintained)
# ─────────────────────────────────────────────────────────────────────────────


def test_summarize_patch_handles_gold_absent_from_all_ranks(tmp_path: Path) -> None:
    """Regression test: gold_rank_counts["absent"] must be incremented, not
    looked up under the literal key "None" (a prior implementation did
    `gold_rank_counts[str(rank)] += 1 if rank is not None else 0`, which
    still evaluates the str(None) == "None" subscript and raised KeyError
    for any rerun row where gold never appears in ranks 1-5)."""
    spec, original_dir, rerun_dir, _ = _full_patch_fixture(tmp_path)
    patched_dir = tmp_path / "patched"
    build_patched_predictions(spec, rerun_dir, patched_dir)
    summary = summarize_patch(spec, rerun_dir, patched_dir)
    # query_id=2 -> gold at rank 2; query_id=3 -> gold absent from all ranks.
    assert summary["gold_rank_among_rerun_rows"]["2"] == 1
    assert summary["gold_rank_among_rerun_rows"]["absent"] == 1


def test_column_groups_partition_predictions_csv_fields_exactly() -> None:
    covered = set(IMMUTABLE_FIELDS) | set(DERIVED_SCORING_FIELDS) | set(MAPPER_OUTPUT_FIELDS)
    assert covered == set(PREDICTIONS_CSV_FIELDS)
    assert len(covered) == len(IMMUTABLE_FIELDS) + len(DERIVED_SCORING_FIELDS) + len(MAPPER_OUTPUT_FIELDS)


# ─────────────────────────────────────────────────────────────────────────────
# UKBB gold-parsing-bug fix: _split_pipe() whitespace hardening (PART 5)
# ─────────────────────────────────────────────────────────────────────────────


def test_split_pipe_strips_whitespace_around_tokens() -> None:
    assert scenario1_patch_module._split_pipe("EFO:0009679 | EFO:0009684") == ("EFO:0009679", "EFO:0009684")
    assert scenario1_patch_module._split_pipe("EFO:0000001|EFO:0000002") == ("EFO:0000001", "EFO:0000002")
    assert scenario1_patch_module._split_pipe("") == ()


# ─────────────────────────────────────────────────────────────────────────────
# Gold-metadata correction: build_gold_corrected_predictions() (PARTS 3/6/7)
# ─────────────────────────────────────────────────────────────────────────────


def _write_dataset_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["query", "ref_match", "ref_match_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gold_correction_fixture(tmp_path: Path) -> tuple[Scenario1DatasetSpec, Path, bytes, bytes]:
    """One clean single-gold query (query_id 0) and one UKBB-872-style
    compound-cell query (query_id 1) whose ORIGINAL predictions.csv still
    carries the pre-fix, unsplit gold text -- exactly the real UKBB state."""
    dataset_path = tmp_path / "dataset.csv"
    _write_dataset_csv(
        dataset_path,
        [
            {"query": "single query", "ref_match": "single label", "ref_match_id": "EFO:0000001"},
            {"query": "compound query", "ref_match": "label a||label b", "ref_match_id": "EFO:0000002 | EFO:0000003"},
        ],
    )

    original_rows = [
        _base_row(
            0, "mapped", query="single query", gold_codes="EFO:0000001", gold_labels="single label", gold_count="1",
            mapped_code="EFO:0000001", mapped_ontology="EFO", rank_1_code="EFO:0000001", rank_1_ontology="EFO",
            first_gold_rank="1", top1_hit="True", top3_hit="True", top5_hit="True", reciprocal_rank="1.0",
        ),
        _base_row(
            1, "mapped", query="compound query",
            gold_codes="EFO:0000002 | EFO:0000003",  # pre-fix, unsplit
            gold_labels="label a||label b",  # pre-fix, unsplit
            gold_count="1",  # pre-fix, undercounted
            mapped_code="EFO:0000003", mapped_ontology="EFO", rank_1_code="EFO:0000003", rank_1_ontology="EFO",
        ),
    ]
    original_dir = _make_original_run_dir(tmp_path, original_rows, name="original")
    predictions_before = (original_dir / "predictions.csv").read_bytes()
    config_before = (original_dir / "experiment_config.json").read_bytes()

    spec = Scenario1DatasetSpec(
        key="gold-fix-test", label="GOLD-FIX-TEST", original_run_dir=original_dir, dataset_path=dataset_path,
        rerun_output_root=tmp_path / "rerun_root", patched_output_root=tmp_path / "patched_root",
        stability_output_root=tmp_path / "stability_root", gold_corrected_output_root=tmp_path / "gold_corrected_root",
    )
    return spec, original_dir, predictions_before, config_before


def test_gold_correction_never_mutates_original_directory(tmp_path: Path) -> None:
    spec, original_dir, predictions_before, config_before = _gold_correction_fixture(tmp_path)
    build_gold_corrected_predictions(spec, tmp_path / "corrected")
    assert (original_dir / "predictions.csv").read_bytes() == predictions_before
    assert (original_dir / "experiment_config.json").read_bytes() == config_before


def test_gold_correction_mapper_output_fields_byte_identical(tmp_path: Path) -> None:
    spec, original_dir, _, _ = _gold_correction_fixture(tmp_path)
    corrected_dir = tmp_path / "corrected"
    result = build_gold_corrected_predictions(spec, corrected_dir)
    assert result.mapper_output_mismatch_count == 0

    original_rows = {r["query_id"]: r for r in read_existing_predictions(original_dir / "predictions.csv")}
    corrected_rows = {r["query_id"]: r for r in read_existing_predictions(corrected_dir / "predictions.csv")}
    for qid in ("0", "1"):
        for f in MAPPER_OUTPUT_FIELDS:
            assert corrected_rows[qid][f] == original_rows[qid][f]
    assert corrected_rows["1"]["status"] == "mapped"
    assert corrected_rows["1"]["mapped_code"] == "EFO:0000003"
    assert corrected_rows["1"]["rank_1_code"] == "EFO:0000003"


def test_gold_correction_only_gold_metadata_and_derived_scoring_change(tmp_path: Path) -> None:
    spec, original_dir, _, _ = _gold_correction_fixture(tmp_path)
    corrected_dir = tmp_path / "corrected"
    build_gold_corrected_predictions(spec, corrected_dir)

    original = {r["query_id"]: r for r in read_existing_predictions(original_dir / "predictions.csv")}
    corrected = {r["query_id"]: r for r in read_existing_predictions(corrected_dir / "predictions.csv")}

    # unaffected row: immutable + mapper-output fields never change (derived
    # scoring is always freshly recomputed for every row by design -- see
    # module note on build_gold_corrected_predictions -- so it is not
    # compared for byte-identity here, only checked for correctness below).
    for f in (*IMMUTABLE_FIELDS, *MAPPER_OUTPUT_FIELDS):
        assert corrected["0"][f] == original["0"][f], f"unexpected change in unaffected row, field {f}"
    assert corrected["0"]["top1_hit"] == "True"  # recomputed, and correctly still a hit

    # affected row: immutable query_id/query untouched
    assert corrected["1"]["query_id"] == original["1"]["query_id"]
    assert corrected["1"]["query"] == original["1"]["query"]
    # mapper-output untouched
    for f in MAPPER_OUTPUT_FIELDS:
        assert corrected["1"][f] == original["1"][f]
    # gold metadata corrected
    assert corrected["1"]["gold_codes"] == "EFO:0000002|EFO:0000003"
    assert corrected["1"]["gold_labels"] == "label a|label b"
    assert corrected["1"]["gold_count"] == "2"
    # derived scoring recomputed: predicted EFO:0000003 now recognized as gold
    assert corrected["1"]["top1_hit"] == "True"
    assert corrected["1"]["first_gold_rank"] == "1"


def test_gold_correction_query_ids_remain_identical_and_unique(tmp_path: Path) -> None:
    spec, original_dir, _, _ = _gold_correction_fixture(tmp_path)
    corrected_dir = tmp_path / "corrected"
    result = build_gold_corrected_predictions(spec, corrected_dir)
    assert result.missing_query_ids == ()
    assert result.unexpected_query_ids == ()
    assert result.duplicate_query_ids == ()
    assert result.original_row_count == result.corrected_row_count == 2
    assert result.passed is True
    assert result.affected_query_ids == (1,)


def test_gold_correction_calls_no_mapper_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _, _, _ = _gold_correction_fixture(tmp_path)

    def _fail(*args, **kwargs):
        raise AssertionError("gold correction must never call mapper/provider/retrieval functions")

    monkeypatch.setattr(scenario1_patch_module, "build_mapper", _fail)
    monkeypatch.setattr(scenario1_patch_module, "build_provider", _fail)
    monkeypatch.setattr(scenario1_patch_module, "check_sapbert_health", _fail)
    monkeypatch.setattr(scenario1_patch_module, "run_preflight", _fail)

    result = build_gold_corrected_predictions(spec, tmp_path / "corrected")
    assert result.passed is True  # completed without ever touching the monkeypatched (raising) functions


def test_gold_correction_validation_json_contents(tmp_path: Path) -> None:
    spec, _, _, _ = _gold_correction_fixture(tmp_path)
    corrected_dir = tmp_path / "corrected"
    result = build_gold_corrected_predictions(spec, corrected_dir)
    path = write_gold_correction_validation_json(result, spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["affected_query_ids"] == [1]
    assert payload["affected_row_count"] == 1
    assert payload["mapper_output_mismatch_count"] == 0
    assert payload["validation_passed"] is True
    assert payload["original_row_count"] == payload["corrected_row_count"] == 2
    assert len(payload["original_predictions_sha256"]) == 64  # sha256 hex digest


# ─────────────────────────────────────────────────────────────────────────────
# TP-taxonomy regression (PART 8/9): exact split-gold Top-1 becomes
# TP-Identical; a still-unmapped row (query 872) remains FN
# ─────────────────────────────────────────────────────────────────────────────


def test_exact_split_gold_top1_becomes_tp_identical() -> None:
    """806/842/868-style case: rank_1 exactly equals ONE of the two split
    gold codes -- derived from the real classify_tp_taxonomy_row(), not
    hard-coded."""
    row = classify_tp_taxonomy_row(
        query_id=806,
        status="mapped",
        rank1_code="EFO:0009641",
        gold_codes=("HP:0012378", "EFO:0009641"),
        graph_relationship="Same",  # EfoGraphIndex.classify() returns "Same" whenever rank1 in gold_codes
    )
    assert row.category == TP_IDENTICAL


def test_query_872_remains_fn_because_status_is_unmapped() -> None:
    row = classify_tp_taxonomy_row(
        query_id=872,
        status="unmapped",
        rank1_code=None,
        gold_codes=("EFO:0009679", "EFO:0009684"),
        graph_relationship=None,
    )
    assert row.category == FN


# ─────────────────────────────────────────────────────────────────────────────
# original_selection_run vs. baseline_run vs. rerun_run (PART 10)
# ─────────────────────────────────────────────────────────────────────────────


def test_patch_distinguishes_original_selection_baseline_and_rerun_runs(tmp_path: Path) -> None:
    # original SELECTION run: determines the targeted (unmapped) set, still
    # carrying the pre-fix compound gold text (as the real UKBB run does).
    selection_rows = [
        _base_row(1, "mapped", mapped_code="EFO:0000001", rank_1_code="EFO:0000001", rank_1_ontology="EFO"),
        _base_row(2, "unmapped", gold_codes="EFO:0000010 | EFO:0000011", gold_labels="a||b", gold_count="1"),
    ]
    selection_dir = _make_original_run_dir(tmp_path, selection_rows, name="selection")

    # corrected BASELINE run: same rows, gold metadata corrected, mapper
    # output for query_id=1 untouched.
    baseline_rows = [
        _base_row(1, "mapped", mapped_code="EFO:0000001", rank_1_code="EFO:0000001", rank_1_ontology="EFO"),
        _base_row(2, "unmapped", gold_codes="EFO:0000010|EFO:0000011", gold_labels="a|b", gold_count="2"),
    ]
    baseline_dir = _make_original_run_dir(tmp_path, baseline_rows, name="baseline")

    # rerun: the ALREADY-COMPLETED mapper rerun for query_id=2, which ran
    # BEFORE the gold fix existed -- its own gold_codes column still carries
    # the old compound text (exactly the real UKBB query-872 situation).
    rerun_rows = [
        _base_row(
            2, "unmapped", gold_codes="EFO:0000010 | EFO:0000011", gold_labels="a||b", gold_count="1",
            rank_2_code="EFO:0000010", rank_2_ontology="EFO",
        ),
    ]
    rerun_dir = tmp_path / "rerun"
    rerun_dir.mkdir()
    _write_predictions_csv(rerun_dir / "predictions.csv", rerun_rows)

    spec = Scenario1DatasetSpec(
        key="distinguish-test", label="DISTINGUISH-TEST", original_run_dir=selection_dir,
        dataset_path=tmp_path / "unused.csv", rerun_output_root=tmp_path / "rerun_root",
        patched_output_root=tmp_path / "patched_root", stability_output_root=tmp_path / "stability_root",
        gold_corrected_output_root=tmp_path / "gold_corrected_root",
    )

    result = build_patched_predictions(
        spec, rerun_dir, tmp_path / "patched",
        original_selection_run=selection_dir, baseline_run=baseline_dir,
    )

    assert result.original_selection_run == selection_dir
    assert result.baseline_run == baseline_dir
    assert result.passed is True
    # Targeting decision came from the SELECTION run (query_id=2 was unmapped there).
    assert result.targeted_row_count == 1
    assert result.replaced_query_ids_count == 1
    # No false immutable mismatch was raised even though baseline/rerun gold text differs textually.
    assert result.immutable_column_mismatches == ()

    patched_rows = {r["query_id"]: r for r in read_existing_predictions((tmp_path / "patched") / "predictions.csv")}
    # Immutable/gold metadata for query_id=2 came from the CORRECTED BASELINE, not the rerun.
    assert patched_rows["2"]["gold_codes"] == "EFO:0000010|EFO:0000011"
    assert patched_rows["2"]["gold_count"] == "2"
    # Mapper output for query_id=2 came from the RERUN.
    assert patched_rows["2"]["rank_2_code"] == "EFO:0000010"
    # Non-target row (query_id=1) came from the baseline untouched.
    assert patched_rows["1"]["mapped_code"] == "EFO:0000001"


# ─────────────────────────────────────────────────────────────────────────────
# Real-data check: the actual UKBB targeted rerun set is exactly {80, 416, 574, 872}
# ─────────────────────────────────────────────────────────────────────────────


def test_real_ukbb_unmapped_targeted_set_is_80_416_574_872() -> None:
    ukbb_spec = DATASET_SPECS["ukbb-efo"]
    predictions_path = ukbb_spec.original_run_dir / "predictions.csv"
    if not predictions_path.exists():
        pytest.skip("real UKBB original run not present in this checkout")
    unmapped_rows = select_unmapped_rows(ukbb_spec.original_run_dir)
    query_ids = {int(r["query_id"]) for r in unmapped_rows}
    assert query_ids == {80, 416, 574, 872}
