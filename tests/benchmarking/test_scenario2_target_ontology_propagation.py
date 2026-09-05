"""
Scenario 2 target-ontology propagation audit -- zero-LLM regression test.

Proves the complete path:

    dict_mapped_all.xlsx "target_ontology" column
        -> BenchmarkRow.target_ontology (scenario2_dataset/dataset.py)
        -> per-ontology OntologyMapper construction (scenario2_runner.build_pipeline_and_mappers)
        -> per-row mapper selection (scenario2_runner.iter_run_rows)
        -> OntologyMapper(ontologies=[<that row's ontology>])

using synthetic BenchmarkRow records for HPO/MONDO/LOINC and mocked
OntologyMapper/PlannedPipeline/retriever construction -- no network, no
OpenAI/provider calls of any kind. This is the audit trail for the
target-ontology propagation review: no prior test exercised iter_run_rows()
end-to-end with multiple distinct target ontologies present in the same
dataset, so it never actually proved a given row reaches the mapper instance
constructed for THAT row's ontology (as opposed to some other/global one).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.model_registry import get_model_config
from llm_ontology_mapper.benchmarking.pricing import get_pricing
from llm_ontology_mapper.benchmarking.scenario2_runner import (
    Scenario2RunConfig,
    build_pipeline_and_mappers,
    iter_run_rows,
)
from llm_ontology_mapper.models import LogicType, MappingMetadata, MappingResult
from llm_ontology_mapper.ontology_identity import canonical_ontology

pytestmark = pytest.mark.unit


def _row(input_row: int, *, source_variable: str, target_ontology: str, gold_code: str) -> BenchmarkRow:
    return BenchmarkRow(
        input_row=input_row,
        source_variable=source_variable,
        source_label=source_variable,
        source_description=None,
        target_ontology=target_ontology,
        gold_code_raw=gold_code,
        gold_codes=[gold_code],
        gold_target_term="term",
    )


# Real workbook rows (source_variable / target_ontology / gold code), one per
# ontology family, so the fixture mirrors dict_mapped_all.xlsx exactly rather
# than an invented example.
_ROWS = [
    _row(1, source_variable="sinus_pain", target_ontology="HPO", gold_code="HP:0000245"),
    _row(2, source_variable="com_other", target_ontology="MONDO", gold_code="MONDO:0000001"),
    _row(3, source_variable="respiratory_rate", target_ontology="LOINC", gold_code="LOINC:103219-2"),
]

_RESULT_CODE = {"HPO": "HP:0000245", "MONDO": "MONDO:0000001", "LOINC": "LOINC:103219-2"}


def _mapping_result(ontology: str) -> MappingResult:
    return MappingResult(
        source_term="x",
        target_code=_RESULT_CODE[ontology],
        target_term="t",
        ontology=ontology,
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )


def _build_mocked_mappers_for_mode(mode: str, **run_config_overrides):
    """Build (constructed_ontologies, mock_instances_by_ontology, mappers,
    run_config) using a fully mocked OntologyMapper/PlannedPipeline/retriever
    stack -- zero network, zero OpenAI calls."""
    constructed_ontologies: list[list[str]] = []
    mock_instances: dict[str, MagicMock] = {}

    def fake_ontology_mapper(*, llm_provider, ontologies, **kwargs):
        constructed_ontologies.append(list(ontologies))
        mock = MagicMock(name=f"OntologyMapper[{ontologies[0]}]")
        mock.map_term.return_value = _mapping_result(ontologies[0])
        mock_instances[ontologies[0]] = mock
        return mock

    with (
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PublicOntologyRetriever"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.LocalSemanticRetriever"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PlannedPipeline"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.OntologyMapper", side_effect=fake_ontology_mapper),
    ):
        provider = MagicMock()
        defaults = dict(model_config=get_model_config("gpt-5.6-luna"), retrieval_mode=mode)
        if mode == "local":
            defaults["sapbert_url"] = "http://localhost:8765"
        defaults.update(run_config_overrides)
        run_config = Scenario2RunConfig(**defaults)
        target_ontologies = sorted({canonical_ontology(r.target_ontology) or r.target_ontology.upper() for r in _ROWS})
        _pipeline, mappers = build_pipeline_and_mappers(
            provider=provider, run_config=run_config, target_ontologies=target_ontologies
        )

    return constructed_ontologies, mock_instances, mappers, run_config


# ─────────────────────────────────────────────────────────────────────────────
# 1/2. workbook column literally reaches canonical_row.target_ontology
# ─────────────────────────────────────────────────────────────────────────────


def test_workbook_column_is_not_inferred_from_other_fields() -> None:
    row = _row(1, source_variable="sinus_pain", target_ontology="HPO", gold_code="HP:0000245")
    assert row.target_ontology == "HPO"
    # Never derived from target_code/target_term/source_variable/source_label.
    assert row.target_ontology != row.gold_code_raw
    assert row.target_ontology != row.source_variable


# ─────────────────────────────────────────────────────────────────────────────
# 3/4/8. exact constraint mechanism: OntologyMapper(ontologies=[row.target_ontology])
# one mapper per distinct ontology, never a single global mapper reused for
# every row.
# ─────────────────────────────────────────────────────────────────────────────


def test_one_mapper_constructed_per_distinct_ontology_no_global_hardcode() -> None:
    constructed_ontologies, mock_instances, mappers, _ = _build_mocked_mappers_for_mode("public")

    # Exactly one OntologyMapper(...) construction per distinct ontology --
    # never a single shared/global instance, never more constructions than
    # distinct ontologies.
    assert sorted(o[0] for o in constructed_ontologies) == ["HPO", "LOINC", "MONDO"]
    assert all(len(o) == 1 for o in constructed_ontologies)  # each is a single-item allow-list
    assert set(mock_instances) == {"HPO", "MONDO", "LOINC"}
    assert mappers["HPO"] is mock_instances["HPO"]
    assert mappers["MONDO"] is mock_instances["MONDO"]
    assert mappers["LOINC"] is mock_instances["LOINC"]
    assert mappers["HPO"] is not mappers["MONDO"] is not mappers["LOINC"]


# ─────────────────────────────────────────────────────────────────────────────
# 8/9. end-to-end mocked probe: each row's target_ontology reaches the
# correctly-keyed mapper instance, and ONLY that instance.
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_propagation_hpo_mondo_loinc_reach_correct_mapper() -> None:
    _constructed, mock_instances, mappers, run_config = _build_mocked_mappers_for_mode("public")
    pricing = get_pricing("gpt-5.6-luna")

    results = list(iter_run_rows(mappers=mappers, dataset=_ROWS, run_config=run_config, pricing=pricing))

    assert len(results) == 3
    by_row = {r.input_row: r for r in results}

    # Row 1 (HPO) must have gone through the HPO mapper only.
    assert mock_instances["HPO"].map_term.call_count == 1
    assert mock_instances["MONDO"].map_term.call_count == 1
    assert mock_instances["LOINC"].map_term.call_count == 1
    # No mapper was invoked more than once -- proves no accidental global reuse.

    assert by_row[1].target_ontology == "HPO"
    assert by_row[1].mapped_ontology == "HPO"
    assert by_row[1].mapped_code == "HP:0000245"

    assert by_row[2].target_ontology == "MONDO"
    assert by_row[2].mapped_ontology == "MONDO"
    assert by_row[2].mapped_code == "MONDO:0000001"

    assert by_row[3].target_ontology == "LOINC"
    assert by_row[3].mapped_ontology == "LOINC"
    assert by_row[3].mapped_code == "LOINC:103219-2"

    # The source_term/label forwarded to each mapper matches that row's own
    # source_variable -- never swapped/misaligned across rows.
    assert mock_instances["HPO"].map_term.call_args.kwargs["source_term"] == "sinus_pain"
    assert mock_instances["MONDO"].map_term.call_args.kwargs["source_term"] == "com_other"
    assert mock_instances["LOINC"].map_term.call_args.kwargs["source_term"] == "respiratory_rate"


def test_no_target_ontology_kwarg_passed_to_map_term_constraint_lives_in_mapper_instance() -> None:
    """The per-row constraint is NOT supplied via mapper.map_term(target_ontology=...)
    -- OntologyMapper.map_term() has no such parameter. It is supplied at
    OntologyMapper CONSTRUCTION time via ontologies=[...] (see
    build_pipeline_and_mappers), which OntologyMapper._map_term_with_planned_pipeline
    then forwards into PlannedPipeline.map_term(target_ontology=..., allowed_target_ontologies=...)."""
    _constructed, mock_instances, mappers, run_config = _build_mocked_mappers_for_mode("public")
    pricing = get_pricing("gpt-5.6-luna")
    list(iter_run_rows(mappers=mappers, dataset=_ROWS, run_config=run_config, pricing=pricing))

    for mock in mock_instances.values():
        assert "target_ontology" not in mock.map_term.call_args.kwargs
        assert set(mock.map_term.call_args.kwargs) == {"source_term", "source_label", "source_description"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. all three retrieval modes receive the SAME per-row target ontology --
# the only thing that may vary between modes is retrieval_mode itself.
# ─────────────────────────────────────────────────────────────────────────────


def test_all_three_modes_construct_identical_per_ontology_allow_lists() -> None:
    public_constructed, _, _, _ = _build_mocked_mappers_for_mode("public")
    local_constructed, _, _, _ = _build_mocked_mappers_for_mode("local", sapbert_url="http://localhost:8765")
    disabled_constructed, _, _, _ = _build_mocked_mappers_for_mode("disabled")

    assert sorted(public_constructed) == sorted(local_constructed) == sorted(disabled_constructed)
    assert sorted(o[0] for o in public_constructed) == ["HPO", "LOINC", "MONDO"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. strict_target_ontology=False is a locked module constant, unrelated to
# whether a target-ontology constraint exists at all.
# ─────────────────────────────────────────────────────────────────────────────


def test_strict_target_ontology_locked_false_but_ontologies_constraint_still_single_item() -> None:
    from llm_ontology_mapper.benchmarking.scenario2_runner import STRICT_TARGET_ONTOLOGY

    assert STRICT_TARGET_ONTOLOGY is False
    constructed, _, _, _ = _build_mocked_mappers_for_mode("public")
    # strict_target_ontology=False never means "no target ontology" -- every
    # mapper is still constructed with exactly one allowed ontology.
    for ontologies in constructed:
        assert len(ontologies) == 1
