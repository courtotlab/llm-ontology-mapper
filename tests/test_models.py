"""
Unit tests for the public data contracts (models.py).

These tests have ZERO external dependencies — they exercise only Pydantic
validation, property behaviour, and serialisation.  They should pass in < 1 s.

Run with:  pytest tests/test_models.py -v -m unit
"""

import pytest
from llm_ontology_mapper.models import (
    AlternativeMapping,
    LogicType,
    MappingBatch,
    MappingResult,
    OntologyPrefix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_result() -> MappingResult:
    return MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )


@pytest.fixture()
def result_with_alternatives(minimal_result: MappingResult) -> MappingResult:
    minimal_result.alternatives = [
        AlternativeMapping(code="HP:0002110", term="Productive cough",   ontology="HPO", confidence=0.72, source="rag"),
        AlternativeMapping(code="HP:0031245", term="Non-productive cough", ontology="HPO", confidence=0.55, source="llm"),
    ]
    return minimal_result


# ─────────────────────────────────────────────────────────────────────────────
# MappingResult — construction & validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_minimal_result_construction(minimal_result: MappingResult) -> None:
    assert minimal_result.source_term == "cough"
    assert minimal_result.target_code == "HP:0012735"
    assert minimal_result.ontology    == "HPO"


@pytest.mark.unit
def test_ontology_prefix_normalised_to_uppercase() -> None:
    r = MappingResult(
        source_term="cough", target_code="HP:0012735", target_term="Cough",
        ontology="hpo", confidence=0.9, logic_type=LogicType.LLM,
    )
    assert r.ontology == "HPO"


@pytest.mark.unit
def test_blank_target_code_raises() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        MappingResult(
            source_term="x", target_code="  ", target_term="X",
            ontology="HPO", confidence=0.9, logic_type=LogicType.LLM,
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range_raises(bad_confidence: float) -> None:
    with pytest.raises(ValueError):
        MappingResult(
            source_term="x", target_code="HP:0000001", target_term="X",
            ontology="HPO", confidence=bad_confidence, logic_type=LogicType.LLM,
        )


# ─────────────────────────────────────────────────────────────────────────────
# target_code CURIE normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_target_code_already_curie_unchanged(minimal_result: MappingResult) -> None:
    assert minimal_result.target_code == "HP:0012735"


@pytest.mark.unit
def test_bare_code_normalised_to_curie() -> None:
    r = MappingResult(
        source_term="wbc", target_code="806-0", target_term="WBC",
        ontology="LOINC", confidence=0.8, logic_type=LogicType.LLM,
    )
    assert r.target_code == "LOINC:806-0"


# ─────────────────────────────────────────────────────────────────────────────
# is_high_confidence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("confidence,expected", [(0.8, True), (0.79, False), (1.0, True), (0.0, False)])
def test_is_high_confidence(confidence: float, expected: bool) -> None:
    r = MappingResult(
        source_term="x", target_code="HP:0000001", target_term="X",
        ontology="HPO", confidence=confidence, logic_type=LogicType.LLM,
    )
    assert r.is_high_confidence is expected


# ─────────────────────────────────────────────────────────────────────────────
# alternatives — sorted descending
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_alternatives_sorted_descending(result_with_alternatives: MappingResult) -> None:
    scores = [a.confidence for a in result_with_alternatives.alternatives]
    assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# to_legacy_dict — backwards compatibility contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_to_legacy_dict_keys(minimal_result: MappingResult) -> None:
    d = minimal_result.to_legacy_dict()
    expected_keys = {"source_field", "source_label", "source_type", "code", "term", "ontology", "confidence", "alternatives", "notes"}
    assert set(d.keys()) == expected_keys


@pytest.mark.unit
def test_to_legacy_dict_field_mapping(minimal_result: MappingResult) -> None:
    d = minimal_result.to_legacy_dict()
    assert d["source_field"] == minimal_result.source_term
    assert d["code"]         == minimal_result.target_code
    assert d["term"]         == minimal_result.target_term
    assert d["ontology"]     == minimal_result.ontology


# ─────────────────────────────────────────────────────────────────────────────
# JSON round-trip
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_json_round_trip(minimal_result: MappingResult) -> None:
    json_str = minimal_result.model_dump_json()
    restored = MappingResult.model_validate_json(json_str)
    assert restored.target_code == minimal_result.target_code
    assert restored.confidence  == minimal_result.confidence


# ─────────────────────────────────────────────────────────────────────────────
# MappingBatch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_batch_high_confidence_filter(minimal_result: MappingResult) -> None:
    low = MappingResult(
        source_term="x", target_code="HP:0000002", target_term="X",
        ontology="HPO", confidence=0.5, logic_type=LogicType.LLM,
    )
    batch = MappingBatch(results=[minimal_result, low])
    assert len(batch.high_confidence) == 1
    assert len(batch.needs_review)    == 1


@pytest.mark.unit
def test_batch_to_csv_records(minimal_result: MappingResult) -> None:
    batch   = MappingBatch(results=[minimal_result])
    records = batch.to_csv_records()
    assert len(records) == 1
    assert records[0]["code"] == minimal_result.target_code
