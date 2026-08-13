"""
Unit tests for the public data contracts (models.py).

These tests have ZERO external dependencies — they exercise only Pydantic
validation, property behaviour, and serialisation.  They should pass in < 1 s.

Run with:  pytest tests/test_models.py -v -m unit
"""

import pytest

from llm_ontology_mapper.models import (
    AlternativeMapping,
    GroundingSource,
    LogicType,
    MappingBatch,
    MappingResult,
    NormalizedCandidate,
    QueryPlan,
    RAGDebugInfo,
    RerankAlternative,
    RerankDecision,
    RetrievalMode,
    RetrievalTrace,
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
        AlternativeMapping(
            code="HP:0002110",
            term="Productive cough",
            ontology="HPO",
            confidence=0.72,
            source="rag",
        ),
        AlternativeMapping(
            code="HP:0031245",
            term="Non-productive cough",
            ontology="HPO",
            confidence=0.55,
            source="llm",
        ),
    ]
    return minimal_result


# ─────────────────────────────────────────────────────────────────────────────
# MappingResult — construction & validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_minimal_result_construction(minimal_result: MappingResult) -> None:
    assert minimal_result.source_term == "cough"
    assert minimal_result.target_code == "HP:0012735"
    assert minimal_result.ontology == "HPO"


@pytest.mark.unit
def test_ontology_prefix_normalised_to_uppercase() -> None:
    r = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="hpo",
        confidence=0.9,
        logic_type=LogicType.LLM,
    )
    assert r.ontology == "HPO"


@pytest.mark.unit
def test_native_efo_candidate_identity_is_valid() -> None:
    candidate = NormalizedCandidate(
        code="EFO:0000408",
        term="disease",
        ontology="EFO",
        source="OLS",
        matched_query="disease",
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert candidate.code == "EFO:0000408"
    assert candidate.ontology == "EFO"


@pytest.mark.unit
def test_retrieved_from_ontologies_are_canonical_and_deduplicated() -> None:
    candidate = NormalizedCandidate(
        code="HP:0002099",
        term="Asthma",
        ontology="HPO",
        source="OLS",
        matched_query="asthma",
        retrieval_mode=RetrievalMode.PUBLIC,
        retrieved_from_ontologies=["efo", "EFO", "HP", "HPO"],
    )

    assert candidate.retrieved_from_ontologies == ["EFO", "HPO"]


@pytest.mark.unit
def test_blank_target_code_raises() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        MappingResult(
            source_term="x",
            target_code="  ",
            target_term="X",
            ontology="HPO",
            confidence=0.9,
            logic_type=LogicType.LLM,
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range_raises(bad_confidence: float) -> None:
    with pytest.raises(ValueError):
        MappingResult(
            source_term="x",
            target_code="HP:0000001",
            target_term="X",
            ontology="HPO",
            confidence=bad_confidence,
            logic_type=LogicType.LLM,
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
        source_term="wbc",
        target_code="806-0",
        target_term="WBC",
        ontology="LOINC",
        confidence=0.8,
        logic_type=LogicType.LLM,
    )
    assert r.target_code == "LOINC:806-0"


# ─────────────────────────────────────────────────────────────────────────────
# is_high_confidence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "confidence,expected", [(0.8, True), (0.79, False), (1.0, True), (0.0, False)]
)
def test_is_high_confidence(confidence: float, expected: bool) -> None:
    r = MappingResult(
        source_term="x",
        target_code="HP:0000001",
        target_term="X",
        ontology="HPO",
        confidence=confidence,
        logic_type=LogicType.LLM,
    )
    assert r.is_high_confidence is expected


# ─────────────────────────────────────────────────────────────────────────────
# alternatives — sorted descending
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_alternatives_sorted_descending(result_with_alternatives: MappingResult) -> None:
    scores = [a.confidence for a in result_with_alternatives.alternatives]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_alternative_mapping_explanation_is_optional() -> None:
    alt = AlternativeMapping(
        code="LOINC:60984-2",
        term="Aortic systolic pressure",
        ontology="LOINC",
        confidence=0.75,
        source="rag",
        explanation="Could fit if the field is specifically an aortic measurement.",
    )

    assert alt.explanation == "Could fit if the field is specifically an aortic measurement."


@pytest.mark.unit
def test_rerank_alternative_serializes() -> None:
    alt = RerankAlternative(
        candidate_id="C1",
        code="LOINC:60984-2",
        confidence=0.75,
        explanation="Could fit if the measurement context is more specific.",
    )

    dumped = alt.model_dump(mode="json")
    assert dumped["candidate_id"] == "C1"
    assert dumped["code"] == "LOINC:60984-2"


# ─────────────────────────────────────────────────────────────────────────────
# to_legacy_dict — backwards compatibility contract
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_to_legacy_dict_keys(minimal_result: MappingResult) -> None:
    d = minimal_result.to_legacy_dict()
    expected_keys = {
        "source_field",
        "source_label",
        "source_type",
        "code",
        "term",
        "ontology",
        "confidence",
        "alternatives",
        "notes",
    }
    assert set(d.keys()) == expected_keys


@pytest.mark.unit
def test_to_legacy_dict_field_mapping(minimal_result: MappingResult) -> None:
    d = minimal_result.to_legacy_dict()
    assert d["source_field"] == minimal_result.source_term
    assert d["code"] == minimal_result.target_code
    assert d["term"] == minimal_result.target_term
    assert d["ontology"] == minimal_result.ontology


# ─────────────────────────────────────────────────────────────────────────────
# JSON round-trip
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_json_round_trip(minimal_result: MappingResult) -> None:
    json_str = minimal_result.model_dump_json()
    restored = MappingResult.model_validate_json(json_str)
    assert restored.target_code == minimal_result.target_code
    assert restored.confidence == minimal_result.confidence


# ─────────────────────────────────────────────────────────────────────────────
# MappingBatch
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_batch_high_confidence_filter(minimal_result: MappingResult) -> None:
    low = MappingResult(
        source_term="x",
        target_code="HP:0000002",
        target_term="X",
        ontology="HPO",
        confidence=0.5,
        logic_type=LogicType.LLM,
    )
    batch = MappingBatch(results=[minimal_result, low])
    assert len(batch.high_confidence) == 1
    assert len(batch.needs_review) == 1


@pytest.mark.unit
def test_batch_to_csv_records(minimal_result: MappingResult) -> None:
    batch = MappingBatch(results=[minimal_result])
    records = batch.to_csv_records()
    assert len(records) == 1
    assert records[0]["code"] == minimal_result.target_code


# ─────────────────────────────────────────────────────────────────────────────
# QueryPlan
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_plan_public_mode() -> None:
    plan = QueryPlan(original_term="sys_bp", retrieval_mode=RetrievalMode.PUBLIC)
    assert plan.retrieval_mode == RetrievalMode.PUBLIC


@pytest.mark.unit
def test_query_plan_local_mode() -> None:
    plan = QueryPlan(original_term="sys_bp", retrieval_mode=RetrievalMode.LOCAL)
    assert plan.retrieval_mode == RetrievalMode.LOCAL


@pytest.mark.unit
def test_query_plan_disabled_mode() -> None:
    plan = QueryPlan(original_term="sys_bp", retrieval_mode=RetrievalMode.DISABLED)
    assert plan.retrieval_mode == RetrievalMode.DISABLED


@pytest.mark.unit
def test_query_plan_source_context_defaults_to_none() -> None:
    plan = QueryPlan(original_term="sys_bp")

    assert plan.source_description is None
    assert plan.source_type is None


@pytest.mark.unit
def test_query_plan_source_context_serializes_when_supplied() -> None:
    plan = QueryPlan(
        original_term="creat",
        source_description="Most recent serum creatinine result collected at enrolment",
        source_type="decimal",
    )

    dumped = plan.model_dump(mode="json")
    assert dumped["source_description"] == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert dumped["source_type"] == "decimal"


@pytest.mark.unit
def test_query_plan_blank_original_term_raises() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        QueryPlan(original_term="   ")


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_query_plan_confidence_out_of_range_raises(bad_confidence: float) -> None:
    with pytest.raises(ValueError):
        QueryPlan(original_term="sys_bp", confidence=bad_confidence)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,expected",
    [
        (RetrievalMode.PUBLIC, True),
        (RetrievalMode.LOCAL, False),
        (RetrievalMode.DISABLED, False),
    ],
)
def test_query_plan_route_public_apis(mode: RetrievalMode, expected: bool) -> None:
    plan = QueryPlan(original_term="sys_bp", retrieval_mode=mode)
    assert plan.route_public_apis is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,expected",
    [
        (RetrievalMode.PUBLIC, False),
        (RetrievalMode.LOCAL, True),
        (RetrievalMode.DISABLED, False),
    ],
)
def test_query_plan_route_local(mode: RetrievalMode, expected: bool) -> None:
    plan = QueryPlan(original_term="sys_bp", retrieval_mode=mode)
    assert plan.route_local is expected


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCandidate
# ─────────────────────────────────────────────────────────────────────────────


def _candidate(
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC, **kwargs
) -> NormalizedCandidate:
    defaults: dict = dict(
        code="LOINC:8480-6",
        term="Systolic blood pressure",
        ontology="LOINC",
        source="LOINC-Search-API",
        matched_query="systolic blood pressure",
        retrieval_mode=retrieval_mode,
    )
    defaults.update(kwargs)
    return NormalizedCandidate(**defaults)


@pytest.mark.unit
def test_normalized_candidate_public() -> None:
    c = _candidate(RetrievalMode.PUBLIC)
    assert c.retrieval_mode == RetrievalMode.PUBLIC
    assert c.ontology == "LOINC"


@pytest.mark.unit
def test_normalized_candidate_local() -> None:
    c = _candidate(RetrievalMode.LOCAL)
    assert c.retrieval_mode == RetrievalMode.LOCAL


@pytest.mark.unit
def test_normalized_candidate_disabled_raises() -> None:
    with pytest.raises(ValueError, match="disabled"):
        _candidate(RetrievalMode.DISABLED)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["code", "term", "ontology", "source", "matched_query"])
def test_normalized_candidate_blank_fields_raise(field: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        _candidate(**{field: "   "})


@pytest.mark.unit
@pytest.mark.parametrize("bad_score", [-0.1, 1.1])
def test_normalized_candidate_normalized_score_out_of_range_raises(bad_score: float) -> None:
    with pytest.raises(ValueError, match="normalized_score"):
        _candidate(normalized_score=bad_score)


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalTrace
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_retrieval_trace_public_grounded() -> None:
    trace = RetrievalTrace(
        retrieval_mode=RetrievalMode.PUBLIC,
        is_grounded=True,
        grounding_source=GroundingSource.PUBLIC_API,
        raw_candidate_count=5,
        merged_candidate_count=3,
    )
    assert trace.is_grounded is True
    assert trace.grounding_source == GroundingSource.PUBLIC_API
    assert trace.retrieval_skipped is False


@pytest.mark.unit
def test_retrieval_trace_local_grounded() -> None:
    trace = RetrievalTrace(
        retrieval_mode=RetrievalMode.LOCAL,
        is_grounded=True,
        grounding_source=GroundingSource.LOCAL_SAPBERT,
        raw_candidate_count=3,
        merged_candidate_count=2,
    )
    assert trace.is_grounded is True
    assert trace.grounding_source == GroundingSource.LOCAL_SAPBERT
    assert trace.retrieval_skipped is False


@pytest.mark.unit
def test_retrieval_trace_disabled_skipped() -> None:
    # retrieval_skipped is auto-set to True when retrieval_mode=disabled
    trace = RetrievalTrace(
        retrieval_mode=RetrievalMode.DISABLED,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_disabled_reason="user requested disabled mode",
    )
    assert trace.retrieval_skipped is True
    assert trace.is_grounded is False
    assert trace.grounding_source == GroundingSource.NONE


@pytest.mark.unit
def test_retrieval_trace_disabled_grounded_raises() -> None:
    with pytest.raises(ValueError, match="is_grounded"):
        RetrievalTrace(
            retrieval_mode=RetrievalMode.DISABLED,
            is_grounded=True,
            grounding_source=GroundingSource.NONE,
        )


@pytest.mark.unit
def test_retrieval_trace_disabled_wrong_grounding_source_raises() -> None:
    with pytest.raises(ValueError, match="grounding_source"):
        RetrievalTrace(
            retrieval_mode=RetrievalMode.DISABLED,
            is_grounded=False,
            grounding_source=GroundingSource.PUBLIC_API,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_candidate_count", -1),
        ("merged_candidate_count", -1),
    ],
)
def test_retrieval_trace_negative_counts_raise(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        RetrievalTrace(
            retrieval_mode=RetrievalMode.PUBLIC,
            is_grounded=True,
            grounding_source=GroundingSource.PUBLIC_API,
            **{field: value},
        )


# ─────────────────────────────────────────────────────────────────────────────
# RerankDecision
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rerank_decision_public_grounded() -> None:
    decision = RerankDecision(
        selected_code="LOINC:8480-6",
        is_grounded=True,
        grounding_source=GroundingSource.PUBLIC_API,
        retrieval_mode=RetrievalMode.PUBLIC,
        confidence=0.9,
    )
    assert decision.selected_code == "LOINC:8480-6"
    assert decision.is_grounded is True
    assert decision.retrieval_mode == RetrievalMode.PUBLIC


@pytest.mark.unit
def test_rerank_decision_local_grounded() -> None:
    decision = RerankDecision(
        selected_code="HP:0000001",
        is_grounded=True,
        grounding_source=GroundingSource.LOCAL_SAPBERT,
        retrieval_mode=RetrievalMode.LOCAL,
        confidence=0.85,
    )
    assert decision.selected_code == "HP:0000001"
    assert decision.retrieval_mode == RetrievalMode.LOCAL


@pytest.mark.unit
def test_rerank_decision_disabled_llm_only() -> None:
    decision = RerankDecision(
        selected_code="HP:0000001",
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=RetrievalMode.DISABLED,
        policy="disabled_llm_only",
    )
    assert decision.is_grounded is False
    assert decision.policy == "disabled_llm_only"
    assert decision.grounding_source == GroundingSource.NONE


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_rerank_decision_confidence_out_of_range_raises(bad_confidence: float) -> None:
    with pytest.raises(ValueError):
        RerankDecision(
            is_grounded=True,
            grounding_source=GroundingSource.PUBLIC_API,
            retrieval_mode=RetrievalMode.PUBLIC,
            confidence=bad_confidence,
        )


@pytest.mark.unit
def test_rerank_decision_unmapped() -> None:
    decision = RerankDecision(
        selected_code=None,
        is_unmapped=True,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    assert decision.is_unmapped is True
    assert decision.selected_code is None


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation round-trip for all new models
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_serialisation_all_new_models() -> None:
    models_under_test = [
        QueryPlan(
            original_term="sys_bp",
            retrieval_mode=RetrievalMode.PUBLIC,
            confidence=0.8,
            expanded_queries=["systolic blood pressure"],
        ),
        NormalizedCandidate(
            code="LOINC:8480-6",
            term="Systolic blood pressure",
            ontology="LOINC",
            source="LOINC-Search-API",
            matched_query="systolic blood pressure",
            retrieval_mode=RetrievalMode.PUBLIC,
            normalized_score=0.95,
        ),
        RetrievalTrace(
            retrieval_mode=RetrievalMode.PUBLIC,
            is_grounded=True,
            grounding_source=GroundingSource.PUBLIC_API,
            raw_candidate_count=5,
            merged_candidate_count=3,
        ),
        RerankDecision(
            selected_code="LOINC:8480-6",
            is_grounded=True,
            grounding_source=GroundingSource.PUBLIC_API,
            retrieval_mode=RetrievalMode.PUBLIC,
            confidence=0.9,
            alternative_codes=["LOINC:55284-4"],
        ),
    ]
    for model in models_under_test:
        dumped = model.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert dumped


# ─────────────────────────────────────────────────────────────────────────────
# Existing MappingResult — non-regression
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_mapping_result_unaffected() -> None:
    r = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )
    assert r.target_code == "HP:0012735"
    assert r.model_dump(mode="json")["ontology"] == "HPO"


@pytest.mark.unit
def test_rag_debug_pipeline_timings_serialize_as_milliseconds() -> None:
    debug = RAGDebugInfo(
        query_sent="sys_bp",
        candidates_retrieved=[],
        top_k=0,
        pipeline_timings={"query_planning_ms": 12.34},
    )

    dumped = debug.model_dump(mode="json")

    assert dumped["pipeline_timings"] == {"query_planning_ms": 12.34}
