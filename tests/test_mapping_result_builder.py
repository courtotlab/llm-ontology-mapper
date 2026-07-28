"""
Unit tests for MappingResultBuilder (Phase 5B).

Validates:
- Grounded selected results for public and local retrieval modes
- UNMAPPED results (empty candidates, LLM decision)
- MappingResult field values: source_term, target_code, confidence, notes, etc.
- Pipeline metadata stored in result.metadata.rag_debug.candidates_retrieved[0]
- Alternative mappings: built from structured reranker alternatives, ordered,
  selected excluded
- Error cases: disabled mode, absent selected_code, ontology constraint mismatch,
  ungrounded selected decision, empty candidates for non-unmapped

Constraints:
- No external API calls, no LLM calls, no live retrieval
- All tests are pure in-memory; no fixtures requiring external state
- Does not modify OntologyMapper, AgenticMapper, or provider behavior
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.mapping_result_builder import (
    MappingResultBuilder,
    MappingResultBuilderError,
)
from llm_ontology_mapper.models import (
    GroundingSource,
    LogicType,
    NormalizedCandidate,
    QueryPlan,
    RerankAlternative,
    RerankDecision,
    RetrievalMode,
    RetrievalTrace,
)

# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_plan(**kwargs) -> QueryPlan:
    defaults: dict = {
        "original_term": "sys_bp",
        "original_label": "Systolic blood pressure",
        "retrieval_mode": RetrievalMode.PUBLIC,
        "expanded_queries": ["systolic blood pressure"],
        "inferred_meaning": "Measurement of systolic arterial blood pressure",
        "semantic_type": "measurement",
        "reasoning": "Clinical vital sign indicator",
        "target_ontology_constraint": None,
    }
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def _make_candidate(
    code: str = "LOINC:8480-6",
    term: str = "Systolic blood pressure",
    ontology: str = "LOINC",
    *,
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    matched_query: str = "systolic blood pressure",
    raw_score: float | None = 0.95,
    normalized_score: float | None = 0.92,
    provenance: dict | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        code=code,
        term=term,
        ontology=ontology,
        source="OLS",
        matched_query=matched_query,
        retrieval_mode=retrieval_mode,
        raw_score=raw_score,
        normalized_score=normalized_score,
        provenance=provenance,
    )


def _make_selected_decision(
    selected_code: str = "LOINC:8480-6",
    selected_candidate_id: str = "C1",
    *,
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    confidence: float = 0.91,
    reasoning: str = "Best LOINC match for systolic blood pressure.",
    alternative_codes: list[str] | None = None,
    alternatives: list[RerankAlternative] | None = None,
    grounding_source: GroundingSource = GroundingSource.PUBLIC_API,
) -> RerankDecision:
    return RerankDecision(
        selected_code=selected_code,
        selected_candidate_id=selected_candidate_id,
        is_unmapped=False,
        is_grounded=True,
        grounding_source=grounding_source,
        retrieval_mode=retrieval_mode,
        confidence=confidence,
        reasoning=reasoning,
        alternative_codes=alternative_codes or [],
        alternatives=alternatives or [],
        policy="production_grounded",
    )


def _make_unmapped_decision(
    *,
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    confidence: float = 0.0,
    reasoning: str = "No semantically correct candidate found.",
) -> RerankDecision:
    return RerankDecision(
        selected_code=None,
        selected_candidate_id=None,
        is_unmapped=True,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=retrieval_mode,
        confidence=confidence,
        reasoning=reasoning,
        alternative_codes=[],
        policy="production_grounded",
    )


def _make_trace(
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
) -> RetrievalTrace:
    return RetrievalTrace(
        retrieval_mode=retrieval_mode,
        is_grounded=True,
        grounding_source=GroundingSource.PUBLIC_API,
        raw_candidate_count=5,
        merged_candidate_count=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic happy paths
# ─────────────────────────────────────────────────────────────────────────────


def test_build_selected_public() -> None:
    builder = MappingResultBuilder()
    plan = _make_plan()
    candidate = _make_candidate()
    decision = _make_selected_decision()

    result = builder.build(
        query_plan=plan,
        rerank_decision=decision,
        candidates=[candidate],
    )

    assert result is not None
    assert result.target_code == "LOINC:8480-6"
    assert result.ontology == "LOINC"


def test_build_selected_local() -> None:
    builder = MappingResultBuilder()
    plan = _make_plan(retrieval_mode=RetrievalMode.LOCAL)
    candidate = _make_candidate(retrieval_mode=RetrievalMode.LOCAL)
    decision = _make_selected_decision(
        retrieval_mode=RetrievalMode.LOCAL,
        grounding_source=GroundingSource.LOCAL_SAPBERT,
    )

    result = builder.build(
        query_plan=plan,
        rerank_decision=decision,
        candidates=[candidate],
    )

    assert result.target_code == "LOINC:8480-6"
    assert result.ontology == "LOINC"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Source fields pass through
# ─────────────────────────────────────────────────────────────────────────────


def test_preserves_source_term_and_label() -> None:
    builder = MappingResultBuilder()
    plan = _make_plan(
        original_term="sys_bp",
        original_label="Systolic blood pressure",
    )
    result = builder.build(
        query_plan=plan,
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert result.source_term == "sys_bp"
    assert result.source_label == "Systolic blood pressure"


def test_source_type_passed_through() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
        source_type="radio",
    )
    assert result.source_type == "radio"


def test_source_type_default_none() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert result.source_type is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output fields from candidate and decision
# ─────────────────────────────────────────────────────────────────────────────


def test_target_code_term_ontology_from_candidate() -> None:
    builder = MappingResultBuilder()
    candidate = _make_candidate(
        code="HP:0012735",
        term="Cough",
        ontology="HPO",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    decision = _make_selected_decision(
        selected_code="HP:0012735",
        selected_candidate_id="C1",
    )
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[candidate],
    )
    assert result.target_code == "HP:0012735"
    assert result.target_term == "Cough"
    assert result.ontology == "HPO"


def test_confidence_from_rerank_decision() -> None:
    builder = MappingResultBuilder()
    decision = _make_selected_decision(confidence=0.77)
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[_make_candidate()],
    )
    assert result.confidence == 0.77


def test_notes_from_rerank_decision_reasoning() -> None:
    builder = MappingResultBuilder()
    reasoning = "High confidence LOINC match for systolic blood pressure measurement."
    decision = _make_selected_decision(reasoning=reasoning)
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[_make_candidate()],
    )
    assert result.notes == reasoning


def test_logic_type_is_rag_for_selected() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert result.logic_type == LogicType.RAG


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline metadata in rag_debug.candidates_retrieved[0]
# ─────────────────────────────────────────────────────────────────────────────


def _get_pipeline_info(result) -> dict:
    """Extract the pipeline metadata dict from the standard storage location."""
    return result.metadata.rag_debug.candidates_retrieved[0]


def test_metadata_includes_retrieval_mode() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(retrieval_mode=RetrievalMode.PUBLIC),
        rerank_decision=_make_selected_decision(retrieval_mode=RetrievalMode.PUBLIC),
        candidates=[_make_candidate()],
    )
    assert _get_pipeline_info(result)["retrieval_mode"] == "public"


def test_metadata_includes_is_grounded() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert _get_pipeline_info(result)["is_grounded"] is True


def test_metadata_includes_grounding_source() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(grounding_source=GroundingSource.PUBLIC_API),
        candidates=[_make_candidate()],
    )
    assert _get_pipeline_info(result)["grounding_source"] == "public_api"


def test_metadata_includes_policy() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert _get_pipeline_info(result)["policy"] == "production_grounded"


def test_metadata_includes_query_plan_summary() -> None:
    builder = MappingResultBuilder()
    plan = _make_plan(
        original_term="bp_sys",
        inferred_meaning="Systolic BP",
        semantic_type="measurement",
    )
    result = builder.build(
        query_plan=plan,
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    qp = _get_pipeline_info(result)["query_plan"]
    assert qp["original_term"] == "bp_sys"
    assert qp["inferred_meaning"] == "Systolic BP"
    assert qp["semantic_type"] == "measurement"


def test_metadata_includes_retrieval_trace_when_provided() -> None:
    builder = MappingResultBuilder()
    trace = _make_trace()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
        retrieval_trace=trace,
    )
    assert "retrieval_trace" in _get_pipeline_info(result)
    rt = _get_pipeline_info(result)["retrieval_trace"]
    assert rt["retrieval_mode"] == "public"


def test_metadata_retrieval_trace_absent_when_not_provided() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert "retrieval_trace" not in _get_pipeline_info(result)


def test_metadata_includes_selected_candidate_provenance_when_available() -> None:
    builder = MappingResultBuilder()
    prov = {"endpoint": "OLS", "raw_score": 0.95}
    candidate = _make_candidate(provenance=prov)
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[candidate],
    )
    assert _get_pipeline_info(result)["selected_candidate_provenance"] == prov


def test_metadata_model_is_pipeline() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert result.metadata.model == "pipeline"


def test_metadata_provider_is_grounded_pipeline() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    assert result.metadata.provider == "grounded_pipeline"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Alternatives
# ─────────────────────────────────────────────────────────────────────────────


def test_alternatives_built_from_structured_reranker_alternatives() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:8462-4", "Diastolic BP")
    c3 = _make_candidate("LOINC:55284-4", "Blood pressure panel")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternative_codes=["LOINC:8462-4", "LOINC:55284-4"],
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.7,
                explanation="Could fit if this is diastolic blood pressure.",
            ),
            RerankAlternative(
                candidate_id="C3",
                code="LOINC:55284-4",
                confidence=0.6,
                explanation="Could fit if this is a blood pressure panel.",
            ),
        ],
    )
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2, c3],
    )
    alt_codes = [a.code for a in result.alternatives]
    assert "LOINC:8462-4" in alt_codes
    assert "LOINC:55284-4" in alt_codes


def test_allowed_target_ontologies_filter_result_alternatives() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate("HP:0012735", "Cough", "HPO")
    allowed_alt = _make_candidate("MONDO:0000001", "Respiratory disease", "MONDO")
    unselected_alt = _make_candidate("LOINC:8480-6", "Systolic BP", "LOINC")
    decision = _make_selected_decision(
        selected_code="HP:0012735",
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="MONDO:0000001",
                confidence=0.7,
                explanation="Allowed disease alternative.",
            ),
            RerankAlternative(
                candidate_id="C3",
                code="LOINC:8480-6",
                confidence=0.6,
                explanation="Unselected ontology.",
            ),
        ],
    )

    result = builder.build(
        query_plan=_make_plan(allowed_target_ontologies=["HPO", "MONDO"]),
        rerank_decision=decision,
        candidates=[selected, allowed_alt, unselected_alt],
    )

    assert [alt.ontology for alt in result.alternatives] == ["MONDO"]


def test_selected_candidate_not_in_alternatives_by_stable_identity() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:8462-4", "Diastolic BP")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternative_codes=["LOINC:8480-6", "LOINC:8462-4"],
        alternatives=[
            RerankAlternative(
                candidate_id="C1",
                code="LOINC:8480-6",
                confidence=0.8,
                explanation="Selected candidate should be excluded.",
            ),
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.7,
                explanation="Could fit if this is diastolic blood pressure.",
            ),
        ],
    )
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2],
    )
    alt_codes = [a.code for a in result.alternatives]
    assert "LOINC:8480-6" not in alt_codes
    assert "LOINC:8462-4" in alt_codes


def test_alternative_order_follows_final_reranker_confidence() -> None:
    """Alternatives are ordered by comparable final reranker confidence."""
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP", normalized_score=0.9)
    c2 = _make_candidate("LOINC:8462-4", "Diastolic BP", normalized_score=0.7)
    c3 = _make_candidate("LOINC:55284-4", "BP panel", normalized_score=0.5)
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternative_codes=["LOINC:55284-4", "LOINC:8462-4"],
        alternatives=[
            RerankAlternative(
                candidate_id="C3",
                code="LOINC:55284-4",
                confidence=0.55,
                explanation="Could fit if this is a panel.",
            ),
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.8,
                explanation="Could fit if this is diastolic blood pressure.",
            ),
        ],
    )
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2, c3],
    )
    assert [a.code for a in result.alternatives] == ["LOINC:8462-4", "LOINC:55284-4"]
    assert [a.confidence for a in result.alternatives] == [0.8, 0.55]


def test_no_alternatives_when_alternative_codes_empty() -> None:
    builder = MappingResultBuilder()
    decision = _make_selected_decision(alternative_codes=[])
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[_make_candidate()],
    )
    assert result.alternatives == []


def test_alternatives_source_is_rag() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:8462-4", "Diastolic BP")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.7,
                explanation="Could fit if this is diastolic blood pressure.",
            )
        ],
    )
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2],
    )
    assert result.alternatives[0].source == "rag"


def test_structured_alternative_explanation_is_preserved() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:60984-2", "Aortic systolic pressure")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:60984-2",
                confidence=0.75,
                explanation=(
                    "Could fit if the measurement is specifically aortic systolic pressure."
                ),
            )
        ],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2],
    )

    assert result.alternatives[0].code == "LOINC:60984-2"
    assert result.alternatives[0].explanation == (
        "Could fit if the measurement is specifically aortic systolic pressure."
    )


def test_legacy_alternative_codes_are_not_exposed_without_final_confidence() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:8462-4", "Mean systolic BP")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternative_codes=["LOINC:8462-4"],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2],
    )

    assert result.alternatives == []
    info = _get_pipeline_info(result)
    assert info["candidate_score_provenance"][1]["code"] == "LOINC:8462-4"
    assert info["candidate_score_provenance"][1]["normalized_retrieval_score"] == 0.92


def test_empty_alternatives_do_not_fall_back_to_retrieval_scores() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:8462-4", "Diastolic BP")
    c3 = _make_candidate("LOINC:55284-4", "Exercise blood pressure")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternative_codes=[],
        alternatives=[],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2, c3],
    )

    assert result.alternatives == []


def test_alternatives_are_capped_by_builder() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate("LOINC:1000-0", "Selected")
    others = [
        _make_candidate(
            f"LOINC:{1000 + i}-{i % 10}",
            f"Candidate {i}",
            normalized_score=0.9 - i * 0.01,
        )
        for i in range(1, 8)
    ]
    decision = _make_selected_decision(
        selected_code="LOINC:1000-0",
        alternatives=[
            RerankAlternative(
                candidate_id=f"C{i + 1}",
                code=f"LOINC:{1000 + i}-{i % 10}",
                confidence=0.9 - i * 0.01,
                explanation=f"Alternative {i}.",
            )
            for i in range(1, 8)
        ],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[selected, *others],
        max_alternatives=5,
    )

    assert len(result.alternatives) == 5


def test_retrieval_score_is_not_exposed_as_final_alternative_confidence() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate(
        "SNOMEDCT:104847001",
        "Oxygen saturation measurement",
        ontology="SNOMEDCT",
        raw_score=0.84,
        normalized_score=0.84,
    )
    loinc_alt = _make_candidate(
        "LOINC:73804-7",
        "Oxygen saturation sensor name",
        ontology="LOINC",
        raw_score=1.0,
        normalized_score=1.0,
    )
    decision = _make_selected_decision(
        selected_code="SNOMEDCT:104847001",
        selected_candidate_id="C1",
        confidence=0.91,
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:73804-7",
                confidence=0.62,
                explanation="Could fit if the field records the sensor name.",
            )
        ],
    )

    result = builder.build(
        query_plan=_make_plan(original_term="oxygen_sat"),
        rerank_decision=decision,
        candidates=[selected, loinc_alt],
    )

    assert result.confidence == 0.91
    assert result.alternatives[0].confidence == 0.62
    assert result.alternatives[0].confidence != 1.0
    assert all(alt.confidence <= result.confidence for alt in result.alternatives)


def test_final_confidence_provenance_is_retained_in_debug_trace() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate(raw_score=0.95, normalized_score=0.92)
    alternative = _make_candidate(
        "LOINC:8462-4",
        "Diastolic BP",
        raw_score=1.0,
        normalized_score=1.0,
    )
    decision = _make_selected_decision(
        confidence=0.91,
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.63,
                explanation="Could fit if this is diastolic blood pressure.",
            )
        ],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[selected, alternative],
    )
    info = _get_pipeline_info(result)

    assert "final LLM reranker confidences" in info["confidence_contract"]
    assert info["candidate_score_provenance"][1]["raw_retrieval_score"] == 1.0
    assert info["candidate_score_provenance"][1]["normalized_retrieval_score"] == 1.0
    assert info["final_ranking_trace"][0]["final_confidence"] == 0.91
    assert info["final_ranking_trace"][0]["confidence_source"] == "llm_reranker"
    assert info["final_ranking_trace"][1]["final_confidence"] == 0.63
    assert info["final_ranking_trace"][1]["normalized_retrieval_score"] == 1.0


def test_alternative_confidence_above_selected_raises() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate("LOINC:8480-6", "Systolic BP")
    alternative = _make_candidate("LOINC:8462-4", "Diastolic BP")
    decision = _make_selected_decision(
        confidence=0.5,
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.8,
                explanation="Inconsistent confidence.",
            )
        ],
    )

    with pytest.raises(MappingResultBuilderError, match="greater than selected"):
        builder.build(
            query_plan=_make_plan(),
            rerank_decision=decision,
            candidates=[selected, alternative],
        )


@pytest.mark.parametrize("mode", [RetrievalMode.PUBLIC, RetrievalMode.LOCAL])
def test_grounded_routes_share_final_confidence_contract(mode: RetrievalMode) -> None:
    builder = MappingResultBuilder()
    source = (
        GroundingSource.PUBLIC_API
        if mode == RetrievalMode.PUBLIC
        else GroundingSource.LOCAL_SAPBERT
    )
    selected = _make_candidate(
        "LOINC:8480-6",
        "Systolic BP",
        retrieval_mode=mode,
        normalized_score=1.0,
    )
    alternative = _make_candidate(
        "LOINC:8462-4",
        "Diastolic BP",
        retrieval_mode=mode,
        normalized_score=1.0,
    )
    decision = _make_selected_decision(
        retrieval_mode=mode,
        grounding_source=source,
        confidence=0.86,
        alternatives=[
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:8462-4",
                confidence=0.6,
                explanation="Could fit if this is diastolic blood pressure.",
            )
        ],
    )

    result = builder.build(
        query_plan=_make_plan(retrieval_mode=mode),
        rerank_decision=decision,
        candidates=[selected, alternative],
    )

    assert result.confidence == 0.86
    assert result.alternatives[0].confidence == 0.6
    assert _get_pipeline_info(result)["retrieval_mode"] == mode.value


def test_selected_code_never_appears_in_structured_alternatives() -> None:
    builder = MappingResultBuilder()
    c1 = _make_candidate("LOINC:8480-6", "Systolic BP")
    c2 = _make_candidate("LOINC:60984-2", "Aortic systolic pressure")
    decision = _make_selected_decision(
        selected_code="LOINC:8480-6",
        alternatives=[
            RerankAlternative(
                candidate_id="C1",
                code="LOINC:8480-6",
                confidence=0.8,
                explanation="Should be excluded.",
            ),
            RerankAlternative(
                candidate_id="C2",
                code="LOINC:60984-2",
                confidence=0.7,
                explanation="Related candidate.",
            ),
        ],
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=decision,
        candidates=[c1, c2],
    )

    assert [alt.code for alt in result.alternatives] == ["LOINC:60984-2"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNMAPPED results
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_produces_unknown_unmapped_convention() -> None:
    builder = MappingResultBuilder()
    candidates = [_make_candidate()]
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(),
        candidates=candidates,
    )
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.target_term == "UNMAPPED"
    assert result.ontology == "UNKNOWN"
    assert result.confidence == 0.0


def test_unmapped_logic_type_is_rag() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(),
        candidates=[_make_candidate()],
    )
    assert result.logic_type == LogicType.RAG


def test_unmapped_with_empty_candidates_is_allowed() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(reasoning="No candidates found."),
        candidates=[],
    )
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.confidence == 0.0


def test_unmapped_alternatives_list_is_empty() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(),
        candidates=[],
    )
    assert result.alternatives == []


def test_unmapped_notes_from_reasoning() -> None:
    builder = MappingResultBuilder()
    reason = "None of the retrieved candidates semantically match the source term."
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(reasoning=reason),
        candidates=[],
    )
    assert result.notes == reason


def test_unmapped_metadata_is_grounded_false() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_unmapped_decision(),
        candidates=[],
    )
    assert _get_pipeline_info(result)["is_grounded"] is False


def test_unmapped_local_mode() -> None:
    builder = MappingResultBuilder()
    decision = _make_unmapped_decision(retrieval_mode=RetrievalMode.LOCAL)
    plan = _make_plan(retrieval_mode=RetrievalMode.LOCAL)
    result = builder.build(
        query_plan=plan,
        rerank_decision=decision,
        candidates=[],
    )
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert _get_pipeline_info(result)["retrieval_mode"] == "local"


def test_allowed_target_ontologies_unselected_final_selection_becomes_unmapped() -> None:
    builder = MappingResultBuilder()
    candidate = _make_candidate("LOINC:8480-6", "Systolic BP", "LOINC")

    result = builder.build(
        query_plan=_make_plan(allowed_target_ontologies=["HPO"]),
        rerank_decision=_make_selected_decision(selected_code="LOINC:8480-6"),
        candidates=[candidate],
    )

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"
    assert result.confidence == 0.0


def test_allowed_target_ontologies_none_applies_no_result_filter() -> None:
    builder = MappingResultBuilder()
    candidate = _make_candidate("LOINC:8480-6", "Systolic BP", "LOINC")

    result = builder.build(
        query_plan=_make_plan(allowed_target_ontologies=None),
        rerank_decision=_make_selected_decision(selected_code="LOINC:8480-6"),
        candidates=[candidate],
    )

    assert result.target_code == "LOINC:8480-6"
    assert result.ontology == "LOINC"


def test_single_allowed_target_ontology_valid_result_still_works() -> None:
    builder = MappingResultBuilder()
    candidate = _make_candidate("LOINC:8480-6", "Systolic BP", "LOINC")

    result = builder.build(
        query_plan=_make_plan(allowed_target_ontologies=["LOINC"]),
        rerank_decision=_make_selected_decision(selected_code="LOINC:8480-6"),
        candidates=[candidate],
    )

    assert result.target_code == "LOINC:8480-6"
    assert result.ontology == "LOINC"


def test_inconsistent_primary_candidate_becomes_unmapped() -> None:
    builder = MappingResultBuilder()
    candidate = NormalizedCandidate.model_construct(
        code="MONDO:HP:0002099",
        term="Asthma",
        ontology="MONDO",
        source="OLS",
        matched_query="asthma",
        retrieval_mode=RetrievalMode.PUBLIC,
        raw_score=0.9,
        normalized_score=0.9,
        definition=None,
        provenance=None,
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(
            selected_code="MONDO:HP:0002099",
            selected_candidate_id="C1",
        ),
        candidates=[candidate],
    )

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_inconsistent_alternative_candidate_is_not_returned() -> None:
    builder = MappingResultBuilder()
    selected = _make_candidate("HP:0002099", "Asthma", "HPO")
    malformed_alt = NormalizedCandidate.model_construct(
        code="MONDO:HP:0012735",
        term="Cough",
        ontology="MONDO",
        source="OLS",
        matched_query="cough",
        retrieval_mode=RetrievalMode.PUBLIC,
        raw_score=0.8,
        normalized_score=0.8,
        definition=None,
        provenance=None,
    )
    malformed_rerank_alt = RerankAlternative.model_construct(
        candidate_id="C2",
        code="MONDO:HP:0012735",
        confidence=0.6,
        explanation="Malformed namespace should not reach the public result.",
    )

    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(
            selected_code="HP:0002099",
            alternatives=[malformed_rerank_alt],
        ),
        candidates=[selected, malformed_alt],
    )

    assert result.target_code == "HP:0002099"
    assert result.alternatives == []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Error cases
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_decision_raises() -> None:
    builder = MappingResultBuilder()
    decision = RerankDecision(
        selected_code=None,
        selected_candidate_id=None,
        is_unmapped=True,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=RetrievalMode.DISABLED,
        confidence=0.0,
        reasoning="Disabled.",
        alternative_codes=[],
        policy="production_grounded",
    )
    with pytest.raises(MappingResultBuilderError, match="[Dd]isabled"):
        builder.build(
            query_plan=_make_plan(),
            rerank_decision=decision,
            candidates=[],
        )


def test_selected_with_empty_candidates_raises() -> None:
    builder = MappingResultBuilder()
    with pytest.raises(MappingResultBuilderError, match="empty"):
        builder.build(
            query_plan=_make_plan(),
            rerank_decision=_make_selected_decision(),
            candidates=[],
        )


def test_selected_code_absent_from_candidates_raises() -> None:
    builder = MappingResultBuilder()
    candidate = _make_candidate(code="LOINC:8462-4", term="Diastolic BP")
    decision = _make_selected_decision(selected_code="LOINC:8480-6")
    with pytest.raises(MappingResultBuilderError, match="LOINC:8480-6"):
        builder.build(
            query_plan=_make_plan(),
            rerank_decision=decision,
            candidates=[candidate],
        )


def test_target_ontology_constraint_mismatch_raises() -> None:
    builder = MappingResultBuilder()
    plan = _make_plan(target_ontology_constraint="LOINC")
    candidate = _make_candidate(code="HP:0012735", term="Cough", ontology="HPO")
    decision = _make_selected_decision(
        selected_code="HP:0012735",
        selected_candidate_id="C1",
    )
    with pytest.raises(MappingResultBuilderError, match="LOINC"):
        builder.build(
            query_plan=plan,
            rerank_decision=decision,
            candidates=[candidate],
        )


def test_ungrounded_selected_decision_raises() -> None:
    """
    A selected (non-unmapped) decision with is_grounded=False must raise.

    The RerankDecision model itself prevents is_grounded=False + is_unmapped=False
    + policy='production_grounded' via its validator.  We bypass this by using
    policy='research_debug', which the builder does not permit for selected results.
    """
    builder = MappingResultBuilder()
    decision = RerankDecision(
        selected_code="LOINC:8480-6",
        selected_candidate_id="C1",
        is_unmapped=False,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=RetrievalMode.PUBLIC,
        confidence=0.5,
        reasoning="Research debug path.",
        alternative_codes=[],
        policy="research_debug",
    )
    with pytest.raises(MappingResultBuilderError, match="grounded"):
        builder.build(
            query_plan=_make_plan(),
            rerank_decision=decision,
            candidates=[_make_candidate()],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. No both-mode
# ─────────────────────────────────────────────────────────────────────────────


def test_no_both_retrieval_mode() -> None:
    """RetrievalMode must not have a BOTH or 'both' value."""
    values = {m.value for m in RetrievalMode}
    assert "both" not in values
    assert len(values) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 9. Serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_result_serializes_cleanly() -> None:
    builder = MappingResultBuilder()
    result = builder.build(
        query_plan=_make_plan(),
        rerank_decision=_make_selected_decision(),
        candidates=[_make_candidate()],
    )
    dumped = result.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert dumped["target_code"] == "LOINC:8480-6"
    assert dumped["logic_type"] == "rag"
    assert dumped["metadata"]["model"] == "pipeline"
    # Pipeline info lives in rag_debug
    rag = dumped["metadata"]["rag_debug"]
    assert rag["candidates_retrieved"][0]["retrieval_mode"] == "public"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Imports from public surface
# ─────────────────────────────────────────────────────────────────────────────


def test_imports_available_from_package() -> None:
    from llm_ontology_mapper import MappingResultBuilder, MappingResultBuilderError  # noqa: F401

    assert MappingResultBuilder is not None
    assert MappingResultBuilderError is not None
