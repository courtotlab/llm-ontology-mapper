"""
Unit tests for LLMReranker (llm_reranker.py).

All LLM calls use a stub provider — no API keys or network access needed.

Run with:  pytest tests/test_llm_reranker.py -v -m unit
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ontology_mapper.llm_reranker import LLMReranker, LLMRerankerError
from llm_ontology_mapper.models import (
    GroundingSource,
    NormalizedCandidate,
    QueryPlan,
    RerankDecision,
    RetrievalMode,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse


# ─────────────────────────────────────────────────────────────────────────────
# Stub provider
# ─────────────────────────────────────────────────────────────────────────────


class _StubProvider(BaseLLMProvider):
    """Returns a fixed response string and records every call for inspection."""

    def __init__(self, response_content: str, model: str = "stub") -> None:
        super().__init__(model=model)
        self._response_content = response_content
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        return CompletionResponse(content=self._response_content, model=self.model)


# ─────────────────────────────────────────────────────────────────────────────
# Shared factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_candidate(
    code: str = "HP:0012735",
    term: str = "Cough",
    ontology: str = "HPO",
    source: str = "OLS",
    matched_query: str = "cough",
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    raw_score: float | None = 0.9,
    normalized_score: float | None = 0.9,
    definition: str | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        code=code,
        term=term,
        ontology=ontology,
        source=source,
        matched_query=matched_query,
        retrieval_mode=retrieval_mode,
        raw_score=raw_score,
        normalized_score=normalized_score,
        definition=definition,
    )


def _make_plan(
    original_term: str = "sys_bp",
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    target_ontology_constraint: str | None = None,
    inferred_meaning: str | None = "systolic blood pressure",
    semantic_type: str | None = "measurement",
    expanded_queries: list[str] | None = None,
    reasoning: str | None = None,
    original_label: str | None = None,
    normalized_term: str | None = None,
) -> QueryPlan:
    return QueryPlan(
        original_term=original_term,
        original_label=original_label,
        normalized_term=normalized_term,
        expanded_queries=expanded_queries or ["systolic blood pressure"],
        inferred_meaning=inferred_meaning,
        semantic_type=semantic_type,
        retrieval_mode=retrieval_mode,
        target_ontology_constraint=target_ontology_constraint,
        reasoning=reasoning,
    )


def _response(
    selected_cid: str | None = "C1",
    selected_code: str | None = "HP:0012735",
    is_unmapped: bool = False,
    confidence: float = 0.92,
    reasoning: str = "Best matching candidate for the source term.",
    alternative_codes: list[str] | None = None,
) -> str:
    return json.dumps({
        "selected_candidate_id": selected_cid,
        "selected_code": selected_code,
        "is_unmapped": is_unmapped,
        "confidence": confidence,
        "reasoning": reasoning,
        "alternative_codes": alternative_codes or [],
    })


_UNMAPPED_RESPONSE = json.dumps({
    "selected_candidate_id": None,
    "selected_code": None,
    "is_unmapped": True,
    "confidence": 0.0,
    "reasoning": "None of the retrieved candidates match the source term.",
    "alternative_codes": [],
})


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: reranker calls fake provider with query plan and candidate context
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_provider_is_called(tmp_path) -> None:
    candidate = _make_candidate()
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    assert len(provider.calls) == 1
    messages = provider.calls[0]
    user_message = messages[1].content

    # Source term should appear in the user prompt
    assert "sys_bp" in user_message
    # Candidate code should appear
    assert "HP:0012735" in user_message


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: reranker parses valid JSON into RerankDecision
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parses_valid_json_into_rerank_decision() -> None:
    candidate = _make_candidate(code="HP:0012735")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert isinstance(result, RerankDecision)
    assert result.selected_code == "HP:0012735"
    assert result.confidence == pytest.approx(0.92)
    assert result.is_unmapped is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: selected candidate by ID returns matching selected_code
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_selected_candidate_id_matches_code() -> None:
    c1 = _make_candidate(code="HP:0012735", term="Cough")
    c2 = _make_candidate(code="HP:0001250", term="Seizure")

    # LLM selects C2
    provider = _StubProvider(_response(selected_cid="C2", selected_code="HP:0001250"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [c1, c2])

    assert result.selected_candidate_id == "C2"
    assert result.selected_code == "HP:0001250"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: public mode selected candidate has is_grounded=True
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_public_mode_is_grounded() -> None:
    candidate = _make_candidate(retrieval_mode=RetrievalMode.PUBLIC)
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(retrieval_mode=RetrievalMode.PUBLIC), [candidate])

    assert result.is_grounded is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: public mode selected candidate has grounding_source=public_api
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_public_mode_grounding_source_public_api() -> None:
    candidate = _make_candidate(retrieval_mode=RetrievalMode.PUBLIC)
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(retrieval_mode=RetrievalMode.PUBLIC), [candidate])

    assert result.grounding_source == GroundingSource.PUBLIC_API


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: local mode selected candidate has is_grounded=True
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_local_mode_is_grounded() -> None:
    candidate = _make_candidate(
        retrieval_mode=RetrievalMode.LOCAL, source="SapBERT"
    )
    provider = _StubProvider(_response())
    plan = _make_plan(retrieval_mode=RetrievalMode.LOCAL)
    reranker = LLMReranker(provider)

    result = reranker.rerank(plan, [candidate])

    assert result.is_grounded is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: local mode selected candidate has grounding_source=local_sapbert
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_local_mode_grounding_source_local_sapbert() -> None:
    candidate = _make_candidate(
        retrieval_mode=RetrievalMode.LOCAL, source="SapBERT"
    )
    provider = _StubProvider(_response())
    plan = _make_plan(retrieval_mode=RetrievalMode.LOCAL)
    reranker = LLMReranker(provider)

    result = reranker.rerank(plan, [candidate])

    assert result.grounding_source == GroundingSource.LOCAL_SAPBERT


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: unmapped response returns selected_code=None and is_unmapped=True
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_unmapped_response_is_unmapped() -> None:
    candidate = _make_candidate()
    provider = _StubProvider(_UNMAPPED_RESPONSE)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert result.selected_candidate_id is None
    assert result.is_grounded is False
    assert result.grounding_source == GroundingSource.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: empty candidate list returns unmapped without calling provider
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_empty_candidate_list_skips_provider() -> None:
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [])

    # Provider must NOT have been called
    assert len(provider.calls) == 0
    assert result.is_unmapped is True
    assert result.selected_code is None
    assert result.is_grounded is False
    assert result.grounding_source == GroundingSource.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: disabled QueryPlan raises clear error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disabled_query_plan_raises_error() -> None:
    disabled_plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.DISABLED,
        retrieval_disabled_reason="Retrieval disabled by caller",
    )
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="disabled"):
        reranker.rerank(disabled_plan, [])

    # Provider must not have been called
    assert len(provider.calls) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: hallucinated selected_candidate_id raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_hallucinated_candidate_id_raises_error() -> None:
    candidate = _make_candidate(code="HP:0012735")
    # LLM returns C99 — does not exist in the map (only C1)
    bad_response = _response(selected_cid="C99", selected_code="HP:0012735")
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="C99"):
        reranker.rerank(_make_plan(), [candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: hallucinated selected_code raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_hallucinated_selected_code_raises_error() -> None:
    candidate = _make_candidate(code="HP:0012735")
    # LLM returns valid candidate ID but an invented code
    bad_response = _response(selected_cid="C1", selected_code="FAKE:0000000")
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="FAKE:0000000"):
        reranker.rerank(_make_plan(), [candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: selected_code mismatch with selected_candidate_id raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_code_mismatch_with_candidate_id_raises_error() -> None:
    c1 = _make_candidate(code="HP:0012735", term="Cough")
    c2 = _make_candidate(code="HP:0001250", term="Seizure")

    # LLM selects C1 but returns C2's code
    bad_response = _response(selected_cid="C1", selected_code="HP:0001250")
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="HP:0001250"):
        reranker.rerank(_make_plan(), [c1, c2])


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: alternative_codes outside candidate list raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_alternative_codes_outside_candidate_list_raises_error() -> None:
    candidate = _make_candidate(code="HP:0012735")
    bad_response = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        alternative_codes=["INVENTED:9999"],
    )
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="INVENTED:9999"):
        reranker.rerank(_make_plan(), [candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: target_ontology_constraint rejects selected candidate outside constraint
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_constraint_rejects_wrong_ontology() -> None:
    # Candidate is HPO; plan constrains to LOINC
    hpo_candidate = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(
        _response(selected_cid="C1", selected_code="HP:0012735")
    )
    reranker = LLMReranker(provider)

    plan = _make_plan(target_ontology_constraint="LOINC")

    with pytest.raises(LLMRerankerError, match="LOINC"):
        reranker.rerank(plan, [hpo_candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: malformed JSON raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_malformed_json_raises_error() -> None:
    candidate = _make_candidate()
    provider = _StubProvider("this is not json at all {broken")
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="malformed JSON"):
        reranker.rerank(_make_plan(), [candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: JSON wrapped in markdown code fences is accepted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_markdown_fences_stripped() -> None:
    candidate = _make_candidate(code="HP:0012735")
    json_body = _response(selected_cid="C1", selected_code="HP:0012735")
    fenced = f"```json\n{json_body}\n```"
    provider = _StubProvider(fenced)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert isinstance(result, RerankDecision)
    assert result.selected_code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: invalid confidence outside 0..1 raises error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [1.5, -0.1, 2.0, -1.0])
def test_invalid_confidence_raises_error(bad_confidence: float) -> None:
    candidate = _make_candidate(code="HP:0012735")
    bad_response = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        confidence=bad_confidence,
    )
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="confidence"):
        reranker.rerank(_make_plan(), [candidate])


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: candidate list includes definitions when available
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_candidate_list_includes_definition() -> None:
    candidate = _make_candidate(
        code="HP:0012735",
        definition="A cough is an expulsive reflex protecting the airways.",
    )
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    user_message = provider.calls[0][1].content
    assert "expulsive reflex" in user_message


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: candidate list includes matched_query and source
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_candidate_list_includes_matched_query_and_source() -> None:
    candidate = _make_candidate(
        matched_query="systolic blood pressure",
        source="LOINC-Search-API",
    )
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    user_message = provider.calls[0][1].content
    assert "systolic blood pressure" in user_message
    assert "LOINC-Search-API" in user_message


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: no both mode is introduced or accepted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_both_mode() -> None:
    assert not hasattr(RetrievalMode, "BOTH")

    with pytest.raises(ValueError):
        RetrievalMode("both")

    # Public result produces public_api, not some combined grounding source
    candidate = _make_candidate(retrieval_mode=RetrievalMode.PUBLIC)
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(retrieval_mode=RetrievalMode.PUBLIC), [candidate])

    assert result.grounding_source == GroundingSource.PUBLIC_API
    assert result.retrieval_mode == RetrievalMode.PUBLIC
    assert result.grounding_source != GroundingSource.LOCAL_SAPBERT


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: reranker does not call retrieval tools
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_retrieval_calls() -> None:
    # The only call the reranker makes is provider.complete().
    # route_calls and candidate fetching are upstream — the reranker never
    # fetches from OLS, LOINC, SapBERT, etc.
    candidates = [
        _make_candidate(code=f"HP:{i:07d}", term=f"Term {i}")
        for i in range(5)
    ]
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0000000"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), candidates)

    # Exactly one provider call, no side effects
    assert len(provider.calls) == 1
    assert result.selected_code == "HP:0000000"
    assert isinstance(result, RerankDecision)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 23–27: existing pipeline tests unaffected
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_candidate_merger_imports_unaffected() -> None:
    from llm_ontology_mapper.candidate_merger import CandidateMergeError, CandidateMerger

    assert CandidateMerger is not None
    assert CandidateMergeError is not None


@pytest.mark.unit
def test_existing_candidate_normalizer_imports_unaffected() -> None:
    from llm_ontology_mapper.candidate_normalizer import (
        CandidateNormalizationError,
        CandidateNormalizer,
    )

    assert CandidateNormalizer is not None
    assert CandidateNormalizationError is not None


@pytest.mark.unit
def test_existing_retrieval_router_imports_unaffected() -> None:
    from llm_ontology_mapper.retrieval_router import RetrievalRouter

    assert RetrievalRouter is not None


@pytest.mark.unit
def test_existing_query_planner_imports_unaffected() -> None:
    from llm_ontology_mapper.query_planner import QueryPlanner, QueryPlanningError

    assert QueryPlanner is not None
    assert QueryPlanningError is not None


@pytest.mark.unit
def test_existing_model_imports_unaffected() -> None:
    from llm_ontology_mapper.models import (
        LogicType,
        MappingResult,
        QueryPlan,
        RetrievalMode,
    )

    plan = QueryPlan(original_term="cough", retrieval_mode=RetrievalMode.PUBLIC)
    assert plan.route_public_apis is True

    r = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )
    assert r.target_code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge-case tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_alternative_codes_in_candidate_list_are_accepted() -> None:
    c1 = _make_candidate(code="HP:0012735", term="Cough")
    c2 = _make_candidate(code="HP:0002110", term="Bronchiectasis")
    resp = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        alternative_codes=["HP:0002110"],
    )
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [c1, c2])

    assert result.alternative_codes == ["HP:0002110"]


@pytest.mark.unit
def test_empty_alternative_codes_is_fine() -> None:
    candidate = _make_candidate(code="HP:0012735")
    resp = _response(
        selected_cid="C1", selected_code="HP:0012735", alternative_codes=[]
    )
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.alternative_codes == []


@pytest.mark.unit
def test_query_context_includes_inferred_meaning_and_semantic_type() -> None:
    plan = _make_plan(
        inferred_meaning="systolic blood pressure — peak arterial pressure",
        semantic_type="measurement",
    )
    candidate = _make_candidate()
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(plan, [candidate])

    user_message = provider.calls[0][1].content
    assert "systolic blood pressure" in user_message
    assert "measurement" in user_message


@pytest.mark.unit
def test_query_context_includes_target_ontology_constraint() -> None:
    plan = _make_plan(target_ontology_constraint="LOINC")
    candidate = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    resp = _response(selected_cid="C1", selected_code="LOINC:8480-6")
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    reranker.rerank(plan, [candidate])

    user_message = provider.calls[0][1].content
    assert "LOINC" in user_message


@pytest.mark.unit
def test_multiple_candidates_get_sequential_ids() -> None:
    candidates = [
        _make_candidate(code=f"HP:{i:07d}", term=f"Term {i}")
        for i in range(4)
    ]
    # LLM selects C3 (the third candidate, index 2)
    third_code = "HP:0000002"
    resp = _response(selected_cid="C3", selected_code=third_code)
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), candidates)

    assert result.selected_candidate_id == "C3"
    assert result.selected_code == third_code


@pytest.mark.unit
def test_unmapped_from_llm_has_grounding_source_none() -> None:
    candidate = _make_candidate()
    provider = _StubProvider(_UNMAPPED_RESPONSE)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(retrieval_mode=RetrievalMode.PUBLIC), [candidate])

    # Even in public mode, an UNMAPPED result has grounding_source NONE
    assert result.grounding_source == GroundingSource.NONE
    assert result.is_grounded is False


@pytest.mark.unit
def test_result_is_frozen_rerank_decision() -> None:
    candidate = _make_candidate(code="HP:0012735")
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    with pytest.raises(Exception):
        result.selected_code = "CHANGED"  # type: ignore[misc]


@pytest.mark.unit
def test_result_serialises_to_json() -> None:
    candidate = _make_candidate(code="HP:0012735")
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])
    dumped = result.model_dump(mode="json")

    assert isinstance(dumped, dict)
    assert dumped["selected_code"] == "HP:0012735"
    assert dumped["retrieval_mode"] == "public"
    assert dumped["grounding_source"] == "public_api"
    assert dumped["is_grounded"] is True


@pytest.mark.unit
def test_reasoning_preserved_in_decision() -> None:
    candidate = _make_candidate(code="HP:0012735")
    resp = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        reasoning="This is the best match for systolic blood pressure.",
    )
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.reasoning == "This is the best match for systolic blood pressure."


@pytest.mark.unit
def test_local_mode_empty_candidates_still_uses_production_policy() -> None:
    provider = _StubProvider(_UNMAPPED_RESPONSE)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(retrieval_mode=RetrievalMode.LOCAL), [])

    assert result.is_unmapped is True
    assert result.policy == "production_grounded"
    assert len(provider.calls) == 0


@pytest.mark.unit
def test_query_context_includes_retrieval_mode() -> None:
    plan = _make_plan(retrieval_mode=RetrievalMode.LOCAL)
    candidate = _make_candidate(retrieval_mode=RetrievalMode.LOCAL, source="SapBERT")
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(plan, [candidate])

    user_message = provider.calls[0][1].content
    assert "local" in user_message


@pytest.mark.unit
def test_scores_appear_in_candidate_list() -> None:
    candidate = _make_candidate(
        normalized_score=0.87,
        raw_score=0.91,
    )
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    user_message = provider.calls[0][1].content
    assert "0.8700" in user_message
    assert "0.9100" in user_message
