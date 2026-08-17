"""
Unit tests for LLMReranker (llm_reranker.py).

All LLM calls use a stub provider — no API keys or network access needed.

Run with:  pytest tests/test_llm_reranker.py -v -m unit
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from llm_ontology_mapper import llm_reranker as llm_reranker_module
from llm_ontology_mapper.llm_reranker import LLMReranker, LLMRerankerError
from llm_ontology_mapper.models import (
    GroundingSource,
    NormalizedCandidate,
    QueryPlan,
    RerankAlternative,
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
        self.call_kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return CompletionResponse(content=self._response_content, model=self.model)


class _NamedStubProvider(_StubProvider):
    def __init__(self, response_content: str, *, model: str, provider_name: str) -> None:
        super().__init__(response_content, model=model)
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


class _SequenceProvider(BaseLLMProvider):
    """Returns response strings in order and records every call."""

    def __init__(self, responses: list[str], model: str = "stub") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        content = self._responses.pop(0) if self._responses else ""
        return CompletionResponse(content=content, model=self.model)


class _NamedSequenceProvider(_SequenceProvider):
    def __init__(self, responses: list[str], *, model: str, provider_name: str) -> None:
        super().__init__(responses, model=model)
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


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
    retrieved_from_ontologies: list[str] | None = None,
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
        retrieved_from_ontologies=retrieved_from_ontologies or [],
    )


def _make_plan(
    original_term: str = "sys_bp",
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    target_ontology_constraint: str | None = None,
    allowed_target_ontologies: list[str] | None = None,
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
        allowed_target_ontologies=allowed_target_ontologies,
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
    return json.dumps(
        {
            "selected_candidate_id": selected_cid,
            "selected_code": selected_code,
            "is_unmapped": is_unmapped,
            "confidence": confidence,
            "reasoning": reasoning,
            "alternative_codes": alternative_codes or [],
        }
    )


def _structured_response(
    *,
    selected_cid: str = "C3",
    selected_code: str = "LOINC:8480-6",
    alternatives: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {
            "selected_candidate_id": selected_cid,
            "selected_code": selected_code,
            "is_unmapped": False,
            "confidence": 0.95,
            "reasoning": "Best direct match.",
            "alternatives": alternatives or [],
        }
    )


def _assert_specific_fallback(
    explanation: str | None,
    *,
    expected_fragment: str,
) -> None:
    assert explanation
    assert len(explanation.split()) <= 25
    lower = explanation.lower()
    assert expected_fragment in lower
    assert "retrieved as a related candidate" not in lower
    assert "review the term and context" not in lower
    assert "may be appropriate if context matches" not in lower
    assert "this is another possible match" not in lower


_UNMAPPED_RESPONSE = json.dumps(
    {
        "selected_candidate_id": None,
        "selected_code": None,
        "is_unmapped": True,
        "confidence": 0.0,
        "reasoning": "None of the retrieved candidates match the source term.",
        "alternative_codes": [],
    }
)


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


@pytest.mark.unit
def test_reranker_records_provider_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([8.0, 8.5])
    monkeypatch.setattr(llm_reranker_module.time, "monotonic", lambda: next(ticks))
    candidate = _make_candidate()
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)
    timings: dict[str, float] = {}

    reranker.rerank(_make_plan(), [candidate], timing_sink=timings)

    assert timings["llm_reranker_provider_ms"] == 500.0


@pytest.mark.unit
def test_reranker_records_zero_provider_timing_when_skipped() -> None:
    provider = _StubProvider(_response())
    reranker = LLMReranker(provider)
    timings: dict[str, float] = {}

    result = reranker.rerank(_make_plan(), [], timing_sink=timings)

    assert result.is_unmapped is True
    assert provider.calls == []
    assert timings["llm_reranker_provider_ms"] == 0.0


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
    candidate = _make_candidate(retrieval_mode=RetrievalMode.LOCAL, source="SapBERT")
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
    candidate = _make_candidate(retrieval_mode=RetrievalMode.LOCAL, source="SapBERT")
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
# Test 14: alternative_codes outside candidate list are dropped
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_alternative_codes_outside_candidate_list_are_dropped() -> None:
    candidate = _make_candidate(code="HP:0012735")
    bad_response = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        alternative_codes=["INVENTED:9999"],
    )
    provider = _StubProvider(bad_response)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert result.alternative_codes == []
    assert result.alternatives == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: target_ontology_constraint rejects selected candidate outside constraint
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_constraint_rejects_wrong_ontology() -> None:
    # Candidate is HPO; plan constrains to LOINC
    hpo_candidate = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    plan = _make_plan(target_ontology_constraint="LOINC")

    result = reranker.rerank(plan, [hpo_candidate])

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert provider.calls == []


@pytest.mark.unit
def test_target_ontology_constraint_accepts_native_efo_candidate() -> None:
    candidate = _make_candidate(code="EFO:0000408", term="disease", ontology="EFO")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="EFO:0000408"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(target_ontology_constraint="EFO"), [candidate])

    assert result.selected_code == "EFO:0000408"
    assert result.is_grounded is True


@pytest.mark.unit
def test_target_ontology_constraint_accepts_imported_efo_retrieved_candidate() -> None:
    candidate = _make_candidate(
        code="MONDO:0004975",
        term="asthma",
        ontology="MONDO",
        retrieved_from_ontologies=["EFO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="MONDO:0004975"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(target_ontology_constraint="EFO"), [candidate])

    assert result.selected_code == "MONDO:0004975"
    assert result.is_grounded is True


@pytest.mark.unit
def test_target_ontology_constraint_rejects_mondo_candidate_not_retrieved_from_efo() -> None:
    candidate = _make_candidate(
        code="MONDO:0004975",
        term="asthma",
        ontology="MONDO",
        retrieved_from_ontologies=["MONDO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="MONDO:0004975"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(target_ontology_constraint="EFO"), [candidate])

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert provider.calls == []


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


@pytest.mark.unit
def test_empty_response_raises_clear_error() -> None:
    candidate = _make_candidate()
    provider = _StubProvider("")
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError) as exc_info:
        reranker.rerank(_make_plan(), [candidate])

    message = str(exc_info.value)
    assert "LLM returned empty response" in message
    assert "max_completion_tokens was too low" in message
    assert "source_term='sys_bp'" in message
    assert "candidate_count=1" in message
    assert "malformed JSON" not in message


@pytest.mark.unit
def test_reasoning_model_reranker_uses_larger_completion_budget() -> None:
    candidate = _make_candidate()
    provider = _NamedStubProvider(
        _response(selected_cid="C1", selected_code="HP:0012735"),
        model="gpt-5",
        provider_name="openai",
    )
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert provider.call_kwargs[0]["max_tokens"] == 512
    assert provider.call_kwargs[0]["min_completion_tokens"] == 4096
    assert provider.call_kwargs[0]["reasoning_effort"] == "minimal"


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.6-luna"])
def test_selected_openai_models_reranker_use_low_reasoning(model: str) -> None:
    candidate = _make_candidate()
    provider = _NamedStubProvider(
        _response(selected_cid="C1", selected_code="HP:0012735"),
        model=model,
        provider_name="openai",
    )
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert provider.call_kwargs[0]["max_tokens"] == 512
    assert provider.call_kwargs[0]["min_completion_tokens"] == 4096
    assert provider.call_kwargs[0]["reasoning_effort"] == "low"


@pytest.mark.unit
def test_reasoning_model_empty_response_retries_once_with_larger_budget() -> None:
    candidate = _make_candidate()
    provider = _NamedSequenceProvider(
        ["", _response(selected_cid="C1", selected_code="HP:0012735")],
        model="gpt-5",
        provider_name="openai",
    )
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert len(provider.calls) == 2
    assert provider.call_kwargs[0]["min_completion_tokens"] == 4096
    assert provider.call_kwargs[1]["max_tokens"] == 8192
    assert provider.call_kwargs[1]["min_completion_tokens"] == 8192


@pytest.mark.unit
@pytest.mark.parametrize("provider_name", ["ollama", "anthropic"])
def test_reranker_does_not_send_openai_reasoning_to_non_openai(
    provider_name: str,
) -> None:
    candidate = _make_candidate()
    provider = _NamedStubProvider(
        _response(selected_cid="C1", selected_code="HP:0012735"),
        model="gpt-5.1",
        provider_name=provider_name,
    )
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert provider.call_kwargs[0]["max_tokens"] == 512
    assert "min_completion_tokens" not in provider.call_kwargs[0]
    assert "reasoning_effort" not in provider.call_kwargs[0]


@pytest.mark.unit
def test_non_reasoning_model_reranker_keeps_default_budget() -> None:
    candidate = _make_candidate()
    provider = _StubProvider(
        _response(selected_cid="C1", selected_code="HP:0012735"),
        model="gpt-4.1-mini",
    )
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [candidate])

    assert result.selected_code == "HP:0012735"
    assert provider.call_kwargs[0]["max_tokens"] == 512
    assert "min_completion_tokens" not in provider.call_kwargs[0]
    assert "reasoning_effort" not in provider.call_kwargs[0]


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


@pytest.mark.unit
def test_imported_efo_candidate_serializes_native_and_retrieval_ontology() -> None:
    candidate = _make_candidate(
        code="MONDO:0004975",
        term="Alzheimer disease",
        ontology="MONDO",
        retrieved_from_ontologies=["EFO"],
        matched_query="Alzheimer disease",
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="MONDO:0004975"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(
        _make_plan(
            original_term="alzheimer_disease",
            original_label="Alzheimer disease",
            target_ontology_constraint="EFO",
            allowed_target_ontologies=["EFO"],
            inferred_meaning="Alzheimer disease",
            semantic_type="disease",
            expanded_queries=["Alzheimer disease"],
        ),
        [candidate],
    )

    user_message = provider.calls[0][1].content
    assert result.selected_code == "MONDO:0004975"
    assert "code=MONDO:0004975" in user_message
    assert "term=Alzheimer disease" in user_message
    assert "native_ontology=MONDO" in user_message
    assert "retrieved_from_ontologies=EFO" in user_message
    assert "already passed deterministic Python-side hard-target filtering" in user_message
    assert "For EFO targets only" in user_message
    assert (
        "Do not reject an EFO-retrieved candidate solely because "
        "native_ontology is not EFO"
    ) in user_message


@pytest.mark.unit
def test_native_efo_candidate_serializes_native_and_retrieval_ontology() -> None:
    candidate = _make_candidate(
        code="EFO:0000408",
        term="disease",
        ontology="EFO",
        retrieved_from_ontologies=["EFO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="EFO:0000408"))
    reranker = LLMReranker(provider)

    reranker.rerank(
        _make_plan(target_ontology_constraint="EFO", allowed_target_ontologies=["EFO"]),
        [candidate],
    )

    user_message = provider.calls[0][1].content
    assert "code=EFO:0000408" in user_message
    assert "term=disease" in user_message
    assert "native_ontology=EFO" in user_message
    assert "retrieved_from_ontologies=EFO" in user_message


@pytest.mark.unit
def test_multiple_retrieval_provenance_serializes_deterministically() -> None:
    candidate = _make_candidate(
        code="HP:0012735",
        term="Cough",
        ontology="HPO",
        retrieved_from_ontologies=["EFO", "HPO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    user_message = provider.calls[0][1].content
    assert "native_ontology=HPO" in user_message
    assert "retrieved_from_ontologies=EFO, HPO" in user_message


@pytest.mark.unit
def test_missing_retrieval_provenance_remains_representable() -> None:
    candidate = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(), [candidate])

    user_message = provider.calls[0][1].content
    assert "native_ontology=HPO" in user_message
    assert "retrieved_from_ontologies=none" in user_message


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
    candidates = [_make_candidate(code=f"HP:{i:07d}", term=f"Term {i}") for i in range(5)]
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
    c2 = _make_candidate(code="HP:0002110", term="Mean cough severity")
    resp = _response(
        selected_cid="C1",
        selected_code="HP:0012735",
        alternative_codes=["HP:0002110"],
    )
    provider = _StubProvider(resp)
    reranker = LLMReranker(provider)

    result = reranker.rerank(_make_plan(), [c1, c2])

    assert result.alternative_codes == ["HP:0002110"]
    assert result.alternatives == []


@pytest.mark.unit
def test_structured_alternatives_are_parsed() -> None:
    c1 = _make_candidate(
        code="LOINC:60984-2",
        term="Aortic systolic pressure",
        ontology="LOINC",
    )
    c2 = _make_candidate(
        code="LOINC:8462-4",
        term="Diastolic blood pressure",
        ontology="LOINC",
    )
    c3 = _make_candidate(
        code="LOINC:8480-6",
        term="Systolic blood pressure",
        ontology="LOINC",
    )
    response = _structured_response(
        alternatives=[
            {
                "candidate_id": "C1",
                "code": "LOINC:60984-2",
                "confidence": 0.75,
                "explanation": (
                    "Could fit if the measurement is specifically aortic systolic pressure."
                ),
            }
        ]
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2, c3])

    assert result.selected_code == "LOINC:8480-6"
    assert len(result.alternatives) == 1
    assert result.alternatives[0] == RerankAlternative(
        candidate_id="C1",
        code="LOINC:60984-2",
        confidence=0.75,
        explanation="Could fit if the measurement is specifically aortic systolic pressure.",
    )


@pytest.mark.unit
def test_allowed_target_ontologies_filter_candidates_before_prompt() -> None:
    loinc = _make_candidate(
        code="LOINC:8480-6",
        term="Disallowed systolic candidate qzx",
        ontology="LOINC",
    )
    hpo = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["HPO"]),
        [loinc, hpo],
    )

    assert result.selected_code == "HP:0012735"
    prompt = " ".join(message.content for message in provider.calls[0])
    assert "HP:0012735" in prompt
    assert "Disallowed systolic candidate qzx" not in prompt


@pytest.mark.unit
def test_allowed_target_ontologies_empty_candidate_set_returns_unmapped() -> None:
    candidate = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="LOINC:8480-6"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["HPO"]),
        [candidate],
    )

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert provider.calls == []


@pytest.mark.unit
def test_efo_allowed_target_filters_mondo_candidate_not_retrieved_from_efo() -> None:
    candidate = _make_candidate(
        code="MONDO:0004975",
        term="Alzheimer disease",
        ontology="MONDO",
        retrieved_from_ontologies=["MONDO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="MONDO:0004975"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["EFO"]),
        [candidate],
    )

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert provider.calls == []


@pytest.mark.unit
def test_non_efo_allowed_target_does_not_use_search_space_exception() -> None:
    candidate = _make_candidate(
        code="HP:0012735",
        term="Cough",
        ontology="HPO",
        retrieved_from_ontologies=["MONDO"],
    )
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["MONDO"]),
        [candidate],
    )

    assert result.is_unmapped is True
    assert result.selected_code is None
    assert provider.calls == []


@pytest.mark.unit
def test_allowed_target_ontologies_reject_unprompted_selection() -> None:
    loinc = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    hpo = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(_response(selected_cid="C2", selected_code="LOINC:8480-6"))
    reranker = LLMReranker(provider)

    with pytest.raises(LLMRerankerError, match="selected_candidate_id"):
        reranker.rerank(
            _make_plan(allowed_target_ontologies=["HPO"]),
            [hpo, loinc],
        )


@pytest.mark.unit
def test_allowed_target_ontologies_alternatives_remain_inside_list() -> None:
    hpo = _make_candidate(code="HP:0012735", ontology="HPO")
    loinc = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    mondo = _make_candidate(code="MONDO:0000001", ontology="MONDO")
    response = _structured_response(
        selected_cid="C1",
        selected_code="HP:0012735",
        alternatives=[
            {
                "candidate_id": "C2",
                "code": "MONDO:0000001",
                "confidence": 0.7,
                "explanation": "Allowed disease alternative.",
            },
            {
                "candidate_id": "C3",
                "code": "LOINC:8480-6",
                "confidence": 0.6,
                "explanation": "Unselected ontology.",
            },
        ],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["HPO", "MONDO"]),
        [hpo, loinc, mondo],
    )

    assert [alt.code for alt in result.alternatives] == ["MONDO:0000001"]


@pytest.mark.unit
def test_efo_allowed_target_keeps_imported_alternative() -> None:
    efo = _make_candidate(code="EFO:0000408", term="disease", ontology="EFO")
    mondo = _make_candidate(
        code="MONDO:0004975",
        term="asthma",
        ontology="MONDO",
        retrieved_from_ontologies=["EFO"],
    )
    loinc = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    response = _structured_response(
        selected_cid="C1",
        selected_code="EFO:0000408",
        alternatives=[
            {
                "candidate_id": "C2",
                "code": "MONDO:0004975",
                "confidence": 0.7,
                "explanation": "Imported EFO-search disease candidate.",
            },
            {
                "candidate_id": "C3",
                "code": "LOINC:8480-6",
                "confidence": 0.6,
                "explanation": "Unselected ontology.",
            },
        ],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(
        _make_plan(allowed_target_ontologies=["EFO"]),
        [efo, mondo, loinc],
    )

    assert [alt.code for alt in result.alternatives] == ["MONDO:0004975"]


@pytest.mark.unit
def test_reranker_prompt_asks_for_reviewer_notes_to_use_actual_codes() -> None:
    hpo = _make_candidate(code="HP:0012735", ontology="HPO")
    provider = _StubProvider(_response(selected_cid="C1", selected_code="HP:0012735"))
    reranker = LLMReranker(provider)

    reranker.rerank(_make_plan(allowed_target_ontologies=["HPO"]), [hpo])

    prompt = "\n".join(message.content for message in provider.calls[0])
    assert "actual ontology codes" in prompt
    assert "not internal candidate IDs" in prompt


@pytest.mark.unit
def test_structured_alternative_code_mismatch_is_dropped() -> None:
    c1 = _make_candidate(code="LOINC:60984-2", ontology="LOINC")
    c2 = _make_candidate(code="LOINC:8462-4", ontology="LOINC")
    c3 = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    response = _structured_response(
        alternatives=[
            {
                "candidate_id": "C1",
                "code": "LOINC:9999-9",
                "confidence": 0.7,
                "explanation": "Bad mismatch.",
            }
        ]
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2, c3])

    assert result.selected_code == "LOINC:8480-6"
    assert result.alternatives == []
    assert "LOINC:9999-9" not in result.alternative_codes


@pytest.mark.unit
def test_legacy_alternative_codes_with_candidate_ids_are_converted() -> None:
    c1 = _make_candidate(
        code="LOINC:60984-2",
        term="Aortic systolic pressure",
        ontology="LOINC",
    )
    c2 = _make_candidate(code="LOINC:8462-4", ontology="LOINC")
    c3 = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    response = _response(
        selected_cid="C3",
        selected_code="LOINC:8480-6",
        alternative_codes=["C1"],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2, c3])

    assert result.alternative_codes == ["LOINC:60984-2"]
    assert result.alternatives == []


@pytest.mark.unit
def test_blank_structured_alternative_explanation_gets_candidate_specific_fallback() -> None:
    c1 = _make_candidate(
        code="LOINC:60984-2",
        term="Aortic systolic pressure",
        ontology="LOINC",
    )
    c2 = _make_candidate(code="LOINC:8462-4", ontology="LOINC")
    c3 = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    response = _structured_response(
        alternatives=[
            {
                "candidate_id": "C1",
                "code": "LOINC:60984-2",
                "confidence": 0.7,
                "explanation": "",
            }
        ]
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2, c3])

    assert result.alternatives[0].code == "LOINC:60984-2"
    _assert_specific_fallback(
        result.alternatives[0].explanation,
        expected_fragment="aorta",
    )


@pytest.mark.unit
def test_structured_alternatives_are_capped() -> None:
    codes = [f"LOINC:{1000 + i}-{i % 10}" for i in range(1, 9)]
    candidates = [_make_candidate(code=code, ontology="LOINC") for code in codes]
    response = _structured_response(
        selected_cid="C8",
        selected_code=codes[7],
        alternatives=[
            {
                "candidate_id": f"C{i}",
                "code": codes[i - 1],
                "confidence": 0.6,
                "explanation": f"Alternative {i}.",
            }
            for i in range(1, 8)
        ],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), candidates)

    assert len(result.alternatives) == 5
    assert len(result.alternative_codes) == 5


@pytest.mark.unit
def test_structured_alternatives_are_sorted_by_final_confidence() -> None:
    codes = [f"LOINC:{1000 + i}-{i % 10}" for i in range(1, 5)]
    candidates = [_make_candidate(code=code, ontology="LOINC") for code in codes]
    response = _structured_response(
        selected_cid="C4",
        selected_code=codes[3],
        alternatives=[
            {
                "candidate_id": "C1",
                "code": codes[0],
                "confidence": 0.5,
                "explanation": "Lower-ranked alternative.",
            },
            {
                "candidate_id": "C2",
                "code": codes[1],
                "confidence": 0.8,
                "explanation": "Higher-ranked alternative.",
            },
        ],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), candidates)

    assert [alt.code for alt in result.alternatives] == [codes[1], codes[0]]
    assert result.alternative_codes[:2] == [codes[1], codes[0]]


@pytest.mark.unit
def test_alternative_confidence_above_selected_raises() -> None:
    c1 = _make_candidate(code="LOINC:1001-1", ontology="LOINC")
    c2 = _make_candidate(code="LOINC:1002-2", ontology="LOINC")
    response = json.dumps(
        {
            "selected_candidate_id": "C1",
            "selected_code": "LOINC:1001-1",
            "is_unmapped": False,
            "confidence": 0.7,
            "reasoning": "Best direct match.",
            "alternatives": [
                {
                    "candidate_id": "C2",
                    "code": "LOINC:1002-2",
                    "confidence": 0.9,
                    "explanation": "Inconsistent runner-up.",
                }
            ],
        }
    )
    reranker = LLMReranker(_StubProvider(response))

    with pytest.raises(LLMRerankerError, match="alternative confidence"):
        reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2])


@pytest.mark.unit
def test_selected_candidate_is_not_repeated_as_structured_alternative() -> None:
    c1 = _make_candidate(code="LOINC:60984-2", ontology="LOINC")
    c2 = _make_candidate(code="LOINC:8480-6", ontology="LOINC")
    response = _structured_response(
        selected_cid="C2",
        selected_code="LOINC:8480-6",
        alternatives=[
            {
                "candidate_id": "C2",
                "code": "LOINC:8480-6",
                "confidence": 0.9,
                "explanation": "Selected candidate should not be repeated.",
            },
            {
                "candidate_id": "C1",
                "code": "LOINC:60984-2",
                "confidence": 0.7,
                "explanation": "A related candidate.",
            },
        ],
    )
    reranker = LLMReranker(_StubProvider(response))

    result = reranker.rerank(_make_plan(target_ontology_constraint="LOINC"), [c1, c2])

    assert [a.code for a in result.alternatives] == ["LOINC:60984-2"]


@pytest.mark.unit
def test_empty_alternative_codes_is_fine() -> None:
    candidate = _make_candidate(code="HP:0012735")
    resp = _response(selected_cid="C1", selected_code="HP:0012735", alternative_codes=[])
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
    candidates = [_make_candidate(code=f"HP:{i:07d}", term=f"Term {i}") for i in range(4)]
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

    with pytest.raises(ValidationError):
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
