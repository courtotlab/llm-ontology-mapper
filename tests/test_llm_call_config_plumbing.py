"""
Unit tests proving LLMCallConfig applies *consistently* to every applicable
LLM call in the planned pipeline (QueryPlanner and LLMReranker), and that
usage_sink telemetry reaches MappingResult.metadata.rag_debug.pipeline_usage.

Run with:  pytest tests/test_llm_call_config_plumbing.py -v -m unit
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.models import NormalizedCandidate, QueryPlan, RetrievalMode
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResponse,
    LLMCallConfig,
)
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.query_planner import QueryPlanner

pytestmark = pytest.mark.unit


class _RecordingProvider(BaseLLMProvider):
    """Returns a fixed response per call (cycling) and records every call's kwargs."""

    def __init__(self, responses: list[str], model: str = "gpt-5.6-luna") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.call_kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.call_kwargs.append({"temperature": temperature, "max_tokens": max_tokens, **kwargs})
        content = self._responses.pop(0) if self._responses else self._responses[-1]
        return CompletionResponse(content=content, model=self.model, prompt_tokens=100, completion_tokens=20)

    @property
    def provider_name(self) -> str:
        return "openai"


_PLAN_RESPONSE = json.dumps(
    {
        "normalized_term": "systolic blood pressure",
        "expanded_queries": ["systolic blood pressure"],
        "inferred_meaning": "systolic blood pressure",
        "semantic_type": "measurement",
        "candidate_ontologies": ["LOINC"],
        "preferred_ontology": "LOINC",
        "reasoning": "measurement",
        "confidence": 0.9,
    }
)

_RERANK_RESPONSE = json.dumps(
    {
        "selected_candidate_id": "C1",
        "selected_code": "LOINC:8480-6",
        "is_unmapped": False,
        "confidence": 0.9,
        "reasoning": "best match",
        "alternatives": [],
    }
)


def _candidate() -> NormalizedCandidate:
    return NormalizedCandidate(
        code="LOINC:8480-6",
        term="Systolic blood pressure",
        ontology="LOINC",
        source="LOINC-Search-API",
        matched_query="systolic blood pressure",
        retrieval_mode=RetrievalMode.PUBLIC,
        raw_score=0.9,
        normalized_score=0.9,
    )


def _plan() -> QueryPlan:
    return QueryPlan(
        original_term="sys_bp",
        expanded_queries=["systolic blood pressure"],
        retrieval_mode=RetrievalMode.PUBLIC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QueryPlanner
# ─────────────────────────────────────────────────────────────────────────────


def test_query_planner_applies_llm_call_config_overrides() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE])
    config = LLMCallConfig(temperature=0.0, seed=42, reasoning_effort="low", force_temperature=True, strict=True)
    planner = QueryPlanner(provider, llm_call_config=config)

    planner.plan(source_term="sys_bp")

    call = provider.call_kwargs[0]
    assert call["temperature"] == 0.0
    assert call["seed"] == 42
    assert call["reasoning_effort"] == "low"
    assert call["force_temperature"] is True
    assert call["strict"] is True


def test_query_planner_omits_temperature_when_llm_call_config_requests_provider_default() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE])
    config = LLMCallConfig(
        temperature=None, seed=42, reasoning_effort="low", force_temperature=True, strict=True
    )
    planner = QueryPlanner(provider, llm_call_config=config)

    planner.plan(source_term="sys_bp")

    call = provider.call_kwargs[0]
    assert call["temperature"] is None
    assert call["seed"] == 42
    assert call["reasoning_effort"] == "low"
    assert call["force_temperature"] is True
    assert call["strict"] is True


def test_query_planner_without_config_preserves_default_temperature() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE])
    planner = QueryPlanner(provider)

    planner.plan(source_term="sys_bp")

    call = provider.call_kwargs[0]
    assert call["temperature"] == 0.1
    assert "seed" not in call
    assert "strict" not in call


def test_query_planner_usage_sink_populated() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE])
    planner = QueryPlanner(provider)
    usage: dict[str, Any] = {}

    planner.plan(source_term="sys_bp", usage_sink=usage)

    assert usage["query_planner"]["input_tokens"] == 100
    assert usage["query_planner"]["output_tokens"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# LLMReranker
# ─────────────────────────────────────────────────────────────────────────────


def test_reranker_applies_llm_call_config_overrides_identically_to_planner() -> None:
    provider = _RecordingProvider([_RERANK_RESPONSE])
    config = LLMCallConfig(temperature=0.0, seed=42, reasoning_effort="low", force_temperature=True, strict=True)
    reranker = LLMReranker(provider, llm_call_config=config)

    reranker.rerank(_plan(), [_candidate()])

    call = provider.call_kwargs[0]
    assert call["temperature"] == 0.0
    assert call["seed"] == 42
    assert call["reasoning_effort"] == "low"
    assert call["force_temperature"] is True
    assert call["strict"] is True


def test_reranker_omits_temperature_when_llm_call_config_requests_provider_default() -> None:
    provider = _RecordingProvider([_RERANK_RESPONSE])
    config = LLMCallConfig(
        temperature=None, seed=42, reasoning_effort="low", force_temperature=True, strict=True
    )
    reranker = LLMReranker(provider, llm_call_config=config)

    reranker.rerank(_plan(), [_candidate()])

    call = provider.call_kwargs[0]
    assert call["temperature"] is None
    assert call["seed"] == 42
    assert call["reasoning_effort"] == "low"
    assert call["force_temperature"] is True
    assert call["strict"] is True


def test_reranker_usage_sink_populated() -> None:
    provider = _RecordingProvider([_RERANK_RESPONSE])
    reranker = LLMReranker(provider)
    usage: dict[str, Any] = {}

    reranker.rerank(_plan(), [_candidate()], usage_sink=usage)

    assert usage["llm_reranker"]["input_tokens"] == 100
    assert usage["llm_reranker"]["output_tokens"] == 20
    assert usage["llm_reranker"]["call_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# PlannedPipeline: same LLMCallConfig threads into both default components,
# and usage_sink telemetry survives all the way into the MappingResult.
# ─────────────────────────────────────────────────────────────────────────────


class _StubSearchTools:
    def search_loinc(
        self,
        query: str,
        top_k: int = 10,
        *,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "code": "LOINC:8480-6",
                "term": "Systolic blood pressure",
                "matched_query": query,
                "requested_ontology": "LOINC",
                "route_name": "LOINC-Search-API",
                "raw_score": 0.9,
                "normalized_score": 0.9,
            }
        ]


def test_planned_pipeline_threads_llm_call_config_into_both_default_components() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE, _RERANK_RESPONSE])
    config = LLMCallConfig(temperature=0.0, seed=42, reasoning_effort="low", force_temperature=True, strict=True)
    pipeline = PlannedPipeline(
        provider=provider,
        public_retriever=PublicOntologyRetriever(search_tools=_StubSearchTools()),
        llm_call_config=config,
    )

    pipeline.map_term("sys_bp", target_ontology="LOINC", retrieval_mode=RetrievalMode.PUBLIC)

    assert len(provider.call_kwargs) == 2
    for call in provider.call_kwargs:
        assert call["temperature"] == 0.0
        assert call["seed"] == 42
        assert call["reasoning_effort"] == "low"
        assert call["strict"] is True


def test_planned_pipeline_omits_temperature_consistently_for_both_default_components() -> None:
    """Benchmark models (e.g. gpt-5.6-luna) request provider-default temperature
    for both stages -- neither the planner nor the reranker call should differ
    in temperature treatment just because one supports temperature=0 and the
    other doesn't."""
    provider = _RecordingProvider([_PLAN_RESPONSE, _RERANK_RESPONSE])
    config = LLMCallConfig(
        temperature=None, seed=42, reasoning_effort="low", force_temperature=True, strict=True
    )
    pipeline = PlannedPipeline(
        provider=provider,
        public_retriever=PublicOntologyRetriever(search_tools=_StubSearchTools()),
        llm_call_config=config,
    )

    pipeline.map_term("sys_bp", target_ontology="LOINC", retrieval_mode=RetrievalMode.PUBLIC)

    assert len(provider.call_kwargs) == 2
    for call in provider.call_kwargs:
        assert call["temperature"] is None
        assert call["seed"] == 42
        assert call["reasoning_effort"] == "low"
        assert call["strict"] is True


def test_planned_pipeline_attaches_pipeline_usage_to_result() -> None:
    provider = _RecordingProvider([_PLAN_RESPONSE, _RERANK_RESPONSE])
    pipeline = PlannedPipeline(
        provider=provider,
        public_retriever=PublicOntologyRetriever(search_tools=_StubSearchTools()),
    )

    result = pipeline.map_term("sys_bp", target_ontology="LOINC", retrieval_mode=RetrievalMode.PUBLIC)

    pipeline_usage = result.metadata.rag_debug.pipeline_usage
    assert pipeline_usage is not None
    assert pipeline_usage["query_planner"]["input_tokens"] == 100
    assert pipeline_usage["llm_reranker"]["input_tokens"] == 100
