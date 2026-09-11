"""
End-to-end regression tests for OntologyMapper.map_term() and
OntologyMapper.map_data_dictionary() -- the two Bridge-facing mapping
workflows -- proving they still run through the full seven-stage planned
pipeline with the REAL, unmodified QueryPlanner, RetrievalRouter,
CandidateNormalizer, CandidateMerger, LLMReranker, and MappingResultBuilder.

Only the LLM provider and the public retrieval HTTP layer are faked (via a
FakeSearchTools double, as in tests/test_public_retriever.py). No live LLM
or ontology API call is made.

This is the single-term/batch "first-class acceptance requirement" coverage
for the retrieval extensibility refactor: it proves the planned pipeline
contract (result content, batch ordering/count, row-level context
propagation, strict_target_ontology, UNMAPPED handling, and per-row failure
isolation) is unaffected by the PublicOntologyRetriever registry/config
rewrite.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ontology_mapper.candidate_merger import CandidateMerger
from llm_ontology_mapper.candidate_normalizer import CandidateNormalizer
from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilder
from llm_ontology_mapper.models import GroundingSource, LogicType, MappingBatch, MappingResult
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.query_planner import QueryPlanner
from llm_ontology_mapper.retrieval_router import RetrievalRouter

pytestmark = pytest.mark.unit


class FakeSearchTools:
    """No-network SearchTools double keyed by query text, so different rows
    in a batch can be routed to different canned OLS responses."""

    def __init__(self, responses_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self._responses_by_query = responses_by_query
        self.calls: list[dict[str, Any]] = []

    def search_ols(
        self,
        query: str,
        ontology: str,
        top_k: int = 10,
        *,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "ontology": ontology, "top_k": top_k})
        return list(self._responses_by_query.get(query, []))

    def search_loinc(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        return []

    def search_rxnorm(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        return []

    def search_icd10(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        return []


class ScriptedProvider(BaseLLMProvider):
    """Replays a scripted sequence of JSON responses, one per .complete()
    call, in order. PlannedPipeline calls QueryPlanner then LLMReranker
    (public/local mode) in that fixed order per map_term()."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(model="stub-model")
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        content = self._responses[len(self.calls) - 1]
        return CompletionResponse(
            content=content, model=self.model, prompt_tokens=10, completion_tokens=5
        )


def _plan_json(query: str = "cough", ontology: str = "HPO") -> str:
    return json.dumps(
        {
            "normalized_term": query,
            "expanded_queries": [query],
            "inferred_meaning": f"a {query}",
            "semantic_type": "phenotype",
            "candidate_ontologies": [ontology],
            "preferred_ontology": ontology,
            "reasoning": "test",
            "confidence": 0.9,
        }
    )


def _select_json(code: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "is_unmapped": False,
            "selected_candidate_id": "C1",
            "selected_code": code,
            "confidence": confidence,
            "reasoning": "best match",
            "alternatives": [],
        }
    )


def _unmapped_json() -> str:
    return json.dumps(
        {
            "is_unmapped": True,
            "confidence": 0.0,
            "reasoning": "no good match",
            "alternatives": [],
        }
    )


def _real_pipeline(provider: BaseLLMProvider, search_tools: FakeSearchTools) -> PlannedPipeline:
    """A PlannedPipeline built entirely from real, unmodified stage classes
    -- only the LLM provider and the retrieval HTTP layer are fakes."""
    retriever = PublicOntologyRetriever(search_tools=search_tools)
    return PlannedPipeline(
        provider=provider,
        query_planner=QueryPlanner(provider),
        retrieval_router=RetrievalRouter(),
        public_retriever=retriever,
        candidate_normalizer=CandidateNormalizer(),
        candidate_merger=CandidateMerger(),
        llm_reranker=LLMReranker(provider),
        mapping_result_builder=MappingResultBuilder(),
    )


_COUGH_CANDIDATE = {
    "code": "HP:0012735",
    "term": "Cough",
    "ontology": "HPO",
    "score": 0.95,
    "definition": "",
    "source": "OLS",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single-term mapping (Bridge: OntologyMapper.map_term)
# ─────────────────────────────────────────────────────────────────────────────


def test_map_term_completes_full_seven_stage_pipeline_and_maps() -> None:
    provider = ScriptedProvider([_plan_json(), _select_json("HP:0012735")])
    tools = FakeSearchTools({"cough": [_COUGH_CANDIDATE]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline)

    result = mapper.map_term(
        source_term="cough_field",
        source_label="Do you have a cough?",
        source_type="radio",
        entity_type="phenotype",
        source_description="Whether the participant reports a cough",
    )

    assert isinstance(result, MappingResult)
    assert result.source_term == "cough_field"
    assert result.source_label == "Do you have a cough?"
    assert result.source_type == "radio"
    assert result.target_code == "HP:0012735"
    assert result.target_term == "Cough"
    assert result.ontology == "HPO"
    assert result.confidence == pytest.approx(0.9)
    assert result.logic_type == LogicType.RAG
    assert result.metadata is not None
    assert result.metadata.latency_ms is not None and result.metadata.latency_ms >= 0
    # Both real LLM-backed stages ran, in the documented order.
    assert len(provider.calls) == 2
    # Retrieval went through the real PublicOntologyRetriever -> SearchTools.
    assert tools.calls == [{"query": "cough", "ontology": "HPO", "top_k": 15}]


def test_map_term_unmapped_when_reranker_abstains_despite_grounded_retrieval() -> None:
    """The exact "metformin" scenario: public retrieval returns candidates,
    CandidateNormalizer/CandidateMerger succeed and retain them, but
    LLMReranker abstains -- the final result is UNKNOWN:UNMAPPED even though
    retrieval itself was fully grounded.

    Asserts BOTH semantic layers explicitly so they cannot silently collapse
    into each other in a future change:
      - the reranker's/MappingResult's own is_grounded/grounding_source
        (RerankDecision-derived, top level of pipeline metadata) reflect
        that no candidate was ultimately selected;
      - the nested RetrievalTrace's is_grounded/grounding_source reflect
        that retrieval itself succeeded and produced candidates.
    See RerankDecision.is_grounded / RetrievalTrace.is_grounded in models.py.
    """
    provider = ScriptedProvider([_plan_json(), _unmapped_json()])
    tools = FakeSearchTools({"cough": [_COUGH_CANDIDATE]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline)

    result = mapper.map_term(source_term="cough_field", source_label="Do you have a cough?")

    # Final mapping result.
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"
    assert result.confidence == 0.0

    # Result-level (RerankDecision-derived) grounding: no candidate selected.
    assert result.metadata is not None
    assert result.metadata.rag_debug is not None
    pipeline_info = result.metadata.rag_debug.candidates_retrieved[0]
    assert pipeline_info["is_grounded"] is False
    assert pipeline_info["grounding_source"] == GroundingSource.NONE.value

    # Retrieval-level (RetrievalTrace) grounding: retrieval itself succeeded.
    retrieval_trace = pipeline_info["retrieval_trace"]
    assert retrieval_trace["is_grounded"] is True
    assert retrieval_trace["grounding_source"] == GroundingSource.PUBLIC_API.value
    assert retrieval_trace["raw_candidate_count"] > 0
    assert retrieval_trace["merged_candidate_count"] > 0
    assert retrieval_trace["selected_candidate_code"] is None
    assert retrieval_trace["errors"] == []


def test_map_term_unmapped_when_no_candidates_retrieved() -> None:
    """Empty retrieval short-circuits the reranker call entirely (see
    LLMReranker.rerank's empty-candidate-list fast path) -- only the
    QueryPlanner call happens."""
    provider = ScriptedProvider([_plan_json()])
    tools = FakeSearchTools({"cough": []})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline)

    result = mapper.map_term(source_term="cough_field", source_label="Do you have a cough?")

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert len(provider.calls) == 1


def test_map_term_strict_target_ontology_propagates_through_real_pipeline() -> None:
    """A candidate retrieved only via an EFO-scoped search, but whose native
    ontology is HPO, is rejected under strict_target_ontology=True even
    though the (fake) LLM tried to select it -- LLMReranker enforces this
    Python-side, not the LLM."""
    imported_candidate = dict(_COUGH_CANDIDATE, requested_ontology="EFO")
    provider = ScriptedProvider([_plan_json(ontology="EFO"), _select_json("HP:0012735")])
    tools = FakeSearchTools({"cough": [imported_candidate]})
    pipeline = _real_pipeline(provider, tools)
    # target_ontology="EFO" (via a single-item ontologies list) is required
    # for strict_target_ontology to have anything to filter against.
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline, ontologies=["EFO"])

    result = mapper.map_term(
        source_term="cough_field",
        source_label="Do you have a cough?",
        strict_target_ontology=True,
    )

    # Rejected under strict mode: HP:0012735 is native HPO, not native EFO.
    assert result.target_code == "UNKNOWN:UNMAPPED"


def test_map_term_lenient_mode_keeps_efo_imported_candidate_for_contrast() -> None:
    """Same scenario as above with strict_target_ontology left at its default
    (False): the EFO imported-candidate exception applies and the mapping
    succeeds -- demonstrates strict mode is what changed the outcome, not an
    unrelated retrieval difference."""
    imported_candidate = dict(_COUGH_CANDIDATE, requested_ontology="EFO")
    provider = ScriptedProvider([_plan_json(ontology="EFO"), _select_json("HP:0012735")])
    tools = FakeSearchTools({"cough": [imported_candidate]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline, ontologies=["EFO"])

    result = mapper.map_term(source_term="cough_field", source_label="Do you have a cough?")

    assert result.target_code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Batch mapping (Bridge: OntologyMapper.map_data_dictionary)
# ─────────────────────────────────────────────────────────────────────────────


def test_map_data_dictionary_preserves_count_order_and_row_context() -> None:
    fever_candidate = {
        "code": "HP:0001945",
        "term": "Fever",
        "ontology": "HPO",
        "score": 0.9,
        "definition": "",
        "source": "OLS",
    }
    provider = ScriptedProvider(
        [
            _plan_json(query="cough"),
            _select_json("HP:0012735"),
            _plan_json(query="fever"),
            _select_json("HP:0001945"),
        ]
    )
    tools = FakeSearchTools({"cough": [_COUGH_CANDIDATE], "fever": [fever_candidate]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline)

    records = [
        {"field_name": "cough_field", "field_label": "cough", "field_type": "radio"},
        {"field_name": "fever_field", "field_label": "fever", "field_type": "radio"},
    ]
    batch = mapper.map_data_dictionary(records, study_id="STUDY-1")

    assert isinstance(batch, MappingBatch)
    assert batch.study_id == "STUDY-1"
    assert [r.source_term for r in batch.results] == ["cough_field", "fever_field"]
    assert [r.target_code for r in batch.results] == ["HP:0012735", "HP:0001945"]


def test_map_data_dictionary_skips_row_that_fails_without_aborting_batch() -> None:
    """A row whose planner call raises (malformed provider JSON on both the
    initial attempt and QueryPlanner's one retry) is skipped; the remaining
    rows still complete through the real pipeline."""
    provider = ScriptedProvider(
        ["not valid json 1", "not valid json 2", _plan_json(), _select_json("HP:0012735")]
    )
    tools = FakeSearchTools({"cough": [_COUGH_CANDIDATE]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline)

    records = [
        {"field_name": "bad_field", "field_label": "malformed", "field_type": "radio"},
        {"field_name": "cough_field", "field_label": "cough", "field_type": "radio"},
    ]
    batch = mapper.map_data_dictionary(records)

    assert len(batch.results) == 1
    assert batch.results[0].source_term == "cough_field"
    assert batch.results[0].target_code == "HP:0012735"


def test_map_data_dictionary_forwards_strict_target_ontology_through_real_pipeline() -> None:
    imported_candidate = dict(_COUGH_CANDIDATE, requested_ontology="EFO")
    provider = ScriptedProvider([_plan_json(ontology="EFO"), _select_json("HP:0012735")])
    tools = FakeSearchTools({"cough": [imported_candidate]})
    pipeline = _real_pipeline(provider, tools)
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=pipeline, ontologies=["EFO"])

    batch = mapper.map_data_dictionary(
        [{"field_name": "cough_field", "field_label": "cough", "field_type": "radio"}],
        strict_target_ontology=True,
    )

    assert len(batch.results) == 1
    assert batch.results[0].target_code == "UNKNOWN:UNMAPPED"
