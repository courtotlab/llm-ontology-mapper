"""
Retrieval extensibility acceptance tests.

These tests define and prove the contributor contract for the two supported
extension cases on the planned pipeline's public retrieval stage:

  CASE 1 — an ontology already hosted by an existing retrieval source
           (e.g. OLS4) is added purely through configuration.
  CASE 2 — a brand-new retrieval endpoint is added by implementing one
           RetrievalSource adapter and registering it in one place.

Both drive a full PlannedPipeline (QueryPlanner -> RetrievalRouter ->
PublicOntologyRetriever -> CandidateNormalizer -> CandidateMerger ->
LLMReranker -> MappingResultBuilder) with the REAL, unmodified downstream
stage classes -- only the retrieval-source registration and ontology
configuration are test-local fakes. No production ontology_config.yaml
changes are required or made; a fake ontology is injected per-test via
PublicOntologyRetriever(ontologies_config=...) instead.

No live network calls: SearchTools/adapters are backed by a FakeSearchTools
double, exactly as in tests/test_public_retriever.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ontology_mapper.candidate_merger import CandidateMerger
from llm_ontology_mapper.candidate_normalizer import CandidateNormalizer
from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilder
from llm_ontology_mapper.models import MappingResult, RetrievalMode
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever, PublicRetrievalError
from llm_ontology_mapper.query_planner import QueryPlanner
from llm_ontology_mapper.retrieval_router import RetrievalRouter
from llm_ontology_mapper.retrieval_sources import (
    RetrievalConfigError,
    register_source,
    unregister_source,
)
from llm_ontology_mapper.retrieval_sources.config import (
    build_alias_index,
    resolve_retrieval_route,
    validate_retrieval_config,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Shared fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeSearchTools:
    """No-network SearchTools double, mirroring test_public_retriever.py's."""

    def __init__(self, ols_returns: list[dict[str, Any]] | None = None) -> None:
        self._ols_returns = ols_returns or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search_ols(
        self,
        query: str,
        ontology: str,
        top_k: int = 10,
        *,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(("search_ols", {"query": query, "ontology": ontology, "top_k": top_k}))
        return list(self._ols_returns)

    def search_loinc(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        raise AssertionError("search_loinc should not be called in these tests")

    def search_rxnorm(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        raise AssertionError("search_rxnorm should not be called in these tests")

    def search_icd10(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        raise AssertionError("search_icd10 should not be called in these tests")


class _TwoStageStubProvider(BaseLLMProvider):
    """Returns the QueryPlanner response on the first call, the LLMReranker
    response on the second -- matches PlannedPipeline's fixed call order for
    public/local retrieval_mode."""

    def __init__(self, query_plan_json: str, rerank_json: str) -> None:
        super().__init__(model="stub-model")
        self._responses = [query_plan_json, rerank_json]
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


def _query_plan_response(*, target_ontology: str) -> str:
    return json.dumps(
        {
            "normalized_term": "test concept",
            "expanded_queries": ["test concept"],
            "inferred_meaning": "a test concept",
            "semantic_type": "phenotype",
            "candidate_ontologies": [target_ontology],
            "preferred_ontology": target_ontology,
            "reasoning": "test",
            "confidence": 0.9,
        }
    )


def _rerank_response(*, candidate_id: str = "C1", code: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "is_unmapped": False,
            "selected_candidate_id": candidate_id,
            "selected_code": code,
            "confidence": confidence,
            "reasoning": "best match",
            "alternatives": [],
        }
    )


def _build_pipeline(
    *,
    provider: BaseLLMProvider,
    public_retriever: PublicOntologyRetriever,
) -> PlannedPipeline:
    """Assemble a PlannedPipeline from the REAL, unmodified downstream stage
    classes -- only the provider and the public retriever are test doubles."""
    return PlannedPipeline(
        provider=provider,
        query_planner=QueryPlanner(provider),
        retrieval_router=RetrievalRouter(),
        public_retriever=public_retriever,
        candidate_normalizer=CandidateNormalizer(),
        candidate_merger=CandidateMerger(),
        llm_reranker=LLMReranker(provider),
        mapping_result_builder=MappingResultBuilder(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 — new ontology on an existing retrieval source (OLS4), config-only
# ─────────────────────────────────────────────────────────────────────────────


def test_case1_new_ols4_ontology_requires_no_routing_code() -> None:
    """Adding ontology XYZ (hosted by the existing OLS4 source) is purely a
    configuration change: no `if ontology == "XYZ"`, no OLS_ONTOLOGY_MAP
    entry, no public_retriever.py branch. The fake `ontologies_config` below
    stands in for the one ontology_config.yaml edit a real contributor would
    make; the retrieval-source code (OLS4Source, SearchTools) is completely
    unmodified and untouched by this ontology addition.
    """
    xyz_candidate = {
        "code": "XYZ:0001",
        "term": "Test XYZ concept",
        "ontology": "XYZ",
        "score": 0.95,
        "definition": "",
        "source": "OLS",
    }
    fake_tools = FakeSearchTools(ols_returns=[xyz_candidate])
    fake_ontologies_config = {
        "XYZ": {
            "full_name": "XYZ Test Ontology",
            "curie_prefix": "XYZ",
            "retrieval": {"source": "ols4", "source_ontology_id": "xyz"},
        }
    }
    retriever = PublicOntologyRetriever(
        search_tools=fake_tools, ontologies_config=fake_ontologies_config
    )
    provider = _TwoStageStubProvider(
        query_plan_json=_query_plan_response(target_ontology="XYZ"),
        rerank_json=_rerank_response(code="XYZ:0001"),
    )
    pipeline = _build_pipeline(provider=provider, public_retriever=retriever)

    result = pipeline.map_term(
        "test_field",
        source_label="A test field",
        target_ontology="XYZ",
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert isinstance(result, MappingResult)
    assert result.target_code == "XYZ:0001"
    assert result.ontology == "XYZ"
    # The existing OLS4 adapter (via SearchTools.search_ols) served this
    # request -- resolved purely from ontologies_config, never from a
    # hard-coded ontology name in public_retriever.py.
    assert fake_tools.calls == [
        ("search_ols", {"query": "test concept", "ontology": "xyz", "top_k": 15})
    ]


def test_case1_unconfigured_ontology_is_rejected_not_silently_dropped() -> None:
    """An ontology absent from config is a clear PublicRetrievalError, not a
    silent 'no candidates found'."""
    fake_tools = FakeSearchTools()
    retriever = PublicOntologyRetriever(
        search_tools=fake_tools, ontologies_config={"HPO": {"retrieval": {"source": "ols4"}}}
    )

    with pytest.raises(PublicRetrievalError, match="not supported"):
        retriever._call_route("cough", "MONDO", 10)


# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 — a brand-new retrieval endpoint, registered once
# ─────────────────────────────────────────────────────────────────────────────


class FakeExampleOntologyAPISource:
    """A minimal RetrievalSource implementation for a hypothetical new
    endpoint. Registered through retrieval_sources.register_source -- the
    exact mechanism a real new endpoint adapter uses."""

    source_id = "example_ontology_api"
    route_name = "ExampleOntologyAPI"

    def __init__(self, tools: Any) -> None:
        self._tools = tools
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        ontology_id: str,
        top_k: int,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, "ontology_id": ontology_id, "top_k": top_k})
        return [
            {
                "code": "ABC:0001",
                "term": "Test ABC concept",
                "ontology": "ABC",
                "score": 0.88,
                "definition": "from the example endpoint",
                "source": "ExampleOntologyAPI",
            }
        ]


@pytest.fixture()
def registered_fake_source():
    register_source("example_ontology_api", FakeExampleOntologyAPISource)
    try:
        yield
    finally:
        unregister_source("example_ontology_api")


def test_case2_new_endpoint_flows_through_unmodified_downstream_stages(
    registered_fake_source: None,
) -> None:
    """A brand-new endpoint reaches CandidateNormalizer, CandidateMerger,
    LLMReranker, and MappingResultBuilder unmodified -- proving those stages
    require no changes to support a new retrieval source."""
    fake_ontologies_config = {
        "ABC": {
            "full_name": "ABC Test Vocabulary",
            "curie_prefix": "ABC",
            "retrieval": {"source": "example_ontology_api"},
        }
    }
    # search_tools is irrelevant here -- FakeExampleOntologyAPISource ignores
    # it -- but PublicOntologyRetriever still requires one.
    retriever = PublicOntologyRetriever(
        search_tools=FakeSearchTools(), ontologies_config=fake_ontologies_config
    )
    provider = _TwoStageStubProvider(
        query_plan_json=_query_plan_response(target_ontology="ABC"),
        rerank_json=_rerank_response(code="ABC:0001"),
    )
    pipeline = _build_pipeline(provider=provider, public_retriever=retriever)

    result = pipeline.map_term(
        "test_field",
        source_label="A test field",
        target_ontology="ABC",
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert isinstance(result, MappingResult)
    assert result.target_code == "ABC:0001"
    assert result.ontology == "ABC"
    assert result.logic_type.value == "rag"
    # The fake source itself was actually invoked -- not bypassed.
    fake_source = retriever._registry["example_ontology_api"]
    assert isinstance(fake_source, FakeExampleOntologyAPISource)
    assert fake_source.search_calls == [
        {"query": "test concept", "ontology_id": "ABC", "top_k": 15}
    ]


def test_case2_source_registered_through_public_registry_helper(
    registered_fake_source: None,
) -> None:
    """register_source/unregister_source is the one registration mechanism;
    a newly registered source is immediately resolvable by ontology config."""
    from llm_ontology_mapper.retrieval_sources.registry import known_source_ids

    assert "example_ontology_api" in known_source_ids()


# ─────────────────────────────────────────────────────────────────────────────
# Config validation — fail fast, never degrade into "no candidates found"
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_retrieval_source_fails_fast_at_construction() -> None:
    with pytest.raises(RetrievalConfigError, match="not a registered retrieval source"):
        PublicOntologyRetriever(
            search_tools=FakeSearchTools(),
            ontologies_config={"XYZ": {"retrieval": {"source": "totally_made_up_source"}}},
        )


def test_malformed_retrieval_block_raises_clear_error() -> None:
    with pytest.raises(RetrievalConfigError, match="retrieval must be a mapping"):
        PublicOntologyRetriever(
            search_tools=FakeSearchTools(),
            ontologies_config={"XYZ": {"retrieval": "ols4"}},
        )


def test_missing_source_field_raises_clear_error() -> None:
    with pytest.raises(RetrievalConfigError, match="retrieval.source is required"):
        PublicOntologyRetriever(
            search_tools=FakeSearchTools(),
            ontologies_config={"XYZ": {"retrieval": {}}},
        )


def test_duplicate_alias_across_two_ontologies_raises_clear_error() -> None:
    with pytest.raises(RetrievalConfigError, match="declared by both"):
        build_alias_index(
            {
                "XYZ": {"retrieval": {"source": "ols4", "aliases": ["SHARED"]}},
                "ABC": {"retrieval": {"source": "ols4", "aliases": ["SHARED"]}},
            }
        )


def test_validate_retrieval_config_passes_for_real_shipped_config() -> None:
    """The real ontology_config.yaml must always pass its own validation --
    this is what PublicOntologyRetriever() (no overrides) runs at
    construction time."""
    from llm_ontology_mapper.ontology_identity import get_ontology_config
    from llm_ontology_mapper.retrieval_sources.registry import known_source_ids

    validate_retrieval_config(
        get_ontology_config().get("ontologies", {}), known_source_ids()
    )


def test_ontology_without_retrieval_block_resolves_to_none() -> None:
    """An ontology present in config but with no `retrieval` block is simply
    not routable -- same "unsupported" signal as an ontology absent
    entirely, not an error."""
    assert resolve_retrieval_route(
        "NOROUTE", ontologies_config={"NOROUTE": {"full_name": "No retrieval configured"}}
    ) is None


def test_default_public_ontology_retriever_construction_validates_real_config() -> None:
    """Constructing PublicOntologyRetriever with no overrides (the production
    path) validates the real shipped ontology_config.yaml at construction
    time -- this must never raise."""
    PublicOntologyRetriever(search_tools=FakeSearchTools())
