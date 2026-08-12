"""
Unit tests for PlannedPipeline (Phase 9).

All dependencies are injected fakes or pure in-memory components.  No live LLM,
public ontology API, or SapBERT service is called.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from llm_ontology_mapper.candidate_merger import CandidateMerger
from llm_ontology_mapper.candidate_normalizer import CandidateNormalizer
from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilder
from llm_ontology_mapper.models import (
    GroundingSource,
    LogicType,
    MappingMetadata,
    MappingResult,
    NormalizedCandidate,
    QueryPlan,
    RAGDebugInfo,
    RerankDecision,
    RetrievalMode,
    RetrievalRoutePlan,
)
from llm_ontology_mapper.planned_pipeline import PlannedPipeline, PlannedPipelineError
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.retrieval_router import RetrievalRouter

pytestmark = pytest.mark.unit


class _StubProvider(BaseLLMProvider):
    def __init__(self, response_content: str = "{}", model: str = "stub") -> None:
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


class _Planner:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_kwargs: dict[str, Any] | None = None

    def plan(
        self,
        source_term: str,
        source_label: str | None = None,
        source_description: str | None = None,
        source_type: str | None = None,
        clinical_area: str | None = None,
        target_ontology: str | None = None,
        allowed_target_ontologies: list[str] | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
    ) -> QueryPlan:
        self.calls.append("planner")
        self.last_kwargs = {
            "source_term": source_term,
            "source_label": source_label,
            "source_description": source_description,
            "source_type": source_type,
            "clinical_area": clinical_area,
            "target_ontology": target_ontology,
            "allowed_target_ontologies": allowed_target_ontologies,
            "retrieval_mode": retrieval_mode,
        }
        mode = (
            retrieval_mode
            if isinstance(retrieval_mode, RetrievalMode)
            else RetrievalMode(retrieval_mode)
        )
        target = target_ontology.upper() if target_ontology else "LOINC"
        return QueryPlan(
            original_term=source_term,
            original_label=source_label,
            source_description=source_description,
            source_type=source_type,
            normalized_term=source_term.replace("_", " "),
            expanded_queries=["systolic blood pressure"],
            inferred_meaning="systolic blood pressure",
            semantic_type="measurement",
            candidate_ontologies=[target],
            preferred_ontology=target,
            allowed_target_ontologies=allowed_target_ontologies,
            retrieval_mode=mode,
            target_ontology_constraint=target_ontology.upper() if target_ontology else None,
            retrieval_disabled_reason=(
                "Retrieval disabled by caller" if mode == RetrievalMode.DISABLED else None
            ),
        )


class _SnomedAliasPlanner:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def plan(
        self,
        source_term: str,
        source_label: str | None = None,
        source_description: str | None = None,
        source_type: str | None = None,
        clinical_area: str | None = None,
        target_ontology: str | None = None,
        allowed_target_ontologies: list[str] | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
    ) -> QueryPlan:
        self.last_kwargs = {
            "source_term": source_term,
            "source_label": source_label,
            "source_description": source_description,
            "source_type": source_type,
            "clinical_area": clinical_area,
            "target_ontology": target_ontology,
            "allowed_target_ontologies": allowed_target_ontologies,
            "retrieval_mode": retrieval_mode,
        }
        mode = (
            retrieval_mode
            if isinstance(retrieval_mode, RetrievalMode)
            else RetrievalMode(retrieval_mode)
        )
        return QueryPlan(
            original_term=source_term,
            original_label=source_label,
            source_description=source_description,
            source_type=source_type,
            normalized_term="inhaled nitric oxide",
            expanded_queries=["inhaled nitric oxide"],
            inferred_meaning="inhaled nitric oxide therapy",
            semantic_type="clinical_concept",
            candidate_ontologies=["SNOMED-CT"],
            preferred_ontology="SNOMED-CT",
            allowed_target_ontologies=["SNOMED"],
            retrieval_mode=mode,
            target_ontology_constraint="SNOMED-CT",
        )


class _Router:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_plan: QueryPlan | None = None

    def route(self, query_plan: QueryPlan) -> RetrievalRoutePlan:
        self.calls.append("router")
        self.last_plan = query_plan
        if query_plan.retrieval_mode == RetrievalMode.DISABLED:
            return RetrievalRoutePlan(
                retrieval_mode=RetrievalMode.DISABLED,
                is_grounded_mode=False,
                retrieval_skipped=True,
                grounding_source=GroundingSource.NONE,
                queries=list(query_plan.expanded_queries),
                target_ontology_constraint=query_plan.target_ontology_constraint,
                allowed_target_ontologies=query_plan.allowed_target_ontologies,
                candidate_ontologies=list(query_plan.candidate_ontologies),
                route_calls=[],
                retrieval_disabled_reason=query_plan.retrieval_disabled_reason,
            )
        grounding_source = (
            GroundingSource.PUBLIC_API
            if query_plan.retrieval_mode == RetrievalMode.PUBLIC
            else GroundingSource.LOCAL_SAPBERT
        )
        route_name = (
            "public_api" if query_plan.retrieval_mode == RetrievalMode.PUBLIC else "local_sapbert"
        )
        return RetrievalRoutePlan(
            retrieval_mode=query_plan.retrieval_mode,
            is_grounded_mode=True,
            retrieval_skipped=False,
            grounding_source=grounding_source,
            queries=list(query_plan.expanded_queries),
            target_ontology_constraint=query_plan.target_ontology_constraint,
            allowed_target_ontologies=query_plan.allowed_target_ontologies,
            candidate_ontologies=list(query_plan.candidate_ontologies),
            route_calls=[
                {
                    "route": route_name,
                    "query": query_plan.expanded_queries[0],
                    "target_ontology": query_plan.target_ontology_constraint,
                    "allowed_target_ontologies": query_plan.allowed_target_ontologies,
                    "candidate_ontologies": list(query_plan.candidate_ontologies),
                }
            ],
        )


class _Retriever:
    def __init__(
        self,
        calls: list[str],
        name: str,
        raw_candidates: list[dict[str, Any]] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.name = name
        self.raw_candidates = raw_candidates or []
        self.exc = exc
        self.last_query_plan: QueryPlan | None = None
        self.last_route_plan: RetrievalRoutePlan | None = None
        self.last_max_results_per_query: int | None = None

    def retrieve(
        self,
        query_plan: QueryPlan,
        *,
        route_plan: RetrievalRoutePlan | None = None,
        max_results_per_query: int = 10,
    ) -> list[dict[str, Any]]:
        self.calls.append(self.name)
        self.last_query_plan = query_plan
        self.last_route_plan = route_plan
        self.last_max_results_per_query = max_results_per_query
        if self.exc is not None:
            raise self.exc
        return list(self.raw_candidates)


class _ForbiddenRetriever:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("forbidden retriever was called")


class _SnomedSearchTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "ontology": ontology, "top_k": top_k})
        return [
            {
                "code": "SNOMEDCT:123456",
                "term": "Inhaled nitric oxide",
                "ontology": "SNOMED-CT",
                "score": 0.97,
                "source": "OLS",
            }
        ]

    def search_loinc(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("LOINC route should not be called")

    def search_rxnorm(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("RxNorm route should not be called")

    def search_icd10(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("ICD route should not be called")


class _RecordingNormalizer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.inner = CandidateNormalizer()
        self.calls_seen: list[dict[str, Any]] = []

    def normalize(self, raw_candidate: dict[str, Any], **kwargs: Any) -> NormalizedCandidate:
        self.calls.append("normalizer")
        self.calls_seen.append({"raw_candidate": raw_candidate, **kwargs})
        return self.inner.normalize(raw_candidate, **kwargs)


class _RecordingMerger:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.inner = CandidateMerger()
        self.last_candidates: list[NormalizedCandidate] | None = None
        self.last_target_ontology_constraint: str | None = None
        self.last_allowed_target_ontologies: list[str] | None = None
        self.last_max_candidates: int | None = None

    def merge(
        self,
        candidates,
        *,
        target_ontology_constraint: str | None = None,
        allowed_target_ontologies: list[str] | None = None,
        max_candidates: int | None = None,
    ) -> list[NormalizedCandidate]:
        self.calls.append("merger")
        self.last_candidates = list(candidates)
        self.last_target_ontology_constraint = target_ontology_constraint
        self.last_allowed_target_ontologies = allowed_target_ontologies
        self.last_max_candidates = max_candidates
        return self.inner.merge(
            self.last_candidates,
            target_ontology_constraint=target_ontology_constraint,
            allowed_target_ontologies=allowed_target_ontologies,
            max_candidates=max_candidates,
        )


class _Reranker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_plan: QueryPlan | None = None
        self.last_candidates: list[NormalizedCandidate] | None = None

    def rerank(
        self,
        query_plan: QueryPlan,
        candidates: list[NormalizedCandidate],
    ) -> RerankDecision:
        self.calls.append("reranker")
        self.last_plan = query_plan
        self.last_candidates = list(candidates)
        if not candidates:
            return _unmapped_decision(query_plan.retrieval_mode)
        grounding_source = (
            GroundingSource.PUBLIC_API
            if query_plan.retrieval_mode == RetrievalMode.PUBLIC
            else GroundingSource.LOCAL_SAPBERT
        )
        return RerankDecision(
            selected_code=candidates[0].code,
            selected_candidate_id="C1",
            is_unmapped=False,
            is_grounded=True,
            grounding_source=grounding_source,
            retrieval_mode=query_plan.retrieval_mode,
            confidence=0.91,
            reasoning="Selected best candidate.",
            alternative_codes=[],
            policy="production_grounded",
        )


class _RecordingBuilder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.inner = MappingResultBuilder()
        self.last_kwargs: dict[str, Any] | None = None

    def build(self, **kwargs: Any) -> MappingResult:
        self.calls.append("builder")
        self.last_kwargs = kwargs
        return self.inner.build(**kwargs)


class _DisabledRunner:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_plan: QueryPlan | None = None
        self.last_source_type: str | None = None

    def map(
        self,
        query_plan: QueryPlan,
        *,
        source_type: str | None = None,
    ) -> MappingResult:
        self.calls.append("disabled")
        self.last_plan = query_plan
        self.last_source_type = source_type
        return _disabled_result(query_plan, source_type=source_type)


class _ForbiddenComponent:
    def normalize(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("normalizer should not be called")

    def merge(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("merger should not be called")

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reranker should not be called")

    def build(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("builder should not be called")


def _public_raw(**overrides: Any) -> dict[str, Any]:
    data = {
        "code": "LOINC:8480-6",
        "term": "Systolic blood pressure",
        "ontology": "LOINC",
        "score": 0.95,
        "source": "LOINC-Search-API",
        "matched_query": "systolic blood pressure",
        "requested_ontology": "LOINC",
        "route_name": "LOINC-Search-API",
    }
    data.update(overrides)
    return data


def _local_raw(**overrides: Any) -> dict[str, Any]:
    data = {
        "code": "LOINC:8480-6",
        "term": "Systolic blood pressure",
        "ontology": "LOINC",
        "score": 0.93,
        "source": "SapBERT",
        "matched_query": "systolic blood pressure",
        "requested_ontology": "LOINC",
        "route_name": "local_sapbert",
    }
    data.update(overrides)
    return data


def _unmapped_decision(mode: RetrievalMode) -> RerankDecision:
    return RerankDecision(
        selected_code=None,
        selected_candidate_id=None,
        is_unmapped=True,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=mode,
        confidence=0.0,
        reasoning="No candidates were provided for reranking.",
        alternative_codes=[],
        policy="production_grounded",
    )


def _disabled_result(
    query_plan: QueryPlan,
    *,
    source_type: str | None = None,
) -> MappingResult:
    return MappingResult(
        source_term=query_plan.original_term,
        source_label=query_plan.original_label,
        source_type=source_type,
        target_code="UNKNOWN:UNMAPPED",
        target_term="UNMAPPED",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.LLM,
        metadata=MappingMetadata(
            model="stub",
            provider="stub",
            rag_debug=RAGDebugInfo(
                query_sent=query_plan.original_term,
                candidates_retrieved=[
                    {
                        "retrieval_mode": "disabled",
                        "is_grounded": False,
                        "grounding_source": "none",
                        "policy": "disabled_llm_only",
                        "retrieval_skipped": True,
                    }
                ],
                top_k=0,
            ),
        ),
    )


def _pipeline(
    *,
    calls: list[str] | None = None,
    public_raw: list[dict[str, Any]] | None = None,
    local_raw: list[dict[str, Any]] | None = None,
    public_retriever: Any | None = None,
    local_retriever: Any | None = None,
    normalizer: Any | None = None,
    merger: Any | None = None,
    reranker: Any | None = None,
    builder: Any | None = None,
    disabled_runner: Any | None = None,
) -> tuple[PlannedPipeline, dict[str, Any]]:
    call_log = calls if calls is not None else []
    planner = _Planner(call_log)
    router = _Router(call_log)
    public = public_retriever or _Retriever(
        call_log,
        "public",
        public_raw if public_raw is not None else [_public_raw()],
    )
    local = local_retriever or _Retriever(
        call_log,
        "local",
        local_raw if local_raw is not None else [_local_raw()],
    )
    rec_normalizer = normalizer or _RecordingNormalizer(call_log)
    rec_merger = merger or _RecordingMerger(call_log)
    rec_reranker = reranker or _Reranker(call_log)
    rec_builder = builder or _RecordingBuilder(call_log)
    disabled = disabled_runner or _DisabledRunner(call_log)
    pipeline = PlannedPipeline(
        provider=_StubProvider(),
        query_planner=planner,
        retrieval_router=router,
        public_retriever=public,
        local_retriever=local,
        candidate_normalizer=rec_normalizer,
        candidate_merger=rec_merger,
        llm_reranker=rec_reranker,
        mapping_result_builder=rec_builder,
        disabled_mapping_runner=disabled,
    )
    return pipeline, {
        "calls": call_log,
        "planner": planner,
        "router": router,
        "public": public,
        "local": local,
        "normalizer": rec_normalizer,
        "merger": rec_merger,
        "reranker": rec_reranker,
        "builder": rec_builder,
        "disabled": disabled,
    }


def _meta(result: MappingResult) -> dict[str, Any]:
    assert result.metadata is not None
    assert result.metadata.rag_debug is not None
    return result.metadata.rag_debug.candidates_retrieved[0]


def test_public_mode_runs_full_grounded_pipeline_in_order() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    assert parts["calls"] == [
        "planner",
        "router",
        "public",
        "normalizer",
        "merger",
        "reranker",
        "builder",
    ]
    assert result.target_code == "LOINC:8480-6"


def test_local_mode_runs_full_grounded_pipeline_in_order() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    assert parts["calls"] == [
        "planner",
        "router",
        "local",
        "normalizer",
        "merger",
        "reranker",
        "builder",
    ]
    assert result.target_code == "LOINC:8480-6"


def test_disabled_mode_runs_planner_router_disabled_runner_only() -> None:
    pipeline, parts = _pipeline(
        public_retriever=_ForbiddenRetriever(),
        local_retriever=_ForbiddenRetriever(),
        normalizer=_ForbiddenComponent(),
        merger=_ForbiddenComponent(),
        reranker=_ForbiddenComponent(),
        builder=_ForbiddenComponent(),
    )

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.DISABLED)

    assert parts["calls"] == ["planner", "router", "disabled"]
    assert result.logic_type == LogicType.LLM


def test_public_mode_does_not_call_local_retriever() -> None:
    pipeline, _ = _pipeline(local_retriever=_ForbiddenRetriever())
    result = pipeline.map_term("sys_bp", retrieval_mode="public")
    assert result.target_code == "LOINC:8480-6"


def test_local_mode_does_not_call_public_retriever() -> None:
    pipeline, _ = _pipeline(public_retriever=_ForbiddenRetriever())
    result = pipeline.map_term("sys_bp", retrieval_mode="local")
    assert result.target_code == "LOINC:8480-6"


def test_target_ontology_passed_to_query_planner_and_metadata() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term("sys_bp", target_ontology="loinc")

    assert parts["planner"].last_kwargs["target_ontology"] == "loinc"
    assert parts["planner"].last_kwargs["allowed_target_ontologies"] is None
    assert _meta(result)["target_ontology_constraint"] == "LOINC"


def test_native_efo_candidate_survives_planned_public_hard_target() -> None:
    pipeline, parts = _pipeline(
        public_raw=[
            _public_raw(
                code="EFO:0000408",
                term="disease",
                ontology="EFO",
                source="OLS",
                requested_ontology="EFO",
                route_name="OLS",
            )
        ]
    )

    result = pipeline.map_term(
        "disease",
        target_ontology="EFO",
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert parts["planner"].last_kwargs["target_ontology"] == "EFO"
    assert parts["merger"].last_target_ontology_constraint == "EFO"
    assert parts["reranker"].last_candidates is not None
    assert parts["reranker"].last_candidates[0].ontology == "EFO"
    assert result.target_code == "EFO:0000408"
    assert result.ontology == "EFO"


def test_imported_efo_candidate_survives_planned_public_hard_target() -> None:
    pipeline, parts = _pipeline(
        public_raw=[
            _public_raw(
                code="MONDO:0004975",
                term="asthma",
                ontology="MONDO",
                source="OLS",
                requested_ontology="EFO",
                route_name="OLS",
            )
        ]
    )

    result = pipeline.map_term(
        "asthma",
        target_ontology="EFO",
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert parts["merger"].last_target_ontology_constraint == "EFO"
    assert parts["reranker"].last_candidates is not None
    assert parts["reranker"].last_candidates[0].ontology == "MONDO"
    assert parts["reranker"].last_candidates[0].retrieved_from_ontologies == ["EFO"]
    assert result.target_code == "MONDO:0004975"
    assert result.ontology == "MONDO"


def test_snomed_alias_from_planner_routes_through_public_snomed() -> None:
    planner = _SnomedAliasPlanner()
    search_tools = _SnomedSearchTools()
    pipeline = PlannedPipeline(
        provider=_StubProvider(),
        query_planner=planner,
        retrieval_router=RetrievalRouter(),
        public_retriever=PublicOntologyRetriever(search_tools=search_tools),
        local_retriever=_ForbiddenRetriever(),
        candidate_normalizer=CandidateNormalizer(),
        candidate_merger=CandidateMerger(),
        llm_reranker=_Reranker([]),
        mapping_result_builder=MappingResultBuilder(),
        disabled_mapping_runner=_DisabledRunner([]),
    )

    result = pipeline.map_term(
        "inhaled_no",
        source_label="Inhaled nitric oxide",
        target_ontology="SNOMED",
        allowed_target_ontologies=["SNOMED"],
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert planner.last_kwargs is not None
    assert planner.last_kwargs["target_ontology"] == "SNOMED"
    assert planner.last_kwargs["allowed_target_ontologies"] == ["SNOMED"]
    assert search_tools.calls == [
        {"query": "inhaled nitric oxide", "ontology": "SNOMED", "top_k": 10}
    ]
    assert result.target_code == "SNOMEDCT:123456"
    assert result.ontology == "SNOMED-CT"
    assert result.metadata is not None
    assert result.metadata.rag_debug is not None
    trace = result.metadata.rag_debug.candidates_retrieved[0]["retrieval_trace"]
    assert trace["route_calls"][0]["target_ontology"] == "SNOMED-CT"
    assert trace["route_calls"][0]["allowed_target_ontologies"] == ["SNOMED"]


def test_allowed_target_ontologies_passed_to_query_planner_and_route() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term(
        "sys_bp",
        allowed_target_ontologies=["LOINC", "HPO"],
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert parts["planner"].last_kwargs["allowed_target_ontologies"] == [
        "LOINC",
        "HPO",
    ]
    assert parts["router"].last_plan.allowed_target_ontologies == ["LOINC", "HPO"]


def test_allowed_target_ontologies_filter_all_candidates_to_unmapped() -> None:
    pipeline, parts = _pipeline(public_raw=[_public_raw(ontology="LOINC")])

    result = pipeline.map_term(
        "sys_bp",
        allowed_target_ontologies=["HPO"],
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert parts["merger"].last_allowed_target_ontologies == ["HPO"]
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_clinical_area_and_source_label_passed_to_query_planner() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term(
        "sys_bp",
        source_label="Systolic BP",
        clinical_area="cardiology",
    )

    assert parts["planner"].last_kwargs["source_label"] == "Systolic BP"
    assert parts["planner"].last_kwargs["clinical_area"] == "cardiology"
    assert result.source_label == "Systolic BP"


def test_source_description_and_type_passed_to_query_planner() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term(
        "creat",
        source_label="Serum creatinine",
        source_description="Most recent serum creatinine result collected at enrolment",
        source_type="decimal",
        clinical_area="measurement",
    )

    assert parts["planner"].last_kwargs["source_description"] == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert parts["planner"].last_kwargs["source_type"] == "decimal"
    assert parts["router"].last_plan.source_description == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert parts["router"].last_plan.source_type == "decimal"
    assert result.source_type == "decimal"


def test_missing_source_context_remains_none_for_query_planner() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term("sys_bp")

    assert parts["planner"].last_kwargs["source_description"] is None
    assert parts["planner"].last_kwargs["source_type"] is None
    assert parts["router"].last_plan.source_description is None
    assert parts["router"].last_plan.source_type is None


def test_source_type_passed_to_mapping_result_builder() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term("sys_bp", source_type="integer")

    assert parts["builder"].last_kwargs["source_type"] == "integer"
    assert result.source_type == "integer"


def test_source_type_passed_to_disabled_mapping_runner() -> None:
    pipeline, parts = _pipeline()

    result = pipeline.map_term(
        "sys_bp",
        source_type="radio",
        retrieval_mode=RetrievalMode.DISABLED,
    )

    assert parts["disabled"].last_source_type == "radio"
    assert result.source_type == "radio"


def test_public_raw_candidates_are_normalized() -> None:
    pipeline, parts = _pipeline(public_raw=[_public_raw(code="LOINC:8462-4")])

    pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    seen = parts["normalizer"].calls_seen[0]
    assert seen["raw_candidate"]["code"] == "LOINC:8462-4"
    assert seen["retrieval_mode"] == RetrievalMode.PUBLIC


def test_local_raw_candidates_are_normalized() -> None:
    pipeline, parts = _pipeline(local_raw=[_local_raw(code="LOINC:8462-4")])

    pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    seen = parts["normalizer"].calls_seen[0]
    assert seen["raw_candidate"]["code"] == "LOINC:8462-4"
    assert seen["retrieval_mode"] == RetrievalMode.LOCAL


def test_merged_candidates_passed_to_reranker() -> None:
    pipeline, parts = _pipeline(
        public_raw=[
            _public_raw(score=0.1),
            _public_raw(code="LOINC:8480-6", score=0.9),
        ]
    )

    pipeline.map_term("sys_bp")

    assert parts["reranker"].last_candidates is not None
    assert len(parts["reranker"].last_candidates) == 1
    assert parts["reranker"].last_candidates[0].code == "LOINC:8480-6"


def test_rerank_decision_passed_to_mapping_result_builder() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term("sys_bp")

    assert parts["builder"].last_kwargs is not None
    decision = parts["builder"].last_kwargs["rerank_decision"]
    assert decision.selected_code == "LOINC:8480-6"


def test_retrieval_trace_includes_counts_and_route_calls() -> None:
    pipeline, parts = _pipeline(public_raw=[_public_raw(), _public_raw(code="LOINC:8462-4")])

    pipeline.map_term("sys_bp")

    trace = parts["builder"].last_kwargs["retrieval_trace"]
    assert trace.raw_candidate_count == 2
    assert trace.merged_candidate_count == 2
    assert trace.route_calls[0]["route"] == "public_api"


def test_empty_public_retrieval_produces_unmapped_result() -> None:
    pipeline, _ = _pipeline(public_raw=[])

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.target_term == "UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_normalize_raw_candidates_skip_one_bad_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One malformed candidate (blank term) + two valid → the two valid normalize,
    the bad one is captured in the retrieval trace errors, no PlannedPipelineError."""
    bad = _public_raw(term="")
    good1 = _public_raw(code="LOINC:8480-6", term="Systolic blood pressure")
    good2 = _public_raw(code="LOINC:8462-4", term="Diastolic blood pressure")

    pipeline, parts = _pipeline(public_raw=[bad, good1, good2])

    with caplog.at_level(logging.WARNING):
        result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    # The two valid candidates flowed through — result is grounded.
    assert result.target_code != "UNKNOWN:UNMAPPED"

    # The skip was logged at WARNING.
    assert "skipped malformed candidate" in caplog.text

    # The skipped record is surfaced in the retrieval trace errors.
    trace = parts["builder"].last_kwargs["retrieval_trace"]
    assert len(trace.errors) == 1
    assert trace.errors[0]["raw_candidate"] == bad
    assert "error" in trace.errors[0]


def test_normalize_raw_candidates_systematic_failure_raises() -> None:
    """More than half the batch failing normalization must raise PlannedPipelineError."""
    # Three bad candidates (blank term), one good — >50% failure rate.
    bad = _public_raw(term="")
    good = _public_raw(code="LOINC:8480-6", term="Systolic blood pressure")

    pipeline, _ = _pipeline(public_raw=[bad, bad, bad, good])

    with pytest.raises(PlannedPipelineError, match="systematic"):
        pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)


def test_empty_local_retrieval_produces_unmapped_result() -> None:
    pipeline, _ = _pipeline(local_raw=[])

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.target_term == "UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_empty_candidate_path_does_not_call_real_reranker_provider() -> None:
    calls: list[str] = []
    provider = _StubProvider('{"unused": true}')
    planner = _Planner(calls)
    router = _Router(calls)
    pipeline = PlannedPipeline(
        provider=provider,
        query_planner=planner,
        retrieval_router=router,
        public_retriever=_Retriever(calls, "public", []),
        local_retriever=_ForbiddenRetriever(),
        candidate_normalizer=_RecordingNormalizer(calls),
        candidate_merger=_RecordingMerger(calls),
        llm_reranker=LLMReranker(provider),
        mapping_result_builder=_RecordingBuilder(calls),
        disabled_mapping_runner=_DisabledRunner(calls),
    )

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert provider.calls == []


def test_public_retriever_error_is_wrapped_with_cause() -> None:
    exc = RuntimeError("public exploded")
    pipeline, _ = _pipeline(public_retriever=_Retriever([], "public", exc=exc))

    with pytest.raises(PlannedPipelineError) as caught:
        pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    assert caught.value.__cause__ is exc


def test_local_retriever_error_is_wrapped_with_cause() -> None:
    exc = RuntimeError("local exploded")
    pipeline, _ = _pipeline(local_retriever=_Retriever([], "local", exc=exc))

    with pytest.raises(PlannedPipelineError) as caught:
        pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    assert caught.value.__cause__ is exc


@pytest.mark.parametrize("mode", ["public", "local", "disabled"])
def test_retrieval_mode_string_inputs_work(mode: str) -> None:
    pipeline, _ = _pipeline()
    result = pipeline.map_term("sys_bp", retrieval_mode=mode)
    assert isinstance(result, MappingResult)


def test_invalid_retrieval_mode_raises_value_error() -> None:
    pipeline, _ = _pipeline()
    with pytest.raises(ValueError):
        pipeline.map_term("sys_bp", retrieval_mode="both")


def test_no_both_mode_is_accepted() -> None:
    pipeline, _ = _pipeline()
    with pytest.raises(ValueError):
        pipeline.map_term("sys_bp", retrieval_mode="both")


def test_max_results_per_query_passed_to_retriever() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term("sys_bp", max_results_per_query=7)

    assert parts["public"].last_max_results_per_query == 7


def test_max_candidates_passed_to_candidate_merger() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term("sys_bp", max_candidates=3)

    assert parts["merger"].last_max_candidates == 3


def test_max_alternatives_passed_to_mapping_result_builder() -> None:
    pipeline, parts = _pipeline()

    pipeline.map_term("sys_bp", max_alternatives=4)

    assert parts["builder"].last_kwargs["max_alternatives"] == 4


def test_public_mapping_result_has_grounded_metadata() -> None:
    pipeline, _ = _pipeline()

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    meta = _meta(result)
    assert meta["retrieval_mode"] == "public"
    assert meta["is_grounded"] is True
    assert meta["grounding_source"] == "public_api"


def test_local_mapping_result_has_grounded_metadata() -> None:
    pipeline, _ = _pipeline()

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    meta = _meta(result)
    assert meta["retrieval_mode"] == "local"
    assert meta["is_grounded"] is True
    assert meta["grounding_source"] == "local_sapbert"


def test_disabled_mapping_result_has_ungrounded_metadata() -> None:
    pipeline, _ = _pipeline()

    result = pipeline.map_term("sys_bp", retrieval_mode=RetrievalMode.DISABLED)

    meta = _meta(result)
    assert meta["retrieval_mode"] == "disabled"
    assert meta["is_grounded"] is False
    assert meta["grounding_source"] == "none"
    assert meta["retrieval_skipped"] is True
