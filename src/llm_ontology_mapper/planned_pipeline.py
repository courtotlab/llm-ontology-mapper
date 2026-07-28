"""
PlannedPipeline — Phase 9 orchestrator for the planned mapping pipeline.

This module wires the already-implemented planned components together without
changing OntologyMapper, AgenticMapper, SearchTools, provider behavior, or the
individual retriever contracts.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_ontology_mapper.candidate_merger import CandidateMerger
from llm_ontology_mapper.candidate_normalizer import (
    CandidateNormalizationError,
    CandidateNormalizer,
)
from llm_ontology_mapper.disabled_mapping import DisabledMappingRunner
from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.local_retriever import LocalSemanticRetriever
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilder
from llm_ontology_mapper.models import (
    GroundingSource,
    MappingResult,
    NormalizedCandidate,
    QueryPlan,
    RerankDecision,
    RetrievalMode,
    RetrievalRoutePlan,
    RetrievalTrace,
)
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.query_planner import QueryPlanner
from llm_ontology_mapper.retrieval_router import RetrievalRouter

logger = logging.getLogger(__name__)


class PlannedPipelineError(Exception):
    """Raised when a planned pipeline component fails during orchestration."""


class PlannedPipeline:
    """
    Orchestrate the planned ontology mapping components.

    Public/local flow:
        QueryPlanner → RetrievalRouter → retriever → CandidateNormalizer →
        CandidateMerger → LLMReranker → MappingResultBuilder.

    Disabled flow:
        QueryPlanner → RetrievalRouter → DisabledMappingRunner.

    Dependencies are injectable so unit tests can use fakes and avoid live
    LLM, public API, or SapBERT calls.
    """

    def __init__(
        self,
        *,
        provider: Any,
        query_planner: QueryPlanner | None = None,
        retrieval_router: RetrievalRouter | None = None,
        public_retriever: PublicOntologyRetriever | None = None,
        local_retriever: LocalSemanticRetriever | None = None,
        candidate_normalizer: CandidateNormalizer | None = None,
        candidate_merger: CandidateMerger | None = None,
        llm_reranker: LLMReranker | None = None,
        mapping_result_builder: MappingResultBuilder | None = None,
        disabled_mapping_runner: DisabledMappingRunner | None = None,
    ) -> None:
        self._provider = provider
        self._query_planner = query_planner or QueryPlanner(provider)
        self._retrieval_router = retrieval_router or RetrievalRouter()
        self._public_retriever = public_retriever or PublicOntologyRetriever()
        self._local_retriever = local_retriever or LocalSemanticRetriever()
        self._candidate_normalizer = candidate_normalizer or CandidateNormalizer()
        self._candidate_merger = candidate_merger or CandidateMerger()
        self._llm_reranker = llm_reranker or LLMReranker(provider)
        self._mapping_result_builder = mapping_result_builder or MappingResultBuilder()
        self._disabled_mapping_runner = disabled_mapping_runner or DisabledMappingRunner(provider)

    def map_term(
        self,
        source_term: str,
        source_label: str | None = None,
        source_type: str | None = None,
        clinical_area: str | None = None,
        target_ontology: str | None = None,
        allowed_target_ontologies: list[str] | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
        max_results_per_query: int = 10,
        max_candidates: int | None = None,
        max_alternatives: int = 5,
    ) -> MappingResult:
        """
        Map one source term through the planned pipeline.

        Args mirror the planned API.  retrieval_mode accepts the three
        user-facing values only: public, local, disabled.

        Raises:
            ValueError: invalid retrieval_mode string.
            PlannedPipelineError: component failure during orchestration.
        """
        mode = self._resolve_mode(retrieval_mode)

        query_plan = self._plan(
            source_term=source_term,
            source_label=source_label,
            clinical_area=clinical_area,
            target_ontology=target_ontology,
            allowed_target_ontologies=allowed_target_ontologies,
            retrieval_mode=mode,
        )
        route_plan = self._route(query_plan)

        if mode == RetrievalMode.DISABLED:
            return self._map_disabled(query_plan, source_type=source_type)

        raw_candidates = self._retrieve(
            query_plan=query_plan,
            route_plan=route_plan,
            max_results_per_query=max_results_per_query,
        )
        normalized_candidates, normalization_errors = self._normalize_raw_candidates(
            raw_candidates,
            query_plan=query_plan,
        )
        merged_candidates = self._merge_candidates(
            normalized_candidates,
            target_ontology_constraint=query_plan.target_ontology_constraint,
            allowed_target_ontologies=query_plan.allowed_target_ontologies,
            max_candidates=max_candidates,
        )
        rerank_decision = self._rerank(query_plan, merged_candidates)
        retrieval_trace = self._build_trace(
            query_plan=query_plan,
            route_plan=route_plan,
            raw_candidate_count=len(raw_candidates),
            merged_candidates=merged_candidates,
            rerank_decision=rerank_decision,
            normalization_errors=normalization_errors,
        )
        return self._build_result(
            query_plan=query_plan,
            rerank_decision=rerank_decision,
            candidates=merged_candidates,
            retrieval_trace=retrieval_trace,
            source_type=source_type,
            max_alternatives=max_alternatives,
        )

    @staticmethod
    def _resolve_mode(retrieval_mode: RetrievalMode | str) -> RetrievalMode:
        if isinstance(retrieval_mode, RetrievalMode):
            return retrieval_mode
        return RetrievalMode(str(retrieval_mode).lower())

    def _plan(
        self,
        *,
        source_term: str,
        source_label: str | None,
        clinical_area: str | None,
        target_ontology: str | None,
        allowed_target_ontologies: list[str] | None,
        retrieval_mode: RetrievalMode,
    ) -> QueryPlan:
        try:
            return self._query_planner.plan(
                source_term=source_term,
                source_label=source_label,
                clinical_area=clinical_area,
                target_ontology=target_ontology,
                allowed_target_ontologies=allowed_target_ontologies,
                retrieval_mode=retrieval_mode,
            )
        except Exception as exc:
            raise PlannedPipelineError("QueryPlanner failed during planned mapping.") from exc

    def _route(self, query_plan: QueryPlan) -> RetrievalRoutePlan:
        try:
            return self._retrieval_router.route(query_plan)
        except Exception as exc:
            raise PlannedPipelineError("RetrievalRouter failed during planned mapping.") from exc

    def _map_disabled(
        self,
        query_plan: QueryPlan,
        *,
        source_type: str | None,
    ) -> MappingResult:
        try:
            return self._disabled_mapping_runner.map(
                query_plan,
                source_type=source_type,
            )
        except Exception as exc:
            raise PlannedPipelineError(
                "DisabledMappingRunner failed during planned mapping."
            ) from exc

    def _retrieve(
        self,
        *,
        query_plan: QueryPlan,
        route_plan: RetrievalRoutePlan,
        max_results_per_query: int,
    ) -> list[dict[str, Any]]:
        try:
            if query_plan.retrieval_mode == RetrievalMode.PUBLIC:
                return self._public_retriever.retrieve(
                    query_plan,
                    route_plan=route_plan,
                    max_results_per_query=max_results_per_query,
                )
            if query_plan.retrieval_mode == RetrievalMode.LOCAL:
                return self._local_retriever.retrieve(
                    query_plan,
                    route_plan=route_plan,
                    max_results_per_query=max_results_per_query,
                )
        except Exception as exc:
            raise PlannedPipelineError(
                f"{query_plan.retrieval_mode.value} retrieval failed during planned mapping."
            ) from exc

        raise PlannedPipelineError(
            f"Unsupported retrieval_mode={query_plan.retrieval_mode.value!r} "
            "for grounded retrieval."
        )

    def _normalize_raw_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        *,
        query_plan: QueryPlan,
    ) -> tuple[list[NormalizedCandidate], list[dict[str, Any]]]:
        """Normalize raw candidates, skipping malformed records with a trace entry.

        Returns:
            (normalized, normalization_errors) where normalization_errors is a list
            of dicts describing each skipped candidate (raw dict + error message).

        Raises:
            PlannedPipelineError: if more than half the batch fails normalization,
                indicating systematic upstream breakage rather than isolated bad records.
            PlannedPipelineError: if any non-CandidateNormalizationError exception
                escapes normalization (unexpected component failure).
        """
        normalized: list[NormalizedCandidate] = []
        normalization_errors: list[dict[str, Any]] = []

        for raw in raw_candidates:
            try:
                normalized.append(
                    self._candidate_normalizer.normalize(
                        raw,
                        retrieval_mode=query_plan.retrieval_mode,
                        matched_query=_matched_query(raw, query_plan),
                        default_ontology=_default_ontology(raw, query_plan),
                        default_source=_default_source(raw, query_plan),
                    )
                )
            except CandidateNormalizationError as exc:
                logger.warning(
                    "CandidateNormalizer skipped malformed candidate: %s — raw=%r",
                    exc,
                    raw,
                )
                normalization_errors.append({"raw_candidate": raw, "error": str(exc)})
            except Exception as exc:
                raise PlannedPipelineError(
                    "CandidateNormalizer raised an unexpected error during planned mapping."
                ) from exc

        if raw_candidates and len(normalization_errors) > len(raw_candidates) / 2:
            raise PlannedPipelineError(
                f"CandidateNormalizer failed on {len(normalization_errors)} of "
                f"{len(raw_candidates)} candidates — systematic upstream breakage suspected."
            )

        return normalized, normalization_errors

    def _merge_candidates(
        self,
        candidates: list[NormalizedCandidate],
        *,
        target_ontology_constraint: str | None,
        allowed_target_ontologies: list[str] | None,
        max_candidates: int | None,
    ) -> list[NormalizedCandidate]:
        try:
            return self._candidate_merger.merge(
                candidates,
                target_ontology_constraint=target_ontology_constraint,
                allowed_target_ontologies=allowed_target_ontologies,
                max_candidates=max_candidates,
            )
        except Exception as exc:
            raise PlannedPipelineError("CandidateMerger failed during planned mapping.") from exc

    def _rerank(
        self,
        query_plan: QueryPlan,
        candidates: list[NormalizedCandidate],
    ) -> RerankDecision:
        try:
            return self._llm_reranker.rerank(query_plan, candidates)
        except Exception as exc:
            raise PlannedPipelineError("LLMReranker failed during planned mapping.") from exc

    def _build_result(
        self,
        *,
        query_plan: QueryPlan,
        rerank_decision: RerankDecision,
        candidates: list[NormalizedCandidate],
        retrieval_trace: RetrievalTrace,
        source_type: str | None,
        max_alternatives: int,
    ) -> MappingResult:
        try:
            return self._mapping_result_builder.build(
                query_plan=query_plan,
                rerank_decision=rerank_decision,
                candidates=candidates,
                retrieval_trace=retrieval_trace,
                source_type=source_type,
                max_alternatives=max_alternatives,
            )
        except Exception as exc:
            raise PlannedPipelineError(
                "MappingResultBuilder failed during planned mapping."
            ) from exc

    @staticmethod
    def _build_trace(
        *,
        query_plan: QueryPlan,
        route_plan: RetrievalRoutePlan,
        raw_candidate_count: int,
        merged_candidates: list[NormalizedCandidate],
        rerank_decision: RerankDecision,
        normalization_errors: list[dict[str, Any]] | None = None,
    ) -> RetrievalTrace:
        selected_code = (
            rerank_decision.selected_code
            if rerank_decision.is_grounded and not rerank_decision.is_unmapped
            else None
        )
        errors: list[dict[str, Any]] = list(normalization_errors) if normalization_errors else []
        return RetrievalTrace(
            query_plan=query_plan,
            retrieval_mode=query_plan.retrieval_mode,
            is_grounded=bool(merged_candidates),
            grounding_source=route_plan.grounding_source,
            retrieval_skipped=False,
            route_calls=list(route_plan.route_calls),
            raw_candidate_count=raw_candidate_count,
            merged_candidate_count=len(merged_candidates),
            errors=errors,
            selected_candidate_code=selected_code,
            retrieval_disabled_reason=None,
        )


def _matched_query(raw: dict[str, Any], query_plan: QueryPlan) -> str:
    value = raw.get("matched_query")
    if value is not None and str(value).strip():
        return str(value).strip()
    for query in query_plan.expanded_queries:
        if query and query.strip():
            return query.strip()
    return query_plan.original_term


def _default_ontology(raw: dict[str, Any], query_plan: QueryPlan) -> str | None:
    value = raw.get("requested_ontology")
    if value is not None and str(value).strip():
        return str(value).strip()
    if query_plan.target_ontology_constraint:
        return query_plan.target_ontology_constraint
    if query_plan.preferred_ontology:
        return query_plan.preferred_ontology
    if len(query_plan.candidate_ontologies) == 1:
        return query_plan.candidate_ontologies[0]
    return _ontology_from_code(raw.get("code"))


def _default_source(raw: dict[str, Any], query_plan: QueryPlan) -> str:
    value = raw.get("route_name") or raw.get("source")
    if value is not None and str(value).strip():
        return str(value).strip()
    if query_plan.retrieval_mode == RetrievalMode.PUBLIC:
        return GroundingSource.PUBLIC_API.value
    return GroundingSource.LOCAL_SAPBERT.value


def _ontology_from_code(code: Any) -> str | None:
    if code is None:
        return None
    text = str(code).strip()
    if ":" not in text:
        return None
    prefix = text.split(":", 1)[0].strip().upper()
    return prefix or None
