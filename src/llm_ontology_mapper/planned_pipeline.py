"""
PlannedPipeline — Phase 9 orchestrator for the planned mapping pipeline.

This module wires the already-implemented planned components together without
changing OntologyMapper, AgenticMapper, SearchTools, provider behavior, or the
individual retriever contracts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any, TypeVar

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
from llm_ontology_mapper.providers import LLMCallConfig
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.query_planner import QueryPlanner
from llm_ontology_mapper.retrieval_router import RetrievalRouter

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class PlannedPipelineError(Exception):
    """Raised when a planned pipeline component fails during orchestration.

    Carries whatever pipeline_timings/pipeline_usage/retrieval telemetry had
    already been recorded before the failing stage raised (see
    PlannedPipeline.map_term()), so a caller catching this exception -- e.g.
    the benchmark runner -- can still report the cost/latency of
    billed-but-failed LLM calls, and any retrieval retries/errors already
    encountered, instead of losing that telemetry entirely. All three default
    to None and are only populated by map_term() itself; never fabricated.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.partial_timings: dict[str, float] | None = None
        self.partial_usage: dict[str, Any] | None = None
        self.partial_retrieval_diagnostics: dict[str, Any] | None = None


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
        llm_call_config: LLMCallConfig | None = None,
    ) -> None:
        """
        llm_call_config: explicit temperature/seed/reasoning_effort override
            forwarded to the default QueryPlanner and LLMReranker so every
            applicable LLM call in the pipeline uses identical sampling
            parameters (see providers.LLMCallConfig). Ignored when
            query_planner / llm_reranker are explicitly injected — those
            instances own their own configuration.
        """
        self._provider = provider
        self._query_planner = query_planner or QueryPlanner(
            provider, llm_call_config=llm_call_config
        )
        self._retrieval_router = retrieval_router or RetrievalRouter()
        self._public_retriever = public_retriever or PublicOntologyRetriever()
        self._local_retriever = local_retriever or LocalSemanticRetriever()
        self._candidate_normalizer = candidate_normalizer or CandidateNormalizer()
        self._candidate_merger = candidate_merger or CandidateMerger()
        self._llm_reranker = llm_reranker or LLMReranker(provider, llm_call_config=llm_call_config)
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
        max_results_per_query: int = 15,
        max_candidates: int | None = None,
        max_alternatives: int = 5,
        *,
        source_description: str | None = None,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        """
        Map one source term through the planned pipeline.

        Args mirror the planned API.  retrieval_mode accepts the three
        user-facing values only: public, local, disabled.

        strict_target_ontology: When True, require canonical native-ontology
            membership only for candidate eligibility (merge, rerank, and
            result building); disables the EFO imported-term provenance
            exception. Has no effect on retrieval routing or query planning.

        Raises:
            ValueError: invalid retrieval_mode string.
            PlannedPipelineError: component failure during orchestration.
        """
        mode = self._resolve_mode(retrieval_mode)
        timings: dict[str, float] = {}
        usage: dict[str, Any] = {}
        # Declared before the try block (not inside it) so it is always bound
        # for the except clause below, even when a failure occurs before
        # retrieval runs (e.g. QueryPlanner failure).
        timed_route_calls: list[dict[str, Any]] = []

        try:
            query_plan = _time_stage(
                timings,
                "query_planning_ms",
                lambda: self._plan(
                    source_term=source_term,
                    source_label=source_label,
                    source_description=source_description,
                    source_type=source_type,
                    clinical_area=clinical_area,
                    target_ontology=target_ontology,
                    allowed_target_ontologies=allowed_target_ontologies,
                    retrieval_mode=mode,
                    timings=timings,
                    usage=usage,
                ),
            )
            timings.setdefault("query_planning_provider_ms", 0.0)
            route_plan = _time_stage(
                timings,
                "routing_ms",
                lambda: self._route(query_plan),
            )

            if mode == RetrievalMode.DISABLED:
                return self._map_disabled(
                    query_plan,
                    source_type=source_type,
                    max_alternatives=max_alternatives,
                )

            raw_candidates = _time_stage(
                timings,
                "retrieval_ms",
                lambda: self._retrieve(
                    query_plan=query_plan,
                    route_plan=route_plan,
                    max_results_per_query=max_results_per_query,
                    route_calls=timed_route_calls,
                ),
            )
            normalized_candidates, normalization_errors = _time_stage(
                timings,
                "candidate_normalization_ms",
                lambda: self._normalize_raw_candidates(
                    raw_candidates,
                    query_plan=query_plan,
                ),
            )
            merged_candidates = _time_stage(
                timings,
                "candidate_merging_ms",
                lambda: self._merge_candidates(
                    normalized_candidates,
                    target_ontology_constraint=query_plan.target_ontology_constraint,
                    allowed_target_ontologies=query_plan.allowed_target_ontologies,
                    max_candidates=max_candidates,
                    strict_target_ontology=strict_target_ontology,
                ),
            )
            rerank_decision = _time_stage(
                timings,
                "llm_reranking_ms",
                lambda: self._rerank(
                    query_plan,
                    merged_candidates,
                    timings=timings,
                    usage=usage,
                    strict_target_ontology=strict_target_ontology,
                ),
            )
            retrieval_trace = _time_stage(
                timings,
                "trace_building_ms",
                lambda: self._build_trace(
                    query_plan=query_plan,
                    route_plan=route_plan,
                    raw_candidate_count=len(raw_candidates),
                    merged_candidates=merged_candidates,
                    rerank_decision=rerank_decision,
                    normalization_errors=normalization_errors,
                    route_calls=timed_route_calls or None,
                ),
            )
            result = _time_stage(
                timings,
                "result_building_ms",
                lambda: self._build_result(
                    query_plan=query_plan,
                    rerank_decision=rerank_decision,
                    candidates=merged_candidates,
                    retrieval_trace=retrieval_trace,
                    source_type=source_type,
                    max_alternatives=max_alternatives,
                    strict_target_ontology=strict_target_ontology,
                ),
            )
        except PlannedPipelineError as exc:
            # Whatever timing/usage/retrieval telemetry had already been
            # recorded before the failing stage raised is still real -- those
            # calls were billed even though the overall mapping failed. Attach
            # it to the exception so a caller catching PlannedPipelineError
            # (e.g. the benchmark runner) does not lose it entirely.
            exc.partial_timings = dict(timings)
            exc.partial_usage = dict(usage)
            exc.partial_retrieval_diagnostics = _summarize_retrieval_diagnostics(timed_route_calls)
            raise

        _attach_pipeline_timings(result, timings)
        _attach_pipeline_usage(result, usage)
        _attach_retrieval_diagnostics(result, timed_route_calls)
        return result

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
        source_description: str | None,
        source_type: str | None,
        clinical_area: str | None,
        target_ontology: str | None,
        allowed_target_ontologies: list[str] | None,
        retrieval_mode: RetrievalMode,
        timings: dict[str, float],
        usage: dict[str, Any],
    ) -> QueryPlan:
        try:
            kwargs: dict[str, Any] = {
                "source_term": source_term,
                "source_label": source_label,
                "source_description": source_description,
                "source_type": source_type,
                "clinical_area": clinical_area,
                "target_ontology": target_ontology,
                "allowed_target_ontologies": allowed_target_ontologies,
                "retrieval_mode": retrieval_mode,
            }
            if _accepts_keyword(self._query_planner.plan, "timing_sink"):
                kwargs["timing_sink"] = timings
            if _accepts_keyword(self._query_planner.plan, "usage_sink"):
                kwargs["usage_sink"] = usage
            return self._query_planner.plan(**kwargs)
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
        max_alternatives: int,
    ) -> MappingResult:
        try:
            return self._disabled_mapping_runner.map(
                query_plan,
                source_type=source_type,
                max_alternatives=max_alternatives,
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
        route_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            kwargs: dict[str, Any] = {
                "route_plan": route_plan,
                "max_results_per_query": max_results_per_query,
            }
            if query_plan.retrieval_mode == RetrievalMode.PUBLIC:
                if _accepts_keyword(self._public_retriever.retrieve, "route_calls"):
                    kwargs["route_calls"] = route_calls
                return self._public_retriever.retrieve(
                    query_plan,
                    **kwargs,
                )
            if query_plan.retrieval_mode == RetrievalMode.LOCAL:
                if _accepts_keyword(self._local_retriever.retrieve, "route_calls"):
                    kwargs["route_calls"] = route_calls
                return self._local_retriever.retrieve(
                    query_plan,
                    **kwargs,
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
        strict_target_ontology: bool = False,
    ) -> list[NormalizedCandidate]:
        try:
            return self._candidate_merger.merge(
                candidates,
                target_ontology_constraint=target_ontology_constraint,
                allowed_target_ontologies=allowed_target_ontologies,
                max_candidates=max_candidates,
                strict_target_ontology=strict_target_ontology,
            )
        except Exception as exc:
            raise PlannedPipelineError("CandidateMerger failed during planned mapping.") from exc

    def _rerank(
        self,
        query_plan: QueryPlan,
        candidates: list[NormalizedCandidate],
        *,
        timings: dict[str, float],
        usage: dict[str, Any],
        strict_target_ontology: bool = False,
    ) -> RerankDecision:
        try:
            kwargs: dict[str, Any] = {}
            if _accepts_keyword(self._llm_reranker.rerank, "timing_sink"):
                kwargs["timing_sink"] = timings
            if _accepts_keyword(self._llm_reranker.rerank, "usage_sink"):
                kwargs["usage_sink"] = usage
            decision = self._llm_reranker.rerank(
                query_plan,
                candidates,
                strict_target_ontology=strict_target_ontology,
                **kwargs,
            )
            timings.setdefault("llm_reranker_provider_ms", 0.0)
            return decision
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
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        try:
            return self._mapping_result_builder.build(
                query_plan=query_plan,
                rerank_decision=rerank_decision,
                candidates=candidates,
                retrieval_trace=retrieval_trace,
                source_type=source_type,
                max_alternatives=max_alternatives,
                strict_target_ontology=strict_target_ontology,
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
        route_calls: list[dict[str, Any]] | None = None,
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
            route_calls=list(route_calls) if route_calls is not None else list(route_plan.route_calls),
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


def _time_stage(
    timings: dict[str, float],
    key: str,
    func: Callable[[], _T],
) -> _T:
    start = time.monotonic()
    try:
        return func()
    finally:
        timings[key] = (time.monotonic() - start) * 1000


def _accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
    parameters = signature(func).parameters.values()
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


def _attach_pipeline_timings(result: MappingResult, timings: dict[str, float]) -> None:
    metadata = result.metadata
    if metadata is None or metadata.rag_debug is None:
        return
    rag_debug = metadata.rag_debug.model_copy(update={"pipeline_timings": dict(timings)})
    result.metadata = metadata.model_copy(update={"rag_debug": rag_debug})


def _attach_pipeline_usage(result: MappingResult, usage: dict[str, Any]) -> None:
    """Attach per-stage LLM token usage collected via usage_sink, when any was recorded.

    Mirrors _attach_pipeline_timings: usage_sink is only populated by
    QueryPlanner/LLMReranker implementations that opt into the usage_sink
    keyword (see _accepts_keyword), so an empty dict here means no usage
    telemetry was available — nothing is fabricated.
    """
    if not usage:
        return
    metadata = result.metadata
    if metadata is None or metadata.rag_debug is None:
        return
    rag_debug = metadata.rag_debug.model_copy(update={"pipeline_usage": dict(usage)})
    result.metadata = metadata.model_copy(update={"rag_debug": rag_debug})


_MAX_RETRIEVAL_ERROR_DETAILS = 10


def _summarize_retrieval_diagnostics(route_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-route-call retry/error diagnostics already captured on
    each route_calls entry (see PublicOntologyRetriever._call_route_timed)
    into the small set of counters and bounded error details useful for
    benchmark output, rather than dumping every route call.

    A route call the retriever never annotated with retrieval_attempts
    (e.g. a local-mode SapBERT call, which carries no HTTP retry telemetry)
    is not counted here -- request_count reflects public-API calls only.

    A route call whose final_error_type is unset succeeded, either on the
    first attempt (attempts == 1) or after one or more retries
    ("recovered"). One whose final_error_type is set exhausted its retries
    (or hit a non-retryable status immediately) and is a final error.
    """
    request_count = 0
    retry_count = 0
    recovered_count = 0
    final_error_count = 0
    error_sources: list[str] = []
    error_types: list[str] = []

    for call in route_calls:
        attempts = call.get("retrieval_attempts")
        if attempts is None:
            continue
        request_count += 1
        retry_count += max(0, attempts - 1)
        final_error_type = call.get("retrieval_final_error_type")
        if final_error_type:
            final_error_count += 1
            route_name = call.get("route_name") or call.get("route") or "unknown"
            ontologies = call.get("candidate_ontologies") or []
            target = ontologies[0] if ontologies else None
            error_sources.append(f"{route_name}:{target}" if target else str(route_name))
            error_types.append(str(final_error_type))
        elif attempts > 1:
            recovered_count += 1

    return {
        "retrieval_request_count": request_count,
        "retrieval_retry_count": retry_count,
        "retrieval_recovered_error_count": recovered_count,
        "retrieval_final_error_count": final_error_count,
        "retrieval_error_sources": error_sources[:_MAX_RETRIEVAL_ERROR_DETAILS],
        "retrieval_error_types": error_types[:_MAX_RETRIEVAL_ERROR_DETAILS],
    }


def _attach_retrieval_diagnostics(result: MappingResult, route_calls: list[dict[str, Any]]) -> None:
    """Attach aggregate retrieval retry/error telemetry, when any route call
    carried it (see _summarize_retrieval_diagnostics). Diagnostic metadata
    only -- never affects scoring, mapped_status, or candidate content.
    """
    if not route_calls:
        return
    summary = _summarize_retrieval_diagnostics(route_calls)
    if summary["retrieval_request_count"] == 0:
        return
    metadata = result.metadata
    if metadata is None or metadata.rag_debug is None:
        return
    rag_debug = metadata.rag_debug.model_copy(update={"retrieval_diagnostics": summary})
    result.metadata = metadata.model_copy(update={"rag_debug": rag_debug})
