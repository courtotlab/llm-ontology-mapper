"""
Scenario 1 (OLS-EFO) execution: SapBERT preflight, mapper construction, and
the per-query execution loop against the real local-mode planned pipeline.

Reuses llm_ontology_mapper.benchmarking.runner wherever its helpers are
retrieval-mode-agnostic (provider construction, LLM preflight, pipeline
timing/usage extraction, exception-chain formatting) instead of
reimplementing them -- this module only adds what is genuinely different
about Scenario 1: local-mode/non-strict mapper wiring, SapBERT health
verification, and the OLS-EFO row/telemetry schema (Part 8).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests  # type: ignore[import-untyped]

from llm_ontology_mapper.benchmarking.model_registry import BenchmarkModelConfig
from llm_ontology_mapper.benchmarking.pricing import ModelPricing, calculate_cost_usd
from llm_ontology_mapper.benchmarking.runner import (  # noqa: F401 -- re-exported for callers
    PreflightError,
    _extract_pipeline_timings,
    _extract_pipeline_usage,
    _extract_retrieval_diagnostics,
    _ms_to_s,
    _retrieval_diagnostics_row_fields,
    _sum_optional,
    _sum_optional_float,
    build_provider,
    describe_temperature,
    format_exception_chain,
    run_preflight,
    safe_error_message,
)
from llm_ontology_mapper.benchmarking.scenario1_dataset import CanonicalQuery
from llm_ontology_mapper.candidate_merger import CandidateMergeError
from llm_ontology_mapper.candidate_normalizer import CandidateNormalizationError
from llm_ontology_mapper.disabled_mapping import DisabledMappingError
from llm_ontology_mapper.llm_reranker import LLMRerankerError
from llm_ontology_mapper.local_retriever import LocalRetrievalError, LocalSemanticRetriever
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilderError
from llm_ontology_mapper.planned_pipeline import PlannedPipeline, PlannedPipelineError
from llm_ontology_mapper.providers import LLMCallConfig, OpenAIProvider
from llm_ontology_mapper.public_retriever import PublicRetrievalError
from llm_ontology_mapper.query_planner import QueryPlanningError

TARGET_ONTOLOGY = "EFO"
RETRIEVAL_MODE = "local"
STRICT_TARGET_ONTOLOGY = False  # Scenario 1 non-strict default -- see module docstring
_UNMAPPED_CODE = "UNKNOWN:UNMAPPED"
_UNMAPPED_ONTOLOGY = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Structural error-stage classification (reliability-audit follow-up)
# ─────────────────────────────────────────────────────────────────────────────

ERROR_STAGE_LOCAL_RETRIEVAL = "local_retrieval"
ERROR_STAGE_QUERY_PLANNER = "query_planner"
ERROR_STAGE_RERANKER = "reranker"
ERROR_STAGE_PIPELINE = "pipeline"
ERROR_STAGE_UNKNOWN = "unknown"

# Typed pipeline exceptions checked by isinstance() against the exception's
# actual __cause__/__context__ chain -- never by string-matching the
# rendered error message (see PlannedPipeline for how each stage wraps its
# own failures in PlannedPipelineError while preserving the original cause).
_TYPED_STAGE_ERRORS: tuple[tuple[type[BaseException], str], ...] = (
    (QueryPlanningError, ERROR_STAGE_QUERY_PLANNER),
    (LLMRerankerError, ERROR_STAGE_RERANKER),
    (CandidateMergeError, ERROR_STAGE_PIPELINE),
    (CandidateNormalizationError, ERROR_STAGE_PIPELINE),
    (MappingResultBuilderError, ERROR_STAGE_PIPELINE),
    (DisabledMappingError, ERROR_STAGE_PIPELINE),
    (PublicRetrievalError, ERROR_STAGE_PIPELINE),
)

_MAX_ERROR_STAGE_CHAIN_DEPTH = 8


def _exception_chain(
    exc: BaseException, *, max_depth: int = _MAX_ERROR_STAGE_CHAIN_DEPTH
) -> list[BaseException]:
    """Return `exc` and its wrapped cause(s), outer to inner (see
    format_exception_chain in runner.py for the same traversal rule, here
    returning exception objects instead of rendered text so callers can
    isinstance()-check the real types)."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < max_depth:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        depth += 1
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
    return chain


def classify_error_stage(exc: BaseException) -> str:
    """Classify which pipeline stage an execute_query() failure originated
    from, using the exception object's actual type in the cause chain.

    local_retrieval (LocalRetrievalError) is checked first and takes
    priority -- it is the specific condition the consecutive-failure guard
    in run_scenario1_ols_efo.py watches for, and must never be confused
    with an unrelated LLM/pipeline failure.

    When no typed pipeline exception is found (e.g. a raw provider/SDK
    exception surfaced directly from QueryPlanner's or LLMReranker's own
    provider.complete() call, which is not wrapped in a typed error -- see
    query_planner.py/_complete_plan_timed), falls back to the fixed
    stage-marker text PlannedPipeline itself writes at each call site
    (planned_pipeline.py _plan/_rerank/etc.) -- a literal this codebase
    controls, not a parsed upstream/opaque message.
    """
    chain = _exception_chain(exc)

    for candidate in chain:
        if isinstance(candidate, LocalRetrievalError):
            return ERROR_STAGE_LOCAL_RETRIEVAL

    for candidate in chain:
        for exc_type, stage in _TYPED_STAGE_ERRORS:
            if isinstance(candidate, exc_type):
                return stage

    for candidate in chain:
        if isinstance(candidate, PlannedPipelineError):
            message = str(candidate)
            if message.startswith("QueryPlanner failed"):
                return ERROR_STAGE_QUERY_PLANNER
            if message.startswith("LLMReranker failed"):
                return ERROR_STAGE_RERANKER
            return ERROR_STAGE_PIPELINE

    return ERROR_STAGE_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Part 5 -- SapBERT preflight
# ─────────────────────────────────────────────────────────────────────────────


class SapBertHealthError(RuntimeError):
    """Raised when the local SapBERT service is unreachable, unhealthy, or
    does not have EFO loaded/available."""


@dataclass(frozen=True)
class SapBertHealth:
    raw_response: dict[str, Any]
    status: str
    model: str | None
    loaded_indexes: list[str]
    available_indexes: list[str]
    lazy_load: bool | None


def check_sapbert_health(sapbert_url: str, *, timeout: float = 10.0) -> SapBertHealth:
    """GET {sapbert_url}/health and require status=ok with EFO present in
    available_indexes or loaded_indexes (Part 5). Raises SapBertHealthError
    on any failure -- never silently proceeds."""
    url = sapbert_url.rstrip("/") + "/health"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        raise SapBertHealthError(f"SapBERT health check failed for {url!r}: {exc}") from exc
    except ValueError as exc:
        raise SapBertHealthError(f"SapBERT health response is not valid JSON ({url!r}): {exc}") from exc

    if not isinstance(data, dict):
        raise SapBertHealthError(f"SapBERT health response is not a JSON object: {data!r}")

    status = str(data.get("status", ""))
    loaded = [str(x) for x in (data.get("loaded_indexes") or [])]
    available = [str(x) for x in (data.get("available_indexes") or [])]

    if status != "ok":
        raise SapBertHealthError(f"SapBERT health status != 'ok' (got {status!r}): {data!r}")
    if "EFO" not in loaded and "EFO" not in available:
        raise SapBertHealthError(
            f"EFO is not present in loaded_indexes or available_indexes: {data!r}"
        )

    return SapBertHealth(
        raw_response=data,
        status=status,
        model=data.get("model"),
        loaded_indexes=loaded,
        available_indexes=available,
        lazy_load=data.get("lazy_load"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario1RunConfig:
    """Locked Scenario 1 configuration (Part 6/19). target_ontology,
    retrieval_mode, and strict_target_ontology are intentionally NOT
    parameters -- they are fixed module constants so Scenario 1 can never
    silently drift from local/EFO/non-strict."""

    model_config: BenchmarkModelConfig
    sapbert_url: str
    temperature: float | None = None
    seed: int = 42
    max_alternatives: int = 4
    max_results_per_query: int = 10
    max_candidates: int | None = 10

    @property
    def temperature_mode(self) -> str:
        return "provider_default" if self.temperature is None else "explicit"

    def to_llm_call_config(self) -> LLMCallConfig:
        return LLMCallConfig(
            temperature=self.temperature,
            seed=self.seed,
            reasoning_effort=self.model_config.reasoning_effort,
            force_temperature=True,
            strict=True,
        )


def build_mapper(*, provider: OpenAIProvider, run_config: Scenario1RunConfig) -> OntologyMapper:
    """Build the real local-mode, non-strict, EFO-target mapper (Part 6)."""
    pipeline = PlannedPipeline(
        provider=provider,
        local_retriever=LocalSemanticRetriever(sapbert_url=run_config.sapbert_url),
        llm_call_config=run_config.to_llm_call_config(),
    )
    return OntologyMapper(
        llm_provider=provider,
        ontologies=[TARGET_ONTOLOGY],
        use_planned_pipeline=True,
        retrieval_mode=RETRIEVAL_MODE,
        planned_pipeline=pipeline,
        rag_top_k=run_config.max_results_per_query,
        max_candidates=run_config.max_candidates,
        max_alternatives=run_config.max_alternatives,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 8 -- per-query row result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankSlot:
    code: str | None = None
    label: str | None = None
    ontology: str | None = None


@dataclass
class Scenario1RowResult:
    query_id: int
    query: str
    gold_codes: list[str]
    gold_labels: list[str | None]
    gold_count: int

    status: str  # "mapped" | "unmapped" | "error"

    mapped_code: str | None = None
    mapped_term: str | None = None
    mapped_ontology: str | None = None
    confidence: float | None = None

    ranks: list[RankSlot] = field(default_factory=lambda: [RankSlot() for _ in range(5)])

    end_to_end_seconds: float = 0.0
    query_planner_seconds: float | None = None
    retrieval_seconds: float | None = None
    reranker_seconds: float | None = None
    llm_seconds: float | None = None

    total_input_tokens: int | None = None
    total_cached_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    api_cost_usd: float | None = None

    retrieval_request_count: int | None = None
    retrieval_retry_count: int | None = None
    retrieval_recovered_error_count: int | None = None
    retrieval_final_error_count: int | None = None
    retrieval_error_sources: str | None = None
    retrieval_error_types: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    error_stage: str | None = None

    @property
    def rank_codes(self) -> tuple[str | None, ...]:
        return tuple(slot.code for slot in self.ranks)

    @property
    def rank_ontologies(self) -> tuple[str | None, ...]:
        return tuple(slot.ontology for slot in self.ranks)


def _error_row(
    *,
    cq: CanonicalQuery,
    end_to_end_seconds: float,
    exc: BaseException,
    pricing: ModelPricing | None,
) -> Scenario1RowResult:
    partial_timings: dict[str, float] = getattr(exc, "partial_timings", None) or {}
    partial_usage: dict[str, Any] = getattr(exc, "partial_usage", None) or {}
    partial_retrieval: dict[str, Any] = getattr(exc, "partial_retrieval_diagnostics", None) or {}
    planner_usage = partial_usage.get("query_planner") or {}
    reranker_usage = partial_usage.get("llm_reranker") or {}
    retrieval_fields = _retrieval_diagnostics_row_fields(partial_retrieval)

    total_input = _sum_optional(planner_usage.get("input_tokens"), reranker_usage.get("input_tokens"))
    total_cached = _sum_optional(
        planner_usage.get("cached_input_tokens"), reranker_usage.get("cached_input_tokens")
    )
    total_output = _sum_optional(planner_usage.get("output_tokens"), reranker_usage.get("output_tokens"))
    total_reasoning = _sum_optional(
        planner_usage.get("reasoning_tokens"), reranker_usage.get("reasoning_tokens")
    )
    api_cost_usd = (
        calculate_cost_usd(
            input_tokens=total_input,
            cached_input_tokens=total_cached,
            output_tokens=total_output,
            pricing=pricing,
        )
        if pricing is not None
        else None
    )

    return Scenario1RowResult(
        query_id=cq.query_id,
        query=cq.source_query,
        gold_codes=list(cq.gold_codes),
        gold_labels=list(cq.gold_labels),
        gold_count=cq.gold_count,
        status="error",
        end_to_end_seconds=end_to_end_seconds,
        query_planner_seconds=_ms_to_s(partial_timings.get("query_planning_ms")),
        retrieval_seconds=_ms_to_s(partial_timings.get("retrieval_ms")),
        reranker_seconds=_ms_to_s(partial_timings.get("llm_reranking_ms")),
        llm_seconds=_ms_to_s(
            _sum_optional_float(
                partial_timings.get("query_planning_provider_ms"),
                partial_timings.get("llm_reranker_provider_ms"),
            )
        ),
        total_input_tokens=total_input,
        total_cached_input_tokens=total_cached,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        api_cost_usd=api_cost_usd,
        retrieval_request_count=retrieval_fields["retrieval_request_count"],
        retrieval_retry_count=retrieval_fields["retrieval_retry_count"],
        retrieval_recovered_error_count=retrieval_fields["retrieval_recovered_error_count"],
        retrieval_final_error_count=retrieval_fields["retrieval_final_error_count"],
        retrieval_error_sources=retrieval_fields["retrieval_error_sources"],
        retrieval_error_types=retrieval_fields["retrieval_error_types"],
        error_type=type(exc).__name__,
        error_message=format_exception_chain(exc),
        error_stage=classify_error_stage(exc),
    )


def execute_query(
    *,
    mapper: OntologyMapper,
    cq: CanonicalQuery,
    pricing: ModelPricing | None,
) -> Scenario1RowResult:
    """Map exactly one canonical query through the real local/non-strict/EFO
    pipeline (Part 6/9). Ranks are taken verbatim from the mapper's returned
    order -- rank 1 = selected result, ranks 2-5 = alternatives[0..3], never
    reranked/sorted/filtered here (Part 9)."""
    start = time.perf_counter()
    try:
        result = mapper.map_term(
            source_term=cq.source_query,
            strict_target_ontology=STRICT_TARGET_ONTOLOGY,
        )
    except Exception as exc:  # noqa: BLE001 -- execution error, not a mapping decision (Part 17)
        return _error_row(cq=cq, end_to_end_seconds=time.perf_counter() - start, exc=exc, pricing=pricing)
    end_to_end_seconds = time.perf_counter() - start

    is_mapped = not (
        result.target_code.upper() == _UNMAPPED_CODE and result.ontology.upper() == _UNMAPPED_ONTOLOGY
    )

    ranks = [
        RankSlot(code=result.target_code, label=result.target_term, ontology=result.ontology)
        if is_mapped
        else RankSlot()
    ]
    for alt in result.alternatives[:4]:
        ranks.append(RankSlot(code=alt.code, label=alt.term, ontology=alt.ontology))
    while len(ranks) < 5:
        ranks.append(RankSlot())

    pipeline_timings = _extract_pipeline_timings(result)
    pipeline_usage = _extract_pipeline_usage(result)
    retrieval_diagnostics = _extract_retrieval_diagnostics(result)
    retrieval_fields = _retrieval_diagnostics_row_fields(retrieval_diagnostics)
    planner_usage = pipeline_usage.get("query_planner") or {}
    reranker_usage = pipeline_usage.get("llm_reranker") or {}

    total_input = _sum_optional(planner_usage.get("input_tokens"), reranker_usage.get("input_tokens"))
    total_cached = _sum_optional(
        planner_usage.get("cached_input_tokens"), reranker_usage.get("cached_input_tokens")
    )
    total_output = _sum_optional(planner_usage.get("output_tokens"), reranker_usage.get("output_tokens"))
    total_reasoning = _sum_optional(
        planner_usage.get("reasoning_tokens"), reranker_usage.get("reasoning_tokens")
    )
    api_cost_usd = (
        calculate_cost_usd(
            input_tokens=total_input,
            cached_input_tokens=total_cached,
            output_tokens=total_output,
            pricing=pricing,
        )
        if pricing is not None
        else None
    )

    return Scenario1RowResult(
        query_id=cq.query_id,
        query=cq.source_query,
        gold_codes=list(cq.gold_codes),
        gold_labels=list(cq.gold_labels),
        gold_count=cq.gold_count,
        status="mapped" if is_mapped else "unmapped",
        mapped_code=result.target_code if is_mapped else None,
        mapped_term=result.target_term if is_mapped else None,
        mapped_ontology=result.ontology if is_mapped else None,
        confidence=result.confidence,
        ranks=ranks,
        end_to_end_seconds=end_to_end_seconds,
        query_planner_seconds=_ms_to_s(pipeline_timings.get("query_planning_ms")),
        retrieval_seconds=_ms_to_s(pipeline_timings.get("retrieval_ms")),
        reranker_seconds=_ms_to_s(pipeline_timings.get("llm_reranking_ms")),
        llm_seconds=_ms_to_s(
            _sum_optional_float(
                pipeline_timings.get("query_planning_provider_ms"),
                pipeline_timings.get("llm_reranker_provider_ms"),
            )
        ),
        total_input_tokens=total_input,
        total_cached_input_tokens=total_cached,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        api_cost_usd=api_cost_usd,
        retrieval_request_count=retrieval_fields["retrieval_request_count"],
        retrieval_retry_count=retrieval_fields["retrieval_retry_count"],
        retrieval_recovered_error_count=retrieval_fields["retrieval_recovered_error_count"],
        retrieval_final_error_count=retrieval_fields["retrieval_final_error_count"],
        retrieval_error_sources=retrieval_fields["retrieval_error_sources"],
        retrieval_error_types=retrieval_fields["retrieval_error_types"],
    )


def iter_predictions(
    *,
    mapper: OntologyMapper,
    canonical_queries: Sequence[CanonicalQuery],
    pricing: ModelPricing | None,
    skip_query_ids: Iterable[int] = (),
) -> Iterator[Scenario1RowResult]:
    """Yield one Scenario1RowResult per canonical query, in dataset order,
    skipping any query_id already present (resume support, Part 18)."""
    skip = set(skip_query_ids)
    for cq in canonical_queries:
        if cq.query_id in skip:
            continue
        yield execute_query(mapper=mapper, cq=cq, pricing=pricing)


__all__ = [
    "ERROR_STAGE_LOCAL_RETRIEVAL",
    "ERROR_STAGE_PIPELINE",
    "ERROR_STAGE_QUERY_PLANNER",
    "ERROR_STAGE_RERANKER",
    "ERROR_STAGE_UNKNOWN",
    "RETRIEVAL_MODE",
    "STRICT_TARGET_ONTOLOGY",
    "TARGET_ONTOLOGY",
    "PlannedPipelineError",
    "PreflightError",
    "RankSlot",
    "SapBertHealth",
    "SapBertHealthError",
    "Scenario1RowResult",
    "Scenario1RunConfig",
    "build_mapper",
    "build_provider",
    "check_sapbert_health",
    "classify_error_stage",
    "describe_temperature",
    "execute_query",
    "iter_predictions",
    "run_preflight",
    "safe_error_message",
]
