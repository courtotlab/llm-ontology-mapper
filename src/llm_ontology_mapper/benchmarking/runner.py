"""
Benchmark execution: preflight validation, mapper construction, and the
per-row / per-run execution loop.

This module exercises the real OntologyMapper + PlannedPipeline (planned
public pipeline) exactly as tests/live/planned_public_smoke.py does -- it
never reimplements retrieval, reranking, or scoring-adjacent mapping logic.

One OntologyMapper is built per distinct target_ontology in the dataset
because OntologyMapper's hard target-ontology constraint (`ontologies=[...]`)
is set at construction time, not per map_term() call. All per-ontology
mappers share one PlannedPipeline (and therefore one provider, retriever,
and reranker instance) so the LLM/HTTP configuration is identical across the
whole run; only the hard target-ontology constraint differs between them.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.model_registry import BenchmarkModelConfig
from llm_ontology_mapper.benchmarking.pricing import ModelPricing, calculate_cost_usd
from llm_ontology_mapper.benchmarking.scoring import RowScore, aggregate_scores, score_row
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.ontology_identity import canonical_ontology, normalize_code_for_ontology
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import (
    ChatMessage,
    LLMCallConfig,
    OpenAIProvider,
    resolve_llm_call_params,
)
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.search_tools import SearchTools

_UNMAPPED_CODE = "UNKNOWN:UNMAPPED"
_UNMAPPED_ONTOLOGY = "UNKNOWN"
_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,})")


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Locked benchmark-wide configuration applied identically to both runs.

    temperature=None means the request omits `temperature` entirely and the
    provider's own default is used -- this is the locked default for every
    reviewed benchmark model (see model_registry.ALLOWED_MODELS), since
    GPT-5-family reasoning models reject a non-default temperature. Passing
    an explicit numeric temperature is still supported as an override; an
    unsupported explicit value still fails strict preflight (see
    run_preflight) rather than being silently dropped.
    """

    model_config: BenchmarkModelConfig
    temperature: float | None
    seed: int
    retrieval_mode: str = "public"
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


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────


class PreflightError(RuntimeError):
    """Raised when the requested benchmark parameters are rejected by the live API."""


def safe_error_message(exc: BaseException, *, max_len: int = 500) -> str:
    """Truncate and redact an exception message before it is logged or persisted."""
    return _SECRET_PATTERN.sub("***REDACTED***", str(exc))[:max_len]


_MAX_EXCEPTION_CHAIN_DEPTH = 5


def format_exception_chain(
    exc: BaseException,
    *,
    max_len: int = 500,
    max_depth: int = _MAX_EXCEPTION_CHAIN_DEPTH,
) -> str:
    """Render `exc` and its wrapped cause(s) as concise, sanitized, CSV-friendly text.

    A wrapper like PlannedPipelineError preserves its underlying cause via
    `raise ... from exc` (see PlannedPipeline._plan), but a plain `str(exc)`
    on the outermost exception only shows the generic wrapper message --
    the actually-useful detail (e.g. the JSONDecodeError under a
    QueryPlanningError) is left unread on `exc.__cause__`. This walks that
    chain so it ends up in the benchmark CSV instead of being discarded.

    Follows the same chaining rule Python's own traceback formatter uses:
    prefer `__cause__` (explicit `raise ... from exc`); fall back to
    `__context__` only when no explicit cause was set and the exception did
    not suppress it. Stops on a cycle or once `max_depth` exceptions have
    been rendered. Each exception's message is independently truncated and
    redacted via safe_error_message() -- never a full stack trace.
    """
    lines: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < max_depth:
        if id(current) in seen:
            break
        seen.add(id(current))
        prefix = "" if depth == 0 else "Caused by: "
        message = safe_error_message(current, max_len=max_len)
        lines.append(f"{prefix}{type(current).__name__}: {message}")
        depth += 1
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
    return "\n".join(lines)


def describe_temperature(temperature: float | None) -> str:
    """Human-readable temperature for logs/preflight/config output.

    Renders omitted temperature as 'provider_default' rather than a bare
    None or a fabricated numeric value, so output never falsely implies a
    specific temperature (e.g. 0.0) was requested.
    """
    return "provider_default" if temperature is None else str(temperature)


def run_preflight(provider: OpenAIProvider, llm_call_config: LLMCallConfig) -> None:
    """
    Cheaply verify the requested benchmark parameters (temperature, seed,
    reasoning_effort) are accepted by the live model/API before launching the
    full dataset run. Uses the same call-construction path
    (resolve_llm_call_params + strict mode) as the real planner/reranker
    calls, so a rejection here reflects what would happen mid-run.
    """
    temperature, extra_kwargs = resolve_llm_call_params(
        llm_call_config,
        default_temperature=0.1,
        default_extra_kwargs={},
    )
    messages = [
        ChatMessage(role="system", content="Reply with exactly one word."),
        ChatMessage(role="user", content="Reply: ok"),
    ]
    try:
        provider.complete(messages, temperature=temperature, max_tokens=16, **extra_kwargs)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any rejection must fail fast
        raise PreflightError(
            "Preflight check failed: model="
            f"{provider.model!r} rejected the requested benchmark configuration "
            f"(temperature={describe_temperature(llm_call_config.temperature)}, "
            f"seed={llm_call_config.seed}, "
            f"reasoning_effort={llm_call_config.reasoning_effort!r}). Refusing to "
            "launch the benchmark rather than silently dropping the parameter. "
            f"Underlying error: {type(exc).__name__}: {safe_error_message(exc)}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Mapper / pipeline construction
# ─────────────────────────────────────────────────────────────────────────────


def build_provider(model: str) -> OpenAIProvider:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the model benchmark.")
    return OpenAIProvider(model=model, api_key=api_key)


def build_pipeline_and_mappers(
    *,
    provider: OpenAIProvider,
    run_config: BenchmarkRunConfig,
    target_ontologies: Sequence[str],
) -> tuple[PlannedPipeline, dict[str, OntologyMapper]]:
    """Build one shared PlannedPipeline and one OntologyMapper per distinct
    target ontology (see module docstring for why one mapper per ontology).
    """
    search_tools = SearchTools(
        loinc_username=os.environ.get("LOINC_USERNAME"),
        loinc_password=os.environ.get("LOINC_PASSWORD"),
    )
    pipeline = PlannedPipeline(
        provider=provider,
        public_retriever=PublicOntologyRetriever(search_tools=search_tools),
        llm_call_config=run_config.to_llm_call_config(),
    )
    mappers: dict[str, OntologyMapper] = {}
    for ontology in target_ontologies:
        mappers[ontology] = OntologyMapper(
            llm_provider=provider,
            ontologies=[ontology],
            use_planned_pipeline=True,
            retrieval_mode=run_config.retrieval_mode,
            planned_pipeline=pipeline,
            rag_top_k=run_config.max_results_per_query,
            max_candidates=run_config.max_candidates,
            max_alternatives=run_config.max_alternatives,
        )
    return pipeline, mappers


# ─────────────────────────────────────────────────────────────────────────────
# Row / run result records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlternativeSlot:
    code: str | None
    term: str | None
    ontology: str | None
    confidence: float | None


@dataclass
class RowResult:
    input_row: int
    source_variable: str
    source_label: str | None
    source_description: str | None
    target_ontology: str

    gold_code_raw: str
    gold_codes_normalized: list[str]
    gold_target_term: str | None

    mapped_status: str  # "mapped" | "unmapped" | "error"
    mapped_code: str | None
    mapped_code_normalized: str | None
    mapped_term: str | None
    mapped_ontology: str | None
    confidence: float | None
    logic_type: str | None

    alternatives: list[AlternativeSlot] = field(default_factory=list)

    gold_rank: int | None = None
    top1_correct: bool = False
    top5_hit: bool = False
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    end_to_end_seconds: float = 0.0
    query_planner_seconds: float | None = None
    retrieval_seconds: float | None = None
    reranker_seconds: float | None = None
    llm_seconds: float | None = None

    planner_input_tokens: int | None = None
    planner_cached_input_tokens: int | None = None
    planner_output_tokens: int | None = None
    planner_reasoning_tokens: int | None = None

    reranker_input_tokens: int | None = None
    reranker_cached_input_tokens: int | None = None
    reranker_output_tokens: int | None = None
    reranker_reasoning_tokens: int | None = None

    total_input_tokens: int | None = None
    total_cached_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    api_cost_usd: float | None = None

    model: str = ""
    requested_reasoning_effort: str = "N/A"
    temperature: float | None = None
    temperature_mode: str = "provider_default"
    seed: int = 0
    retrieval_mode: str = "public"
    max_alternatives: int = 4
    run_number: int = 1
    system_fingerprint: str | None = None

    # Diagnostic-only retrieval telemetry (see planned_pipeline._summarize_
    # retrieval_diagnostics): never influences tp/fp/fn/gold_rank/mapped_status.
    retrieval_request_count: int | None = None
    retrieval_retry_count: int | None = None
    retrieval_recovered_error_count: int | None = None
    retrieval_final_error_count: int | None = None
    retrieval_error_sources: str | None = None
    retrieval_error_types: str | None = None

    error_type: str | None = None
    error_message: str | None = None


def _sum_optional(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _sum_optional_float(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _ms_to_s(value: float | None) -> float | None:
    return None if value is None else value / 1000.0


def _extract_pipeline_timings(result: Any) -> dict[str, float]:
    rag_debug = getattr(getattr(result, "metadata", None), "rag_debug", None)
    return dict(getattr(rag_debug, "pipeline_timings", None) or {})


def _extract_pipeline_usage(result: Any) -> dict[str, dict[str, Any]]:
    rag_debug = getattr(getattr(result, "metadata", None), "rag_debug", None)
    return dict(getattr(rag_debug, "pipeline_usage", None) or {})


def _extract_retrieval_diagnostics(result: Any) -> dict[str, Any]:
    rag_debug = getattr(getattr(result, "metadata", None), "rag_debug", None)
    return dict(getattr(rag_debug, "retrieval_diagnostics", None) or {})


def _retrieval_diagnostics_row_fields(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Map a retrieval_diagnostics dict (see planned_pipeline._summarize_
    retrieval_diagnostics) onto the RowResult retrieval_* fields. Bounded
    error-source/type lists are joined into a single CSV-friendly string;
    an empty diagnostics dict (no public-API route calls recorded) leaves
    every field None rather than fabricating zeros.
    """
    if not diagnostics:
        return {
            "retrieval_request_count": None,
            "retrieval_retry_count": None,
            "retrieval_recovered_error_count": None,
            "retrieval_final_error_count": None,
            "retrieval_error_sources": None,
            "retrieval_error_types": None,
        }
    error_sources = diagnostics.get("retrieval_error_sources") or []
    error_types = diagnostics.get("retrieval_error_types") or []
    return {
        "retrieval_request_count": diagnostics.get("retrieval_request_count"),
        "retrieval_retry_count": diagnostics.get("retrieval_retry_count"),
        "retrieval_recovered_error_count": diagnostics.get("retrieval_recovered_error_count"),
        "retrieval_final_error_count": diagnostics.get("retrieval_final_error_count"),
        "retrieval_error_sources": "; ".join(error_sources) or None,
        "retrieval_error_types": "; ".join(error_types) or None,
    }


def _error_row_result(
    *,
    row: BenchmarkRow,
    run_number: int,
    run_config: BenchmarkRunConfig,
    gold_codes_normalized: list[str],
    end_to_end_seconds: float,
    exc: BaseException,
    pricing: ModelPricing,
) -> RowResult:
    """Build the RowResult for a mapping execution error.

    Recovers partial pipeline timing/usage telemetry from `exc` when present
    (see PlannedPipelineError.partial_timings/partial_usage, attached by
    PlannedPipeline.map_term() before re-raising) so a failed row still
    reports the cost/latency of billed-but-failed LLM calls -- e.g. a
    malformed-JSON retry that was itself billed before the row ultimately
    errored. When `exc` carries no such telemetry (a plain exception raised
    before QueryPlanner ever ran, or one that never passed through
    PlannedPipeline), these fields stay None exactly as before -- nothing is
    fabricated.
    """
    partial_timings: dict[str, float] = getattr(exc, "partial_timings", None) or {}
    partial_usage: dict[str, Any] = getattr(exc, "partial_usage", None) or {}
    partial_retrieval: dict[str, Any] = getattr(exc, "partial_retrieval_diagnostics", None) or {}
    planner_usage = partial_usage.get("query_planner") or {}
    reranker_usage = partial_usage.get("llm_reranker") or {}
    retrieval_fields = _retrieval_diagnostics_row_fields(partial_retrieval)

    total_input = _sum_optional(
        planner_usage.get("input_tokens"), reranker_usage.get("input_tokens")
    )
    total_cached = _sum_optional(
        planner_usage.get("cached_input_tokens"), reranker_usage.get("cached_input_tokens")
    )
    total_output = _sum_optional(
        planner_usage.get("output_tokens"), reranker_usage.get("output_tokens")
    )
    total_reasoning = _sum_optional(
        planner_usage.get("reasoning_tokens"), reranker_usage.get("reasoning_tokens")
    )
    api_cost_usd = calculate_cost_usd(
        input_tokens=total_input,
        cached_input_tokens=total_cached,
        output_tokens=total_output,
        pricing=pricing,
    )
    system_fingerprint = reranker_usage.get("system_fingerprint") or planner_usage.get(
        "system_fingerprint"
    )

    return RowResult(
        input_row=row.input_row,
        source_variable=row.source_variable,
        source_label=row.source_label,
        source_description=row.source_description,
        target_ontology=row.target_ontology,
        gold_code_raw=row.gold_code_raw,
        gold_codes_normalized=gold_codes_normalized,
        gold_target_term=row.gold_target_term,
        mapped_status="error",
        mapped_code=None,
        mapped_code_normalized=None,
        mapped_term=None,
        mapped_ontology=None,
        confidence=None,
        logic_type=None,
        alternatives=[AlternativeSlot(None, None, None, None) for _ in range(4)],
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
        planner_input_tokens=planner_usage.get("input_tokens"),
        planner_cached_input_tokens=planner_usage.get("cached_input_tokens"),
        planner_output_tokens=planner_usage.get("output_tokens"),
        planner_reasoning_tokens=planner_usage.get("reasoning_tokens"),
        reranker_input_tokens=reranker_usage.get("input_tokens"),
        reranker_cached_input_tokens=reranker_usage.get("cached_input_tokens"),
        reranker_output_tokens=reranker_usage.get("output_tokens"),
        reranker_reasoning_tokens=reranker_usage.get("reasoning_tokens"),
        total_input_tokens=total_input,
        total_cached_input_tokens=total_cached,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        api_cost_usd=api_cost_usd,
        model=run_config.model_config.model,
        requested_reasoning_effort=run_config.model_config.reasoning_effort or "N/A",
        temperature=run_config.temperature,
        temperature_mode=run_config.temperature_mode,
        seed=run_config.seed,
        retrieval_mode=run_config.retrieval_mode,
        max_alternatives=run_config.max_alternatives,
        run_number=run_number,
        system_fingerprint=system_fingerprint,
        retrieval_request_count=retrieval_fields["retrieval_request_count"],
        retrieval_retry_count=retrieval_fields["retrieval_retry_count"],
        retrieval_recovered_error_count=retrieval_fields["retrieval_recovered_error_count"],
        retrieval_final_error_count=retrieval_fields["retrieval_final_error_count"],
        retrieval_error_sources=retrieval_fields["retrieval_error_sources"],
        retrieval_error_types=retrieval_fields["retrieval_error_types"],
        error_type=type(exc).__name__,
        error_message=format_exception_chain(exc),
    )


def execute_row(
    *,
    mapper: OntologyMapper,
    row: BenchmarkRow,
    run_number: int,
    run_config: BenchmarkRunConfig,
    pricing: ModelPricing,
) -> RowResult:
    """Map one row through the real planned pipeline and score it.

    Execution exceptions are captured as mapped_status="error" rows -- never
    silently converted into an ordinary unmapped/FN outcome.
    """
    normalized_target_ontology = canonical_ontology(row.target_ontology) or row.target_ontology.upper()
    gold_codes_normalized = [
        normalize_code_for_ontology(code, normalized_target_ontology) for code in row.gold_codes
    ]

    start = time.perf_counter()
    try:
        result = mapper.map_term(
            source_term=row.source_variable,
            source_label=row.source_label,
            source_description=row.source_description,
        )
    except Exception as exc:  # noqa: BLE001 -- execution error, not a mapping decision
        return _error_row_result(
            row=row,
            run_number=run_number,
            run_config=run_config,
            gold_codes_normalized=gold_codes_normalized,
            end_to_end_seconds=time.perf_counter() - start,
            exc=exc,
            pricing=pricing,
        )
    end_to_end_seconds = time.perf_counter() - start

    is_mapped = not (
        result.target_code.upper() == _UNMAPPED_CODE and result.ontology.upper() == _UNMAPPED_ONTOLOGY
    )

    mapped_code_normalized = (
        normalize_code_for_ontology(result.target_code, result.ontology) if is_mapped else None
    )

    raw_alts = list(result.alternatives[: run_config.max_alternatives])
    padded_alts = raw_alts + [None] * max(0, 4 - len(raw_alts))
    alt_slots = [
        AlternativeSlot(None, None, None, None)
        if alt is None
        else AlternativeSlot(
            code=normalize_code_for_ontology(alt.code, alt.ontology),
            term=alt.term,
            ontology=alt.ontology,
            confidence=alt.confidence,
        )
        for alt in padded_alts[:4]
    ]

    ranked_codes: list[str | None] = [mapped_code_normalized if is_mapped else None]
    ranked_codes.extend(slot.code for slot in alt_slots)

    row_score: RowScore = score_row(
        is_mapped=is_mapped,
        ranked_codes=ranked_codes,
        gold_codes=gold_codes_normalized,
    )

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

    api_cost_usd = calculate_cost_usd(
        input_tokens=total_input,
        cached_input_tokens=total_cached,
        output_tokens=total_output,
        pricing=pricing,
    )

    system_fingerprint = reranker_usage.get("system_fingerprint") or planner_usage.get(
        "system_fingerprint"
    )

    return RowResult(
        input_row=row.input_row,
        source_variable=row.source_variable,
        source_label=row.source_label,
        source_description=row.source_description,
        target_ontology=row.target_ontology,
        gold_code_raw=row.gold_code_raw,
        gold_codes_normalized=gold_codes_normalized,
        gold_target_term=row.gold_target_term,
        mapped_status="mapped" if is_mapped else "unmapped",
        mapped_code=result.target_code if is_mapped else None,
        mapped_code_normalized=mapped_code_normalized,
        mapped_term=result.target_term if is_mapped else None,
        mapped_ontology=result.ontology if is_mapped else None,
        confidence=result.confidence,
        logic_type=result.logic_type.value,
        alternatives=alt_slots,
        gold_rank=row_score.gold_rank,
        top1_correct=row_score.top1_correct,
        top5_hit=row_score.top5_hit,
        tp=row_score.tp,
        fp=row_score.fp,
        fn=row_score.fn,
        tn=row_score.tn,
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
        planner_input_tokens=planner_usage.get("input_tokens"),
        planner_cached_input_tokens=planner_usage.get("cached_input_tokens"),
        planner_output_tokens=planner_usage.get("output_tokens"),
        planner_reasoning_tokens=planner_usage.get("reasoning_tokens"),
        reranker_input_tokens=reranker_usage.get("input_tokens"),
        reranker_cached_input_tokens=reranker_usage.get("cached_input_tokens"),
        reranker_output_tokens=reranker_usage.get("output_tokens"),
        reranker_reasoning_tokens=reranker_usage.get("reasoning_tokens"),
        total_input_tokens=total_input,
        total_cached_input_tokens=total_cached,
        total_output_tokens=total_output,
        total_reasoning_tokens=total_reasoning,
        api_cost_usd=api_cost_usd,
        model=run_config.model_config.model,
        requested_reasoning_effort=run_config.model_config.reasoning_effort or "N/A",
        temperature=run_config.temperature,
        temperature_mode=run_config.temperature_mode,
        seed=run_config.seed,
        retrieval_mode=run_config.retrieval_mode,
        max_alternatives=run_config.max_alternatives,
        run_number=run_number,
        system_fingerprint=system_fingerprint,
        retrieval_request_count=retrieval_fields["retrieval_request_count"],
        retrieval_retry_count=retrieval_fields["retrieval_retry_count"],
        retrieval_recovered_error_count=retrieval_fields["retrieval_recovered_error_count"],
        retrieval_final_error_count=retrieval_fields["retrieval_final_error_count"],
        retrieval_error_sources=retrieval_fields["retrieval_error_sources"],
        retrieval_error_types=retrieval_fields["retrieval_error_types"],
        error_type=None,
        error_message=None,
    )


def iter_run_rows(
    *,
    mappers: dict[str, OntologyMapper],
    dataset: Sequence[BenchmarkRow],
    run_number: int,
    run_config: BenchmarkRunConfig,
    pricing: ModelPricing,
) -> Iterator[RowResult]:
    """Yield one RowResult per dataset row, in dataset order, sequentially.

    A generator (rather than returning a list) so the caller can write/print
    each row as it completes -- required for incremental checkpointing of a
    long-running paid benchmark.
    """
    for row in dataset:
        normalized_ontology = canonical_ontology(row.target_ontology) or row.target_ontology.upper()
        mapper = mappers.get(normalized_ontology)
        if mapper is None:
            raise RuntimeError(
                f"No mapper configured for target_ontology={row.target_ontology!r} "
                f"(normalized={normalized_ontology!r}) at input_row={row.input_row}. "
                "This indicates a mapper-construction bug, not a mapping failure."
            )
        yield execute_row(
            mapper=mapper,
            row=row,
            run_number=run_number,
            run_config=run_config,
            pricing=pricing,
        )


def compute_run_metrics(rows: Sequence[RowResult]) -> tuple[Any, int]:
    """Aggregate scores over successfully evaluated rows (mapped_status != 'error').

    Returns (RunMetrics, error_count).
    """
    evaluated = [r for r in rows if r.mapped_status != "error"]
    error_count = len(rows) - len(evaluated)
    row_scores = [
        RowScore(
            gold_rank=r.gold_rank,
            top1_correct=r.top1_correct,
            top5_hit=r.top5_hit,
            tp=r.tp,
            fp=r.fp,
            fn=r.fn,
            tn=r.tn,
        )
        for r in evaluated
    ]
    return aggregate_scores(row_scores), error_count
