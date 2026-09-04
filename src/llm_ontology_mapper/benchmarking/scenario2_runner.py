"""
Scenario 2 (retrieval-mode ablation) execution: mapper/pipeline construction
for each of the three retrieval_mode values, and the per-row execution loop.

Reuses llm_ontology_mapper.benchmarking.runner and .scenario1_runner wherever
their helpers are retrieval-mode-agnostic (provider construction, LLM
preflight, pipeline timing/usage extraction, exception-chain formatting,
SapBERT health checks, error-stage classification) instead of reimplementing
them -- this module only adds what is genuinely new for Scenario 2: building
a mapper for ANY of the three retrieval_mode values from one shared run
config, and capturing grounding evidence (Part 19) alongside the standard
per-row telemetry.

retrieval_mode is the ONLY experiment-defining input that varies across the
three Scenario 2 runs (Part 3/4) -- model, reasoning_effort, temperature,
seed, max_alternatives, and strict_target_ontology (locked False, matching
scripts/run_model_benchmark.py) are identical in every mode.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.model_registry import BenchmarkModelConfig
from llm_ontology_mapper.benchmarking.pricing import ModelPricing, calculate_cost_usd
from llm_ontology_mapper.benchmarking.runner import (  # noqa: F401 -- re-exported for callers
    AlternativeSlot,
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
from llm_ontology_mapper.benchmarking.scenario1_runner import (  # noqa: F401 -- re-exported for callers
    ERROR_STAGE_LOCAL_RETRIEVAL,
    SapBertHealth,
    SapBertHealthError,
    check_sapbert_health,
    classify_error_stage,
)
from llm_ontology_mapper.benchmarking.scenario2_grounding import extract_grounding_info
from llm_ontology_mapper.local_retriever import LocalSemanticRetriever
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.ontology_identity import canonical_ontology, normalize_code_for_ontology
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import LLMCallConfig, OpenAIProvider
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.search_tools import SearchTools

RETRIEVAL_MODES: tuple[str, ...] = ("public", "local", "disabled")
STRICT_TARGET_ONTOLOGY = False  # locked -- matches scripts/run_model_benchmark.py
_UNMAPPED_CODE = "UNKNOWN:UNMAPPED"
_UNMAPPED_ONTOLOGY = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration -- retrieval_mode is the only field that legitimately
# varies between the three Scenario 2 invocations.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario2RunConfig:
    model_config: BenchmarkModelConfig
    retrieval_mode: str
    temperature: float | None = None
    seed: int = 42
    max_alternatives: int = 4
    max_results_per_query: int = 15
    max_candidates: int | None = 20
    sapbert_url: str | None = None  # required iff retrieval_mode == "local"

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}"
            )
        if self.retrieval_mode == "local" and not self.sapbert_url:
            raise ValueError("sapbert_url is required when retrieval_mode='local'")

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


def build_pipeline_and_mappers(
    *,
    provider: OpenAIProvider,
    run_config: Scenario2RunConfig,
    target_ontologies: Sequence[str],
) -> tuple[PlannedPipeline, dict[str, OntologyMapper]]:
    """Build one shared PlannedPipeline (retriever chosen by retrieval_mode)
    and one OntologyMapper per distinct target ontology in the dataset, all
    sharing that one pipeline/provider so LLM configuration is identical
    across the whole run -- only the hard per-ontology target constraint
    differs between mapper instances (same pattern as
    runner.build_pipeline_and_mappers).

    disabled mode passes neither public_retriever nor local_retriever:
    PlannedPipeline.map_term() short-circuits to DisabledMappingRunner before
    either retriever is ever called, so constructing one would be pure
    unused overhead (and, for local, a needless SapBERT dependency).
    """
    pipeline_kwargs: dict[str, Any] = {}
    if run_config.retrieval_mode == "public":
        search_tools = SearchTools(
            loinc_username=os.environ.get("LOINC_USERNAME"),
            loinc_password=os.environ.get("LOINC_PASSWORD"),
        )
        pipeline_kwargs["public_retriever"] = PublicOntologyRetriever(search_tools=search_tools)
    elif run_config.retrieval_mode == "local":
        pipeline_kwargs["local_retriever"] = LocalSemanticRetriever(sapbert_url=run_config.sapbert_url)

    pipeline = PlannedPipeline(
        provider=provider,
        llm_call_config=run_config.to_llm_call_config(),
        **pipeline_kwargs,
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
# Row result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Scenario2RowResult:
    input_row: int
    source_variable: str
    source_label: str | None
    source_description: str | None
    target_ontology: str

    gold_code_raw: str
    gold_codes_normalized: list[str]
    gold_target_term: str | None

    retrieval_mode: str

    mapped_status: str  # "mapped" | "unmapped" | "error"
    mapped_code: str | None
    mapped_code_normalized: str | None
    mapped_term: str | None
    mapped_ontology: str | None
    confidence: float | None
    logic_type: str | None

    alternatives: list[AlternativeSlot] = field(default_factory=list)

    # Part 19 -- grounding evidence (None for execution-error rows: no
    # MappingResult was ever produced to extract evidence from).
    is_grounded: bool | None = None
    grounding_source: str | None = None
    selected_code_was_retrieved: bool | None = None
    retrieval_skipped: bool | None = None

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

    model: str = ""
    requested_reasoning_effort: str = "N/A"
    temperature: float | None = None
    temperature_mode: str = "provider_default"
    seed: int = 0
    max_alternatives: int = 4
    strict_target_ontology: bool = STRICT_TARGET_ONTOLOGY

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
        codes: list[str | None] = [self.mapped_code_normalized if self.mapped_status == "mapped" else None]
        codes.extend(slot.code for slot in self.alternatives[:4])
        while len(codes) < 5:
            codes.append(None)
        return tuple(codes[:5])


def _error_row_result(
    *,
    row: BenchmarkRow,
    run_config: Scenario2RunConfig,
    gold_codes_normalized: list[str],
    end_to_end_seconds: float,
    exc: BaseException,
    pricing: ModelPricing,
) -> Scenario2RowResult:
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
    api_cost_usd = calculate_cost_usd(
        input_tokens=total_input,
        cached_input_tokens=total_cached,
        output_tokens=total_output,
        pricing=pricing,
    )

    return Scenario2RowResult(
        input_row=row.input_row,
        source_variable=row.source_variable,
        source_label=row.source_label,
        source_description=row.source_description,
        target_ontology=row.target_ontology,
        gold_code_raw=row.gold_code_raw,
        gold_codes_normalized=gold_codes_normalized,
        gold_target_term=row.gold_target_term,
        retrieval_mode=run_config.retrieval_mode,
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
        max_alternatives=run_config.max_alternatives,
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


def execute_row(
    *,
    mapper: OntologyMapper,
    row: BenchmarkRow,
    run_config: Scenario2RunConfig,
    pricing: ModelPricing,
) -> Scenario2RowResult:
    """Map one dataset row through the real planned pipeline in whichever
    retrieval_mode `mapper` was built with, and score/instrument it.

    Calls mapper.map_term(source_term=row.source_variable,
    source_label=row.source_label, source_description=row.source_description)
    -- the EXACT same call shape as runner.execute_row() (the preceding
    model-selection benchmark), with no strict_target_ontology override
    (defaults to False), so Scenario 2 never redefines what source_variable/
    source_label/source_description mean to the mapper (Part 2/21).
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

    grounding = extract_grounding_info(
        result, is_mapped=is_mapped, mapped_code_normalized=mapped_code_normalized
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

    return Scenario2RowResult(
        input_row=row.input_row,
        source_variable=row.source_variable,
        source_label=row.source_label,
        source_description=row.source_description,
        target_ontology=row.target_ontology,
        gold_code_raw=row.gold_code_raw,
        gold_codes_normalized=gold_codes_normalized,
        gold_target_term=row.gold_target_term,
        retrieval_mode=run_config.retrieval_mode,
        mapped_status="mapped" if is_mapped else "unmapped",
        mapped_code=result.target_code if is_mapped else None,
        mapped_code_normalized=mapped_code_normalized,
        mapped_term=result.target_term if is_mapped else None,
        mapped_ontology=result.ontology if is_mapped else None,
        confidence=result.confidence,
        logic_type=result.logic_type.value,
        alternatives=alt_slots,
        is_grounded=grounding.is_grounded,
        grounding_source=grounding.grounding_source,
        selected_code_was_retrieved=grounding.selected_code_was_retrieved,
        retrieval_skipped=grounding.retrieval_skipped,
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
        model=run_config.model_config.model,
        requested_reasoning_effort=run_config.model_config.reasoning_effort or "N/A",
        temperature=run_config.temperature,
        temperature_mode=run_config.temperature_mode,
        seed=run_config.seed,
        max_alternatives=run_config.max_alternatives,
        retrieval_request_count=retrieval_fields["retrieval_request_count"],
        retrieval_retry_count=retrieval_fields["retrieval_retry_count"],
        retrieval_recovered_error_count=retrieval_fields["retrieval_recovered_error_count"],
        retrieval_final_error_count=retrieval_fields["retrieval_final_error_count"],
        retrieval_error_sources=retrieval_fields["retrieval_error_sources"],
        retrieval_error_types=retrieval_fields["retrieval_error_types"],
        error_type=None,
        error_message=None,
        error_stage=None,
    )


def iter_run_rows(
    *,
    mappers: dict[str, OntologyMapper],
    dataset: Sequence[BenchmarkRow],
    run_config: Scenario2RunConfig,
    pricing: ModelPricing,
    skip_input_rows: Iterable[int] = (),
) -> Iterator[Scenario2RowResult]:
    """Yield one Scenario2RowResult per dataset row, in dataset order,
    skipping any input_row already present (resume support)."""
    skip = set(skip_input_rows)
    for row in dataset:
        if row.input_row in skip:
            continue
        normalized_ontology = canonical_ontology(row.target_ontology) or row.target_ontology.upper()
        mapper = mappers.get(normalized_ontology)
        if mapper is None:
            raise RuntimeError(
                f"No mapper configured for target_ontology={row.target_ontology!r} "
                f"(normalized={normalized_ontology!r}) at input_row={row.input_row}. "
                "This indicates a mapper-construction bug, not a mapping failure."
            )
        yield execute_row(mapper=mapper, row=row, run_config=run_config, pricing=pricing)
