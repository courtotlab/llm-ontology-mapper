"""
Unit tests for llm_ontology_mapper.benchmarking.runner.

Uses stub providers/mappers -- no network, no real OpenAI calls.

Run with:  pytest tests/benchmarking/test_runner.py -v -m unit
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.model_registry import get_model_config
from llm_ontology_mapper.benchmarking.pricing import get_pricing
from llm_ontology_mapper.benchmarking.runner import (
    BenchmarkRunConfig,
    PreflightError,
    build_pipeline_and_mappers,
    describe_temperature,
    execute_row,
    format_exception_chain,
    run_preflight,
)
from llm_ontology_mapper.candidate_merger import CandidateMerger
from llm_ontology_mapper.candidate_normalizer import CandidateNormalizer
from llm_ontology_mapper.llm_reranker import LLMReranker
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.mapping_result_builder import MappingResultBuilder
from llm_ontology_mapper.models import (
    AlternativeMapping,
    LogicType,
    MappingMetadata,
    MappingResult,
    RAGDebugInfo,
)
from llm_ontology_mapper.planned_pipeline import PlannedPipeline, PlannedPipelineError
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse
from llm_ontology_mapper.query_planner import QueryPlanner, QueryPlanningError
from llm_ontology_mapper.retrieval_router import RetrievalRouter

pytestmark = pytest.mark.unit


def _row(**overrides: Any) -> BenchmarkRow:
    defaults = dict(
        input_row=1,
        source_variable="sinus_pain",
        source_label="Sinus pain/congestion",
        source_description=None,
        target_ontology="HPO",
        gold_code_raw="HP:0000245 | HP:0001742",
        gold_codes=["HP:0000245", "HP:0001742"],
        gold_target_term="Abnormal paranasal sinus morphology | Nasal congestion",
    )
    defaults.update(overrides)
    return BenchmarkRow(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature: benchmark models default to provider-default (None), not a
# mandatory numeric value -- see the GPT-5.6 Luna "temperature does not
# support 0.0" preflight failure this fixes.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini"])
def test_benchmark_run_config_defaults_to_provider_default_temperature(model: str) -> None:
    run_config = BenchmarkRunConfig(model_config=get_model_config(model), temperature=None, seed=42)

    assert run_config.temperature is None
    assert run_config.temperature_mode == "provider_default"

    llm_call_config = run_config.to_llm_call_config()
    assert llm_call_config.temperature is None
    assert llm_call_config.force_temperature is True
    assert llm_call_config.seed == 42


def test_benchmark_run_config_explicit_temperature_is_mode_explicit_and_forwarded() -> None:
    run_config = BenchmarkRunConfig(
        model_config=get_model_config("gpt-4.1-mini"), temperature=0.2, seed=42
    )

    assert run_config.temperature_mode == "explicit"
    llm_call_config = run_config.to_llm_call_config()
    assert llm_call_config.temperature == 0.2
    assert llm_call_config.force_temperature is True


def test_describe_temperature_renders_provider_default_and_numeric_values() -> None:
    assert describe_temperature(None) == "provider_default"
    assert describe_temperature(0.0) == "0.0"


def test_execute_row_records_provider_default_temperature_mode() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="HP:0000245",
        target_term="Abnormal paranasal sinus morphology",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    run_config = BenchmarkRunConfig(
        model_config=get_model_config("gpt-5.6-luna"), temperature=None, seed=42
    )
    pricing = get_pricing("gpt-5.6-luna")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert result.temperature is None
    assert result.temperature_mode == "provider_default"
    assert result.seed == 42


# ─────────────────────────────────────────────────────────────────────────────
# Target ontology hard constraint + max_alternatives plumbing
# ─────────────────────────────────────────────────────────────────────────────


def test_build_pipeline_and_mappers_creates_one_mapper_per_ontology_with_hard_constraint() -> None:
    provider = MagicMock()
    provider.model = "gpt-4.1-mini"
    run_config = BenchmarkRunConfig(
        model_config=get_model_config("gpt-4.1-mini"),
        temperature=0.0,
        seed=42,
    )

    _pipeline, mappers = build_pipeline_and_mappers(
        provider=provider,
        run_config=run_config,
        target_ontologies=["HPO", "LOINC", "MONDO"],
    )

    assert set(mappers) == {"HPO", "LOINC", "MONDO"}
    for ontology, mapper in mappers.items():
        # ontologies=[ontology] is OntologyMapper's existing one-item hard-constraint
        # mechanism -- each row's target ontology must map to exactly this.
        assert mapper.ontologies == [ontology]
        assert mapper.use_planned_pipeline is True


def test_max_alternatives_is_four_by_default() -> None:
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    assert run_config.max_alternatives == 4

    provider = MagicMock()
    provider.model = "gpt-4.1-mini"
    _pipeline, mappers = build_pipeline_and_mappers(
        provider=provider, run_config=run_config, target_ontologies=["HPO"]
    )
    assert mappers["HPO"].max_alternatives == 4


# ─────────────────────────────────────────────────────────────────────────────
# execute_row: execution errors are not scored as ordinary unmapped results
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_row_execution_error_is_not_scored_as_unmapped() -> None:
    mapper = MagicMock()
    mapper.map_term.side_effect = RuntimeError("simulated API failure sk-should-be-redacted-1234567890")
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(
        mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing
    )

    assert result.mapped_status == "error"
    assert result.error_type == "RuntimeError"
    assert "sk-" not in (result.error_message or "") or "REDACTED" in (result.error_message or "")
    # An execution error must not silently become FN=1 (the genuine-unmapped-with-gold outcome).
    assert result.tp == 0.0
    assert result.fp == 0.0
    assert result.fn == 0.0
    assert result.gold_rank is None


def test_execute_row_unmapped_with_gold_scores_fn_one() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="UNKNOWN:UNMAPPED",
        target_term="UNMAPPED",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert result.mapped_status == "unmapped"
    assert result.tp == 0.0
    assert result.fp == 0.0
    assert result.fn == 1.0


def test_execute_row_missing_alternatives_preserved_as_blank_not_synthesized() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="HP:0000245",
        target_term="Abnormal paranasal sinus morphology",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],  # no alternatives returned at all
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert len(result.alternatives) == 4
    assert all(a.code is None for a in result.alternatives)
    assert result.gold_rank == 1
    assert result.top1_correct is True


def test_execute_row_gold_rank_uses_highest_ranked_alternative() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="HP:9999999",  # not gold
        target_term="Other",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[
            AlternativeMapping(code="HP:0001742", term="Nasal congestion", ontology="HPO", confidence=0.8, source="rag"),
        ],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert result.gold_rank == 2
    assert result.top1_correct is False
    assert result.top5_hit is True


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────


def test_preflight_raises_on_rejected_parameter() -> None:
    provider = MagicMock()
    provider.model = "gpt-5.6-luna"
    provider.complete.side_effect = RuntimeError("Unrecognized request argument: seed")
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5.6-luna"), temperature=0.0, seed=42)

    with pytest.raises(PreflightError):
        run_preflight(provider, run_config.to_llm_call_config())


def test_preflight_raises_on_rejected_seed_even_with_provider_default_temperature() -> None:
    """seed=42 must still be sent (and still fail preflight loudly if the live
    API rejects it) when temperature is omitted for provider-default -- the
    switch to optional temperature must not accidentally weaken the seed
    strictness."""
    provider = MagicMock()
    provider.model = "gpt-5.6-luna"
    provider.complete.side_effect = RuntimeError("Unrecognized request argument: seed")
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5.6-luna"), temperature=None, seed=42)

    with pytest.raises(PreflightError, match="seed"):
        run_preflight(provider, run_config.to_llm_call_config())

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["seed"] == 42
    assert "temperature" not in call_kwargs or call_kwargs["temperature"] is None


def test_preflight_passes_with_provider_default_temperature() -> None:
    provider = MagicMock()
    provider.model = "gpt-5.6-luna"
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5.6-luna"), temperature=None, seed=42)

    run_preflight(provider, run_config.to_llm_call_config())  # must not raise

    call_kwargs = provider.complete.call_args.kwargs
    assert call_kwargs["temperature"] is None
    assert call_kwargs["seed"] == 42
    assert call_kwargs["reasoning_effort"] == "low"


def test_preflight_passes_when_provider_accepts_call() -> None:
    provider = MagicMock()
    provider.model = "gpt-4.1-mini"
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)

    run_preflight(provider, run_config.to_llm_call_config())  # must not raise
    assert provider.complete.called


# ─────────────────────────────────────────────────────────────────────────────
# format_exception_chain: preserves the underlying cause of a wrapped
# exception (e.g. PlannedPipelineError -> QueryPlanningError -> JSONDecodeError)
# in the benchmark CSV instead of only the generic outer wrapper message.
# ─────────────────────────────────────────────────────────────────────────────


def _make_two_level_chain() -> RuntimeError:
    try:
        raise ValueError("inner detail")
    except ValueError as inner:
        try:
            raise RuntimeError("outer wrapper") from inner
        except RuntimeError as outer:
            return outer
    raise AssertionError("unreachable")  # pragma: no cover


def _make_real_shaped_three_level_chain() -> PlannedPipelineError:
    """Reproduces the exact GPT-5 mini `nausea` failure shape: an empty LLM
    response reaching json.loads(""), wrapped by QueryPlanner, wrapped again
    by PlannedPipeline -- using the real exception classes and a real
    JSONDecodeError rather than a hand-built stand-in."""
    try:
        json.loads("")
    except json.JSONDecodeError as decode_exc:
        try:
            raise QueryPlanningError(
                f"LLM returned malformed JSON for source_term='nausea': {decode_exc}"
            ) from decode_exc
        except QueryPlanningError as planning_exc:
            try:
                raise PlannedPipelineError(
                    "QueryPlanner failed during planned mapping."
                ) from planning_exc
            except PlannedPipelineError as outer:
                return outer
    raise AssertionError("unreachable")  # pragma: no cover


def test_format_exception_chain_single_exception() -> None:
    assert format_exception_chain(RuntimeError("boom")) == "RuntimeError: boom"


def test_format_exception_chain_two_levels_includes_both() -> None:
    chain = format_exception_chain(_make_two_level_chain())

    assert chain == "RuntimeError: outer wrapper\nCaused by: ValueError: inner detail"


def test_format_exception_chain_three_levels_includes_all() -> None:
    chain = format_exception_chain(_make_real_shaped_three_level_chain())
    lines = chain.splitlines()

    assert len(lines) == 3
    assert lines[0] == "PlannedPipelineError: QueryPlanner failed during planned mapping."
    assert lines[1].startswith("Caused by: QueryPlanningError:")
    assert lines[2].startswith("Caused by: JSONDecodeError:")


def test_format_exception_chain_cyclic_references_do_not_hang() -> None:
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle

    chain = format_exception_chain(a)  # must return promptly, not hang

    assert chain.startswith("RuntimeError: a\nCaused by: RuntimeError: b")
    # The cycle must not be walked more than once per exception.
    assert chain.count("RuntimeError") == 2


def test_format_exception_chain_respects_max_depth() -> None:
    current: BaseException = RuntimeError("level-0")
    for i in range(1, 10):
        nxt = RuntimeError(f"level-{i}")
        nxt.__cause__ = current
        current = nxt  # current == level-9, chain runs back to level-0

    chain = format_exception_chain(current, max_depth=3)

    assert len(chain.splitlines()) == 3
    assert "level-9" in chain
    assert "level-7" in chain
    assert "level-6" not in chain
    assert "level-0" not in chain


def test_format_exception_chain_redacts_secrets_at_every_level() -> None:
    try:
        raise RuntimeError("token sk-abcdefgh12345678 leaked")
    except RuntimeError as inner:
        try:
            raise RuntimeError("outer sk-zzzzzzzzzzzzzzzz leaked too") from inner
        except RuntimeError as outer:
            chain = format_exception_chain(outer)

    assert "sk-" not in chain
    assert chain.count("REDACTED") == 2


def test_execute_row_error_type_is_outer_exception_class_only() -> None:
    mapper = MagicMock()
    mapper.map_term.side_effect = _make_two_level_chain()
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert result.error_type == "RuntimeError"  # outer exception class only
    assert "ValueError" in (result.error_message or "")
    assert "inner detail" in (result.error_message or "")


def test_execute_row_error_message_uses_exception_chain_serializer() -> None:
    """Reproduces the real GPT-5 mini `nausea` failure end to end through
    execute_row(): previously error_message was just "QueryPlanner failed
    during planned mapping." with the QueryPlanningError/JSONDecodeError
    detail silently discarded -- it must now be recoverable from the CSV."""
    mapper = MagicMock()
    mapper.map_term.side_effect = _make_real_shaped_three_level_chain()
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing)

    assert result.mapped_status == "error"
    assert result.error_type == "PlannedPipelineError"
    assert "PlannedPipelineError: QueryPlanner failed during planned mapping." in (
        result.error_message or ""
    )
    assert "Caused by: QueryPlanningError" in (result.error_message or "")
    assert "Caused by: JSONDecodeError" in (result.error_message or "")


# ─────────────────────────────────────────────────────────────────────────────
# Mocked end-to-end benchmark-path verification: real QueryPlanner + real
# PlannedPipeline + real OntologyMapper + real execute_row, wired to a stub
# provider so the QueryPlanner empty-response retry actually runs. Retrieval
# is stubbed to return no candidates so the rest of the pipeline is
# deterministic -- the only thing under test here is whether QueryPlanner's
# retry lets the row reach a scored outcome instead of mapped_status="error".
# No network access, no OpenAI credentials.
# ─────────────────────────────────────────────────────────────────────────────


class _PlanSequenceProvider(BaseLLMProvider):
    """OpenAI reasoning-model stub returning queued QueryPlanner responses."""

    def __init__(self, responses: list[str], model: str = "gpt-5-mini") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        content = self._responses.pop(0) if self._responses else ""
        return CompletionResponse(content=content, model=self.model)


class _UsageSequenceProviderForRunner(BaseLLMProvider):
    """Like _PlanSequenceProvider, but also reports token usage so a failed
    row's recovered partial-usage telemetry can be asserted on."""

    def __init__(self, responses: list[tuple[str, int, int]], model: str = "gpt-5-mini") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        content, prompt_tokens, completion_tokens = (
            self._responses.pop(0) if self._responses else ("", 0, 0)
        )
        return CompletionResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class _EmptyPublicRetriever:
    """No-op public retriever -- keeps everything downstream of QueryPlanner
    deterministic (an empty candidate list resolves to an unmapped, not
    errored, MappingResult without any further provider calls)."""

    def retrieve(
        self,
        query_plan: Any,
        *,
        route_plan: Any,
        max_results_per_query: int,
        route_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return []


class _SingleCandidatePublicRetriever:
    """Returns one raw HPO candidate -- enough for the reranker to actually
    call the provider (an empty candidate list would short-circuit it)."""

    def retrieve(
        self,
        query_plan: Any,
        *,
        route_plan: Any,
        max_results_per_query: int,
        route_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "code": "HP:0002017",
                "term": "Nausea",
                "ontology": "HPO",
                "score": 0.9,
                "source": "OLS",
            }
        ]


_VALID_PLAN_JSON = json.dumps(
    {
        "normalized_term": "nausea",
        "expanded_queries": ["nausea"],
        "inferred_meaning": "A sensation of unease in the stomach with an urge to vomit.",
        "semantic_type": "symptom",
        "candidate_ontologies": ["HPO"],
        "preferred_ontology": "HPO",
        "reasoning": "nausea maps to an HPO phenotype term.",
        "confidence": 0.9,
    }
)

_MALFORMED_JSON = "this is not JSON {{ broken }"

_VALID_RERANK_JSON = json.dumps(
    {
        "selected_candidate_id": "C1",
        "selected_code": "HP:0002017",
        "is_unmapped": False,
        "confidence": 0.9,
        "reasoning": "Best matching candidate for the source term.",
        "alternative_codes": [],
    }
)


def _build_end_to_end_mapper(
    provider: BaseLLMProvider,
    *,
    public_retriever: Any | None = None,
) -> OntologyMapper:
    pipeline = PlannedPipeline(
        provider=provider,
        query_planner=QueryPlanner(provider),
        retrieval_router=RetrievalRouter(),
        public_retriever=public_retriever or _EmptyPublicRetriever(),
        candidate_normalizer=CandidateNormalizer(),
        candidate_merger=CandidateMerger(),
        llm_reranker=LLMReranker(provider),
        mapping_result_builder=MappingResultBuilder(),
    )
    return OntologyMapper(
        llm_provider=provider,
        ontologies=["HPO"],
        use_planned_pipeline=True,
        retrieval_mode="public",
        planned_pipeline=pipeline,
        cache_dir=None,
    )


def test_end_to_end_query_planner_retry_success_row_is_not_error() -> None:
    provider = _PlanSequenceProvider(["", _VALID_PLAN_JSON])
    mapper = _build_end_to_end_mapper(provider)
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="nausea", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status != "error"
    assert result.error_type is None
    assert result.error_message is None
    assert len(provider.calls) == 2  # QueryPlanner: empty first attempt, then the retry


def test_end_to_end_query_planner_retry_still_empty_row_is_error_with_full_chain() -> None:
    provider = _PlanSequenceProvider(["", ""])
    mapper = _build_end_to_end_mapper(provider)
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="nausea", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status == "error"
    assert result.error_type == "PlannedPipelineError"
    assert "PlannedPipelineError: QueryPlanner failed during planned mapping." in (
        result.error_message or ""
    )
    assert "Caused by: QueryPlanningError" in (result.error_message or "")
    assert "Caused by: JSONDecodeError" in (result.error_message or "")
    assert len(provider.calls) == 2  # no third call


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark integration: malformed-JSON recovery (same real
# QueryPlanner/PlannedPipeline/OntologyMapper/execute_row stack as above).
# Reproduces the real GPT-5 mini/GPT-4.1 mini benchmark failures: a
# non-empty response that looks like JSON but fails json.loads(), with the
# same input succeeding on the other repetition.
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_query_planner_malformed_then_valid_row_is_not_error() -> None:
    provider = _PlanSequenceProvider([_MALFORMED_JSON, _VALID_PLAN_JSON])
    mapper = _build_end_to_end_mapper(provider)
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="respiratory_rate", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status != "error"
    assert result.error_type is None
    assert len(provider.calls) == 2  # QueryPlanner: malformed first attempt, then the retry


def test_end_to_end_llm_reranker_malformed_then_valid_row_succeeds() -> None:
    # Call order: QueryPlanner (valid, 1 call), then LLMReranker (malformed,
    # then a valid retry) -- all through the one shared stub provider queue.
    provider = _PlanSequenceProvider([_VALID_PLAN_JSON, _MALFORMED_JSON, _VALID_RERANK_JSON])
    mapper = _build_end_to_end_mapper(provider, public_retriever=_SingleCandidatePublicRetriever())
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="com_pancreas", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status != "error"
    assert result.error_type is None
    assert result.mapped_code == "HP:0002017"
    assert len(provider.calls) == 3  # planner (1) + reranker malformed + reranker retry


def test_end_to_end_both_responses_malformed_row_is_error_with_full_chain() -> None:
    provider = _PlanSequenceProvider([_MALFORMED_JSON, _MALFORMED_JSON])
    mapper = _build_end_to_end_mapper(provider)
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="chills", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status == "error"
    assert result.error_type == "PlannedPipelineError"
    assert "PlannedPipelineError: QueryPlanner failed during planned mapping." in (
        result.error_message or ""
    )
    assert "Caused by: QueryPlanningError" in (result.error_message or "")
    assert "Caused by: JSONDecodeError" in (result.error_message or "")
    assert len(provider.calls) == 2  # no third call


def test_end_to_end_failed_row_preserves_partial_usage_and_cost() -> None:
    """Both malformed-JSON attempts are billed API calls even though the row
    ultimately errors -- the failed row must not silently show zero cost."""
    provider = _UsageSequenceProviderForRunner(
        [(_MALFORMED_JSON, 100, 20), (_MALFORMED_JSON, 100, 20)],
        model="gpt-5-mini",
    )
    mapper = _build_end_to_end_mapper(provider)
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-5-mini"), temperature=None, seed=42)
    pricing = get_pricing("gpt-5-mini")

    result = execute_row(
        mapper=mapper,
        row=_row(source_variable="chills", gold_code_raw="HP:0002017", gold_codes=["HP:0002017"]),
        run_number=1,
        run_config=run_config,
        pricing=pricing,
    )

    assert result.mapped_status == "error"
    assert result.planner_input_tokens == 200
    assert result.planner_output_tokens == 40
    assert result.total_input_tokens == 200
    assert result.api_cost_usd is not None
    assert result.api_cost_usd > 0
    assert result.query_planner_seconds is not None
    assert result.query_planner_seconds >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval telemetry: benchmark rows can preserve retrieval retry/error
# diagnostics (see planned_pipeline._summarize_retrieval_diagnostics), on
# both a successful row and an error row, without those diagnostics
# affecting scoring.
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_row_preserves_retrieval_diagnostics_without_affecting_scoring() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="HP:0000245",
        target_term="Abnormal paranasal sinus morphology",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(
            model="m",
            provider="p",
            rag_debug=RAGDebugInfo(
                query_sent="sinus_pain",
                top_k=10,
                retrieval_diagnostics={
                    "retrieval_request_count": 3,
                    "retrieval_retry_count": 2,
                    "retrieval_recovered_error_count": 1,
                    "retrieval_final_error_count": 1,
                    "retrieval_error_sources": ["OLS:HPO", "OLS:MONDO"],
                    "retrieval_error_types": ["timeout", "http_503"],
                },
            ),
        ),
    )
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(
        mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing
    )

    assert result.retrieval_request_count == 3
    assert result.retrieval_retry_count == 2
    assert result.retrieval_recovered_error_count == 1
    assert result.retrieval_final_error_count == 1
    assert result.retrieval_error_sources == "OLS:HPO; OLS:MONDO"
    assert result.retrieval_error_types == "timeout; http_503"
    # Scoring must be entirely unaffected by retrieval diagnostics.
    assert result.mapped_status == "mapped"
    assert result.gold_rank == 1
    assert result.top1_correct is True


def test_execute_row_no_retrieval_diagnostics_leaves_fields_none() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="sinus_pain",
        target_code="HP:0000245",
        target_term="Abnormal paranasal sinus morphology",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(
        mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing
    )

    assert result.retrieval_request_count is None
    assert result.retrieval_retry_count is None
    assert result.retrieval_error_sources is None


def test_error_row_preserves_partial_retrieval_diagnostics() -> None:
    mapper = MagicMock()
    exc = PlannedPipelineError("LLMReranker failed during planned mapping.")
    exc.partial_retrieval_diagnostics = {
        "retrieval_request_count": 2,
        "retrieval_retry_count": 2,
        "retrieval_recovered_error_count": 0,
        "retrieval_final_error_count": 1,
        "retrieval_error_sources": ["OLS:NCIT"],
        "retrieval_error_types": ["connection_reset"],
    }
    mapper.map_term.side_effect = exc
    run_config = BenchmarkRunConfig(model_config=get_model_config("gpt-4.1-mini"), temperature=0.0, seed=42)
    pricing = get_pricing("gpt-4.1-mini")

    result = execute_row(
        mapper=mapper, row=_row(), run_number=1, run_config=run_config, pricing=pricing
    )

    assert result.mapped_status == "error"
    assert result.retrieval_request_count == 2
    assert result.retrieval_final_error_count == 1
    assert result.retrieval_error_sources == "OLS:NCIT"
    assert result.retrieval_error_types == "connection_reset"
    # Retrieval diagnostics never influence scoring on an error row either.
    assert result.tp == 0.0
    assert result.fp == 0.0
    assert result.fn == 0.0
    assert result.gold_rank is None
