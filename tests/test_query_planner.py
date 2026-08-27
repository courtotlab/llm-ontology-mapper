"""
Unit tests for QueryPlanner (query_planner.py).

All LLM calls use a stub provider — no API keys or network needed.

Run with:  pytest tests/test_query_planner.py -v -m unit
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ontology_mapper import query_planner as query_planner_module
from llm_ontology_mapper.models import (
    GroundingSource,
    LogicType,
    MappingResult,
    QueryPlan,
    RetrievalMode,
)
from llm_ontology_mapper.providers import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResponse,
    LLMCallConfig,
)
from llm_ontology_mapper.query_planner import QueryPlanner, QueryPlanningError

# ─────────────────────────────────────────────────────────────────────────────
# Stub provider
# ─────────────────────────────────────────────────────────────────────────────


class _RecordingStubProvider(BaseLLMProvider):
    """Returns a fixed response string and records every call for inspection."""

    def __init__(self, response_content: str, model: str = "stub") -> None:
        super().__init__(model=model)
        self._response_content = response_content
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return CompletionResponse(content=self._response_content, model=self.model)


class _NamedRecordingStubProvider(_RecordingStubProvider):
    def __init__(self, response_content: str, *, model: str, provider_name: str) -> None:
        super().__init__(response_content, model=model)
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


# ─────────────────────────────────────────────────────────────────────────────
# Shared fake LLM responses
# ─────────────────────────────────────────────────────────────────────────────

_SYS_BP_RESPONSE = json.dumps(
    {
        "normalized_term": "systolic blood pressure",
        "expanded_queries": ["systolic blood pressure", "systolic BP"],
        "inferred_meaning": (
            "systolic blood pressure — peak arterial pressure during cardiac contraction"
        ),
        "semantic_type": "measurement",
        "candidate_ontologies": ["LOINC", "HPO"],
        "preferred_ontology": "LOINC",
        "reasoning": (
            "sys_bp is a standard clinical abbreviation for systolic blood pressure, "
            "a vital sign measurement."
        ),
        "confidence": 0.95,
    }
)

_ELEVATED_SBP_RESPONSE = json.dumps(
    {
        "normalized_term": "elevated systolic blood pressure",
        "expanded_queries": [
            "elevated systolic blood pressure",
            "hypertension",
            "high blood pressure",
        ],
        "inferred_meaning": "abnormally elevated systolic blood pressure, a phenotypic finding",
        "semantic_type": "phenotype",
        "candidate_ontologies": ["HPO", "MONDO"],
        "preferred_ontology": "HPO",
        "reasoning": "Phrased as an abnormal finding. HPO is appropriate for phenotypic observations.",
        "confidence": 0.9,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def sys_bp_stub() -> _RecordingStubProvider:
    return _RecordingStubProvider(_SYS_BP_RESPONSE)


@pytest.fixture()
def sys_bp_planner(sys_bp_stub: _RecordingStubProvider) -> QueryPlanner:
    return QueryPlanner(sys_bp_stub)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: QueryPlanner calls provider with source_term in messages
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_calls_provider(sys_bp_stub: _RecordingStubProvider) -> None:
    planner = QueryPlanner(sys_bp_stub)
    planner.plan("sys_bp")

    assert len(sys_bp_stub.calls) == 1
    all_content = " ".join(m.content for m in sys_bp_stub.calls[0])
    assert "sys_bp" in all_content


@pytest.mark.unit
def test_query_planner_uses_planning_token_budget(
    sys_bp_stub: _RecordingStubProvider,
) -> None:
    planner = QueryPlanner(sys_bp_stub)
    planner.plan("sys_bp")

    assert sys_bp_stub.call_kwargs == [{"temperature": 0.1, "max_tokens": 2048}]


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.6-luna"])
def test_query_planner_uses_low_reasoning_for_selected_openai_models(model: str) -> None:
    provider = _NamedRecordingStubProvider(
        _SYS_BP_RESPONSE,
        model=model,
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    planner.plan("sys_bp")

    assert provider.call_kwargs == [
        {
            "temperature": 0.1,
            "max_tokens": 2048,
            "reasoning_effort": "low",
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize("provider_name", ["ollama", "anthropic"])
def test_query_planner_does_not_send_openai_reasoning_to_non_openai(
    provider_name: str,
) -> None:
    provider = _NamedRecordingStubProvider(
        _SYS_BP_RESPONSE,
        model="gpt-5.1",
        provider_name=provider_name,
    )
    planner = QueryPlanner(provider)

    planner.plan("sys_bp")

    assert provider.call_kwargs == [{"temperature": 0.1, "max_tokens": 2048}]


@pytest.mark.unit
def test_query_planner_records_provider_timing(
    sys_bp_stub: _RecordingStubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([4.0, 4.25])
    monkeypatch.setattr(query_planner_module.time, "monotonic", lambda: next(ticks))
    planner = QueryPlanner(sys_bp_stub)
    timings: dict[str, float] = {}

    planner.plan("sys_bp", timing_sink=timings)

    assert timings["query_planning_provider_ms"] == 250.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: QueryPlanner parses valid JSON into QueryPlan
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_parses_valid_json(sys_bp_planner: QueryPlanner) -> None:
    plan = sys_bp_planner.plan("sys_bp")

    assert isinstance(plan, QueryPlan)
    assert plan.normalized_term == "systolic blood pressure"
    assert "systolic blood pressure" in plan.expanded_queries
    assert plan.semantic_type == "measurement"
    assert plan.confidence == pytest.approx(0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: QueryPlanner preserves original_term and original_label from caller
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_preserves_original_term_and_label(
    sys_bp_planner: QueryPlanner,
) -> None:
    plan = sys_bp_planner.plan("sys_bp", source_label="Systolic Blood Pressure")

    assert plan.original_term == "sys_bp"
    assert plan.original_label == "Systolic Blood Pressure"
    # LLM normalized_term is separate; original_term must not change
    assert plan.normalized_term != plan.original_term


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: QueryPlanner uses LLM-provided expanded_queries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_uses_llm_expanded_queries(sys_bp_planner: QueryPlanner) -> None:
    plan = sys_bp_planner.plan("sys_bp")

    assert plan.expanded_queries == ["systolic blood pressure", "systolic BP"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: sys_bp is interpreted as systolic blood pressure via fake LLM output
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_sys_bp_as_systolic_blood_pressure(
    sys_bp_planner: QueryPlanner,
) -> None:
    plan = sys_bp_planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert plan.inferred_meaning is not None
    assert "systolic" in plan.inferred_meaning.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: semantic_type from fake provider output
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_semantic_type_measurement(sys_bp_planner: QueryPlanner) -> None:
    plan = sys_bp_planner.plan("sys_bp")

    assert plan.semantic_type == "measurement"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: preferred_ontology from fake provider output
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_planner_preferred_ontology_loinc(sys_bp_planner: QueryPlanner) -> None:
    plan = sys_bp_planner.plan("sys_bp")

    assert plan.preferred_ontology == "LOINC"
    assert "LOINC" in plan.candidate_ontologies


# ─────────────────────────────────────────────────────────────────────────────
# Tests 8–9: target_ontology overrides LLM preferred_ontology
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_hpo_overrides_loinc(
    sys_bp_stub: _RecordingStubProvider,
) -> None:
    # LLM returns preferred_ontology="LOINC", but caller says target_ontology="HPO"
    planner = QueryPlanner(sys_bp_stub)
    plan = planner.plan("sys_bp", target_ontology="HPO")

    assert plan.preferred_ontology == "HPO"
    assert plan.target_ontology_constraint == "HPO"
    assert plan.allowed_target_ontologies == ["HPO"]
    assert plan.candidate_ontologies == ["HPO"]


@pytest.mark.unit
def test_target_ontology_mondo_overrides_loinc(
    sys_bp_stub: _RecordingStubProvider,
) -> None:
    planner = QueryPlanner(sys_bp_stub)
    plan = planner.plan("sys_bp", target_ontology="MONDO")

    assert plan.preferred_ontology == "MONDO"
    assert plan.target_ontology_constraint == "MONDO"
    assert plan.allowed_target_ontologies == ["MONDO"]
    assert plan.candidate_ontologies == ["MONDO"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests 10–12: retrieval_mode is enforced by Python
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_retrieval_mode_public_enforced(sys_bp_planner: QueryPlanner) -> None:
    plan = sys_bp_planner.plan("sys_bp", retrieval_mode=RetrievalMode.PUBLIC)

    assert plan.retrieval_mode == RetrievalMode.PUBLIC
    assert plan.route_public_apis is True
    assert plan.route_local is False
    assert plan.retrieval_disabled is False


@pytest.mark.unit
def test_retrieval_mode_local_enforced(sys_bp_stub: _RecordingStubProvider) -> None:
    planner = QueryPlanner(sys_bp_stub)
    plan = planner.plan("sys_bp", retrieval_mode=RetrievalMode.LOCAL)

    assert plan.retrieval_mode == RetrievalMode.LOCAL
    assert plan.route_local is True
    assert plan.route_public_apis is False
    assert plan.retrieval_disabled is False


@pytest.mark.unit
def test_retrieval_mode_disabled_enforced(sys_bp_stub: _RecordingStubProvider) -> None:
    planner = QueryPlanner(sys_bp_stub)
    plan = planner.plan("sys_bp", retrieval_mode=RetrievalMode.DISABLED)

    assert plan.retrieval_mode == RetrievalMode.DISABLED
    assert plan.retrieval_disabled is True
    assert plan.route_public_apis is False
    assert plan.route_local is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: disabled mode populates retrieval_disabled_reason
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disabled_mode_populates_retrieval_disabled_reason(
    sys_bp_stub: _RecordingStubProvider,
) -> None:
    planner = QueryPlanner(sys_bp_stub)
    plan = planner.plan("sys_bp", retrieval_mode=RetrievalMode.DISABLED)

    assert plan.retrieval_disabled_reason is not None
    assert len(plan.retrieval_disabled_reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: no user-facing both mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_both_mode(sys_bp_planner: QueryPlanner) -> None:
    assert not hasattr(RetrievalMode, "BOTH")
    with pytest.raises(ValueError):
        sys_bp_planner.plan("sys_bp", retrieval_mode="both")


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: markdown code fences are stripped before JSON parsing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handles_markdown_code_fences() -> None:
    fenced = f"```json\n{_SYS_BP_RESPONSE}\n```"
    stub = _RecordingStubProvider(fenced)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"


@pytest.mark.unit
def test_handles_plain_code_fences() -> None:
    fenced = f"```\n{_SYS_BP_RESPONSE}\n```"
    stub = _RecordingStubProvider(fenced)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: fallback for empty expanded_queries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_fallback_expanded_queries_uses_source_label_when_present() -> None:
    response = json.dumps(
        {
            "normalized_term": "systolic blood pressure",
            "expanded_queries": [],
            "inferred_meaning": "systolic blood pressure",
            "semantic_type": "measurement",
            "candidate_ontologies": ["LOINC"],
            "preferred_ontology": "LOINC",
            "reasoning": "abbreviation",
            "confidence": 0.8,
        }
    )
    planner = QueryPlanner(_RecordingStubProvider(response))
    plan = planner.plan("sys_bp", source_label="Systolic Blood Pressure")

    assert plan.expanded_queries == ["Systolic Blood Pressure"]


@pytest.mark.unit
def test_fallback_expanded_queries_uses_source_term_when_no_label() -> None:
    response = json.dumps(
        {
            "normalized_term": "systolic blood pressure",
            "expanded_queries": [],
            "inferred_meaning": "systolic blood pressure",
            "semantic_type": "measurement",
            "candidate_ontologies": ["LOINC"],
            "preferred_ontology": "LOINC",
            "reasoning": "abbreviation",
            "confidence": 0.8,
        }
    )
    planner = QueryPlanner(_RecordingStubProvider(response))
    plan = planner.plan("sys_bp")

    assert plan.expanded_queries == ["sys_bp"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: malformed JSON raises QueryPlanningError
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_raises_query_planning_error_on_malformed_json() -> None:
    stub = _RecordingStubProvider("this is not JSON {{ broken }")
    planner = QueryPlanner(stub)

    with pytest.raises(QueryPlanningError, match="malformed JSON"):
        planner.plan("sys_bp")


@pytest.mark.unit
def test_raises_query_planning_error_on_empty_response() -> None:
    stub = _RecordingStubProvider("")
    planner = QueryPlanner(stub)

    with pytest.raises(QueryPlanningError):
        planner.plan("sys_bp")


# ─────────────────────────────────────────────────────────────────────────────
# Tests 18–19: infrastructure guarantees (no credentials, no external APIs)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_does_not_require_external_credentials() -> None:
    # Stub provider needs no API key. plan() must succeed without any network.
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert plan.original_term == "sys_bp"
    assert stub.calls  # provider was called exactly once


@pytest.mark.unit
def test_does_not_call_external_ontology_apis() -> None:
    # Stub intercepts all provider calls; no real HTTP is made.
    # Reaching this assertion without a network exception confirms no external call.
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert plan is not None
    assert len(stub.calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: existing model and pipeline contracts are unaffected
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_model_imports_unaffected() -> None:
    assert RetrievalMode.PUBLIC.value == "public"
    assert RetrievalMode.LOCAL.value == "local"
    assert RetrievalMode.DISABLED.value == "disabled"
    assert GroundingSource.NONE.value == "none"

    plan = QueryPlan(original_term="cough")
    assert plan.retrieval_mode == RetrievalMode.PUBLIC
    assert plan.route_public_apis is True

    r = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )
    assert r.target_code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge-case tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_context_fields_appear_in_messages() -> None:
    # source_label and clinical_area should appear in the prompt sent to the provider.
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)
    planner.plan(
        "sys_bp",
        source_label="Systolic BP reading",
        clinical_area="cardiology",
        target_ontology="LOINC",
    )

    all_content = " ".join(m.content for m in stub.calls[0])
    assert "Systolic BP reading" in all_content
    assert "cardiology" in all_content
    assert "LOINC" in all_content


@pytest.mark.unit
def test_source_description_and_type_appear_in_messages_and_plan() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "creat",
        source_label="Serum creatinine",
        source_description="Most recent serum creatinine result collected at enrolment",
        source_type="decimal",
        clinical_area="measurement",
        target_ontology="LOINC",
    )

    all_content = " ".join(m.content for m in stub.calls[0])
    assert (
        "Source description: Most recent serum creatinine result collected at enrolment"
        in all_content
    )
    assert "Source data type: decimal" in all_content
    assert "- Clinical area: measurement" in all_content
    assert plan.source_description == ("Most recent serum creatinine result collected at enrolment")
    assert plan.source_type == "decimal"
    assert plan.retrieval_mode == RetrievalMode.PUBLIC
    assert plan.target_ontology_constraint == "LOINC"
    assert plan.candidate_ontologies == ["LOINC"]


@pytest.mark.unit
def test_source_context_is_omitted_when_absent_or_blank() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "sys_bp",
        source_description="   ",
        source_type="",
    )

    all_content = " ".join(m.content for m in stub.calls[0])
    assert "Source description:" not in all_content
    assert "Source data type:" not in all_content
    assert "None" not in all_content
    assert "null" not in all_content
    assert "N/A" not in all_content
    assert plan.source_description is None
    assert plan.source_type is None


@pytest.mark.unit
def test_source_context_values_are_stripped_before_prompt_and_plan() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "creat",
        source_description="  collected at enrolment  ",
        source_type="  decimal  ",
        clinical_area="measurement",
    )

    all_content = " ".join(m.content for m in stub.calls[0])
    assert "Source description: collected at enrolment" in all_content
    assert "Source data type: decimal" in all_content
    assert "  collected at enrolment  " not in all_content
    assert "  decimal  " not in all_content
    assert plan.source_description == "collected at enrolment"
    assert plan.source_type == "decimal"


@pytest.mark.unit
def test_source_type_is_not_rendered_as_clinical_area() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    planner.plan(
        "creat",
        source_type="decimal",
        clinical_area="measurement",
    )

    all_content = " ".join(m.content for m in stub.calls[0])
    assert "Source data type: decimal" in all_content
    assert "Clinical area: measurement" in all_content
    assert "Clinical area: decimal" not in all_content


@pytest.mark.unit
def test_structured_source_context_values_are_rejected() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    with pytest.raises(TypeError, match="source_description"):
        planner.plan("sys_bp", source_description={"text": "bad"})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="source_type"):
        planner.plan("sys_bp", source_type=["decimal"])  # type: ignore[arg-type]


@pytest.mark.unit
def test_target_ontology_absent_uses_llm_candidate_ontologies() -> None:
    # When no target_ontology is supplied, the LLM's candidate_ontologies are used as-is.
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert "LOINC" in plan.candidate_ontologies
    assert "HPO" in plan.candidate_ontologies
    assert plan.target_ontology_constraint is None
    assert plan.allowed_target_ontologies is None


@pytest.mark.unit
def test_allowed_target_ontologies_filter_planner_candidates() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "sys_bp",
        allowed_target_ontologies=["MONDO", "HPO"],
    )

    assert plan.allowed_target_ontologies == ["MONDO", "HPO"]
    assert plan.target_ontology_constraint is None
    assert plan.candidate_ontologies == ["HPO"]
    assert plan.preferred_ontology == "HPO"
    assert "LOINC" not in plan.candidate_ontologies


@pytest.mark.unit
def test_allowed_target_ontologies_fall_back_when_planner_selects_none() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "sys_bp",
        allowed_target_ontologies=["MONDO", "NCIT"],
    )

    assert plan.allowed_target_ontologies == ["MONDO", "NCIT"]
    assert plan.candidate_ontologies == ["MONDO", "NCIT"]
    assert plan.preferred_ontology is None


@pytest.mark.unit
def test_allowed_target_ontologies_are_deduplicated_and_blank_filtered() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan(
        "sys_bp",
        allowed_target_ontologies=[" loinc ", "", "LOINC", " hpo "],
    )

    assert plan.allowed_target_ontologies == ["LOINC", "HPO"]
    assert plan.candidate_ontologies == ["LOINC", "HPO"]
    assert plan.preferred_ontology == "LOINC"


@pytest.mark.unit
def test_confidence_clamped_to_valid_range() -> None:
    # LLM returns confidence > 1.0; Python must clamp it.
    response = json.dumps(
        {
            "normalized_term": "systolic blood pressure",
            "expanded_queries": ["systolic blood pressure"],
            "inferred_meaning": "systolic blood pressure",
            "semantic_type": "measurement",
            "candidate_ontologies": ["LOINC"],
            "preferred_ontology": "LOINC",
            "reasoning": "test",
            "confidence": 1.5,
        }
    )
    stub = _RecordingStubProvider(response)
    planner = QueryPlanner(stub)
    plan = planner.plan("sys_bp")

    assert plan.confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_elevated_sbp_phenotype_from_fake_output() -> None:
    stub = _RecordingStubProvider(_ELEVATED_SBP_RESPONSE)
    planner = QueryPlanner(stub)
    plan = planner.plan("elevated systolic blood pressure")

    assert plan.semantic_type == "phenotype"
    assert plan.preferred_ontology == "HPO"
    assert "HPO" in plan.candidate_ontologies


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning-model empty-response retry
#
# Mirrors LLMReranker's equivalent empty-response-from-reasoning-model retry
# (see tests/test_llm_reranker.py::test_reasoning_model_empty_response_retries_once_with_larger_budget).
# Root cause: a GPT-5-family reasoning model can spend its whole completion
# budget on reasoning tokens and return content="" with finish_reason="length"
# without the provider raising -- QueryPlanner previously had no recovery for
# this and failed with a bare QueryPlanningError (see the GPT-5 mini "nausea"
# benchmark failure this fixes).
# ─────────────────────────────────────────────────────────────────────────────


class _SequenceProvider(BaseLLMProvider):
    """Returns response strings in order and records every call."""

    def __init__(self, responses: list[str], model: str = "stub") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        content = self._responses.pop(0) if self._responses else ""
        return CompletionResponse(content=content, model=self.model)


class _NamedSequenceProvider(_SequenceProvider):
    def __init__(self, responses: list[str], *, model: str, provider_name: str) -> None:
        super().__init__(responses, model=model)
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name


class _UsageSequenceProvider(BaseLLMProvider):
    """Returns (content, prompt_tokens, completion_tokens) tuples in order."""

    def __init__(self, responses: list[tuple[str, int, int]], model: str = "gpt-5") -> None:
        super().__init__(model=model)
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict[str, Any]] = []

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
        self.call_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        content, prompt_tokens, completion_tokens = (
            self._responses.pop(0) if self._responses else ("", 0, 0)
        )
        return CompletionResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


@pytest.mark.unit
def test_non_reasoning_model_valid_content_makes_one_call() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE, model="gpt-4.1-mini")
    planner = QueryPlanner(stub)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(stub.calls) == 1


@pytest.mark.unit
def test_reasoning_model_valid_content_makes_one_call() -> None:
    provider = _NamedRecordingStubProvider(_SYS_BP_RESPONSE, model="gpt-5", provider_name="openai")
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 1


@pytest.mark.unit
def test_reasoning_model_empty_then_valid_retries_once_with_larger_budget() -> None:
    provider = _NamedSequenceProvider(
        ["", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2
    assert provider.call_kwargs[0]["max_tokens"] == 2048
    assert "min_completion_tokens" not in provider.call_kwargs[0]
    assert provider.call_kwargs[1]["max_tokens"] == 8192
    assert provider.call_kwargs[1]["min_completion_tokens"] == 8192


@pytest.mark.unit
def test_reasoning_model_whitespace_only_then_valid_retries_once() -> None:
    provider = _NamedSequenceProvider(
        ["   \n\t  ", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2


@pytest.mark.unit
def test_reasoning_model_empty_twice_fails_after_exactly_two_calls() -> None:
    provider = _NamedSequenceProvider(
        ["", ""],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    with pytest.raises(QueryPlanningError):
        planner.plan("sys_bp")

    assert len(provider.calls) == 2


@pytest.mark.unit
def test_non_reasoning_model_empty_content_does_not_retry() -> None:
    provider = _SequenceProvider(["", _SYS_BP_RESPONSE], model="gpt-4.1-mini")
    planner = QueryPlanner(provider)

    with pytest.raises(QueryPlanningError):
        planner.plan("sys_bp")

    # No reasoning-model retry -- the second (valid) response is never consumed.
    assert len(provider.calls) == 1


@pytest.mark.unit
def test_reasoning_model_nonempty_malformed_json_retries_once() -> None:
    """Malformed-JSON recovery (added alongside the empty-response recovery)
    applies to any model, including reasoning models -- see the dedicated
    malformed-JSON retry test section below."""
    provider = _NamedSequenceProvider(
        ["this is not JSON {{ broken }", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2


@pytest.mark.unit
def test_retry_preserves_llm_call_config() -> None:
    provider = _NamedSequenceProvider(
        ["", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    llm_call_config = LLMCallConfig(
        temperature=0.3,
        seed=42,
        reasoning_effort="low",
        force_temperature=True,
    )
    planner = QueryPlanner(provider, llm_call_config=llm_call_config)

    planner.plan("sys_bp")

    assert len(provider.call_kwargs) == 2
    for call_kwargs in provider.call_kwargs:
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["seed"] == 42
        assert call_kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
def test_first_call_ceiling_remains_2048_regardless_of_retry() -> None:
    provider = _NamedSequenceProvider(
        ["", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    planner.plan("sys_bp")

    assert provider.call_kwargs[0]["max_tokens"] == 2048


@pytest.mark.unit
def test_larger_budget_used_only_on_retry_not_first_call() -> None:
    provider = _NamedRecordingStubProvider(_SYS_BP_RESPONSE, model="gpt-5", provider_name="openai")
    planner = QueryPlanner(provider)

    planner.plan("sys_bp")

    assert len(provider.calls) == 1
    assert provider.call_kwargs[0]["max_tokens"] == 2048
    assert "min_completion_tokens" not in provider.call_kwargs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Usage-sink accumulation across the empty-response retry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_usage_sink_accumulates_across_empty_response_retry() -> None:
    provider = _UsageSequenceProvider(
        [("", 100, 10), (_SYS_BP_RESPONSE, 100, 50)],
        model="gpt-5",
    )
    planner = QueryPlanner(provider)
    usage: dict[str, Any] = {}

    planner.plan("sys_bp", usage_sink=usage)

    planner_usage = usage["query_planner"]
    assert planner_usage["input_tokens"] == 200
    assert planner_usage["output_tokens"] == 60
    assert planner_usage["call_count"] == 2


@pytest.mark.unit
def test_usage_sink_no_retry_matches_single_call_usage() -> None:
    provider = _UsageSequenceProvider(
        [(_SYS_BP_RESPONSE, 120, 45)],
        model="gpt-5",
    )
    planner = QueryPlanner(provider)
    usage: dict[str, Any] = {}

    planner.plan("sys_bp", usage_sink=usage)

    planner_usage = usage["query_planner"]
    assert planner_usage["input_tokens"] == 120
    assert planner_usage["output_tokens"] == 45
    assert planner_usage["call_count"] == 1
    assert len(provider.calls) == 1


@pytest.mark.unit
def test_retry_usage_recorded_under_query_planner_key_only() -> None:
    provider = _UsageSequenceProvider(
        [("", 100, 10), (_SYS_BP_RESPONSE, 100, 50)],
        model="gpt-5",
    )
    planner = QueryPlanner(provider)
    usage: dict[str, Any] = {}

    planner.plan("sys_bp", usage_sink=usage)

    assert set(usage.keys()) == {"query_planner"}


@pytest.mark.unit
def test_timing_sink_accumulates_across_empty_response_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 0.1, 0.1, 0.35])  # call 1: 100ms, call 2: 250ms
    monkeypatch.setattr(query_planner_module.time, "monotonic", lambda: next(ticks))
    provider = _NamedSequenceProvider(
        ["", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)
    timings: dict[str, float] = {}

    planner.plan("sys_bp", timing_sink=timings)

    assert timings["query_planning_provider_ms"] == pytest.approx(350.0)


# ─────────────────────────────────────────────────────────────────────────────
# Malformed non-empty JSON retry
#
# A second, distinct structured-output failure mode from the empty-response
# case above: a non-empty response that looks like JSON but fails
# json.loads() (truncated, missing comma, unterminated string, ...).
# Model-agnostic (unlike the empty-response retry, which is reasoning-model
# only) -- observed on both GPT-5 mini (QueryPlanner) and GPT-4.1 mini
# (LLMReranker) in real benchmark runs, with the same input succeeding on the
# other repetition, i.e. stochastic formatting failure rather than a
# deterministic input problem. Hard cap: at most one recovery attempt total,
# whether the first failure was empty content or malformed JSON -- see
# _MAX_PLAN_ATTEMPTS.
# ─────────────────────────────────────────────────────────────────────────────

_MALFORMED_JSON = "this is not JSON {{ broken }"
_MALFORMED_JSON_2 = '{"normalized_term": "nope", '  # different truncation


@pytest.mark.unit
def test_malformed_json_valid_first_response_makes_one_call() -> None:
    stub = _RecordingStubProvider(_SYS_BP_RESPONSE)
    planner = QueryPlanner(stub)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(stub.calls) == 1


@pytest.mark.unit
def test_malformed_then_valid_retries_once_and_succeeds() -> None:
    provider = _SequenceProvider([_MALFORMED_JSON, _SYS_BP_RESPONSE])
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2


@pytest.mark.unit
def test_malformed_then_malformed_fails_after_exactly_two_calls() -> None:
    provider = _SequenceProvider([_MALFORMED_JSON, _MALFORMED_JSON_2])
    planner = QueryPlanner(provider)

    with pytest.raises(QueryPlanningError, match="malformed JSON"):
        planner.plan("sys_bp")

    assert len(provider.calls) == 2


@pytest.mark.unit
def test_empty_reasoning_response_then_valid_still_works() -> None:
    """Existing empty-response recovery is unaffected by adding the
    malformed-JSON recovery."""
    provider = _NamedSequenceProvider(
        ["", _SYS_BP_RESPONSE],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2


@pytest.mark.unit
def test_empty_first_then_malformed_second_fails_after_exactly_two_calls() -> None:
    """The empty-response retry's outcome is never re-triaged: if the one
    recovery attempt comes back malformed instead of empty, that is still
    the final attempt -- no third (malformed-JSON) retry is layered on."""
    provider = _NamedSequenceProvider(
        ["", _MALFORMED_JSON],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    with pytest.raises(QueryPlanningError, match="malformed JSON"):
        planner.plan("sys_bp")

    assert len(provider.calls) == 2


@pytest.mark.unit
def test_malformed_first_then_empty_second_fails_after_exactly_two_calls() -> None:
    """Symmetric to the above: the malformed-JSON retry's outcome is never
    re-triaged either -- if it comes back empty, that is still the final
    attempt, not a trigger for the empty-response retry."""
    provider = _NamedSequenceProvider(
        [_MALFORMED_JSON, ""],
        model="gpt-5",
        provider_name="openai",
    )
    planner = QueryPlanner(provider)

    with pytest.raises(QueryPlanningError):
        planner.plan("sys_bp")

    assert len(provider.calls) == 2


@pytest.mark.unit
def test_non_reasoning_model_malformed_then_valid_still_retries() -> None:
    """Malformed-JSON recovery is NOT reasoning-model-specific, unlike the
    empty-response recovery."""
    provider = _SequenceProvider([_MALFORMED_JSON, _SYS_BP_RESPONSE], model="gpt-4.1-mini")
    planner = QueryPlanner(provider)

    plan = planner.plan("sys_bp")

    assert plan.normalized_term == "systolic blood pressure"
    assert len(provider.calls) == 2


@pytest.mark.unit
def test_malformed_retry_preserves_llm_call_config() -> None:
    provider = _SequenceProvider([_MALFORMED_JSON, _SYS_BP_RESPONSE], model="gpt-5")
    llm_call_config = LLMCallConfig(
        temperature=0.3,
        seed=42,
        reasoning_effort="low",
        force_temperature=True,
    )
    planner = QueryPlanner(provider, llm_call_config=llm_call_config)

    planner.plan("sys_bp")

    assert len(provider.call_kwargs) == 2
    for call_kwargs in provider.call_kwargs:
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["seed"] == 42
        assert call_kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
def test_malformed_retry_uses_normal_2048_token_limit_not_larger_budget() -> None:
    provider = _SequenceProvider([_MALFORMED_JSON, _SYS_BP_RESPONSE], model="gpt-5")
    planner = QueryPlanner(provider)

    planner.plan("sys_bp")

    assert provider.call_kwargs[0]["max_tokens"] == 2048
    assert provider.call_kwargs[1]["max_tokens"] == 2048
    assert "min_completion_tokens" not in provider.call_kwargs[1]


@pytest.mark.unit
def test_malformed_retry_usage_and_timing_accumulate_across_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 0.1, 0.1, 0.3])  # call 1: 100ms, call 2: 200ms
    monkeypatch.setattr(query_planner_module.time, "monotonic", lambda: next(ticks))
    provider = _UsageSequenceProvider(
        [(_MALFORMED_JSON, 80, 12), (_SYS_BP_RESPONSE, 90, 40)],
        model="gpt-5",
    )
    planner = QueryPlanner(provider)
    usage: dict[str, Any] = {}
    timings: dict[str, float] = {}

    planner.plan("sys_bp", usage_sink=usage, timing_sink=timings)

    planner_usage = usage["query_planner"]
    assert planner_usage["input_tokens"] == 170
    assert planner_usage["output_tokens"] == 52
    assert planner_usage["call_count"] == 2
    assert timings["query_planning_provider_ms"] == pytest.approx(300.0)
