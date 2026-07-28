"""
Unit tests for DisabledMappingRunner (Phase 6).

Validates:
- Only disabled-mode QueryPlans are accepted
- Public/local QueryPlans raise DisabledMappingError
- Valid JSON builds a correct MappingResult
- Markdown-fenced JSON is accepted
- Malformed JSON raises DisabledMappingError
- Source fields pass through from QueryPlan
- logic_type is LogicType.LLM (not RAG)
- Pipeline metadata stored in rag_debug.candidates_retrieved[0]
- target_ontology_constraint enforcement (Python-side)
- UNMAPPED convention: UNKNOWN:UNMAPPED / UNMAPPED / UNKNOWN
- Confidence validation: strict raise on out-of-range
- No candidate objects, no normalizer/merger/reranker calls
- No both mode introduced

Constraints:
- No external API calls, no LLM calls, no retrieval
- Stub provider pattern (same as test_llm_reranker.py)
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.disabled_mapping import (
    DisabledMappingError,
    DisabledMappingRunner,
)
from llm_ontology_mapper.models import (
    LogicType,
    QueryPlan,
    RetrievalMode,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# Stub provider
# ─────────────────────────────────────────────────────────────────────────────


class _StubProvider(BaseLLMProvider):
    def __init__(self, response_content: str, model: str = "stub-model") -> None:
        super().__init__(model=model)
        self._response_content = response_content
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages, temperature=0.1, max_tokens=1024, **kwargs) -> CompletionResponse:
        self.calls.append(list(messages))
        return CompletionResponse(content=self._response_content, model=self.model)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALID_JSON = """{
  "target_code": "LOINC:8480-6",
  "target_term": "Systolic blood pressure",
  "ontology": "LOINC",
  "confidence": 0.65,
  "notes": "LLM-only mapping; retrieval disabled.",
  "alternatives": []
}"""

_UNMAPPED_JSON = """{
  "target_code": "UNKNOWN:UNMAPPED",
  "target_term": "UNMAPPED",
  "ontology": "UNKNOWN",
  "confidence": 0.0,
  "notes": "Could not map confidently.",
  "alternatives": []
}"""


def _disabled_plan(**kwargs) -> QueryPlan:
    defaults: dict = {
        "original_term": "sys_bp",
        "original_label": "Systolic blood pressure",
        "retrieval_mode": RetrievalMode.DISABLED,
        "expanded_queries": ["systolic blood pressure"],
        "inferred_meaning": "Measurement of systolic arterial blood pressure",
        "semantic_type": "measurement",
        "retrieval_disabled_reason": "User explicitly disabled retrieval.",
        "reasoning": "Disabled mode — no retrieval.",
    }
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def _runner(response: str = _VALID_JSON) -> tuple[DisabledMappingRunner, _StubProvider]:
    provider = _StubProvider(response)
    return DisabledMappingRunner(provider), provider


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mode routing
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_plan_calls_provider() -> None:
    runner, provider = _runner()
    runner.map(_disabled_plan())
    assert len(provider.calls) == 1


def test_public_plan_raises() -> None:
    runner, _ = _runner()
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    with pytest.raises(DisabledMappingError, match="[Dd]isabled"):
        runner.map(plan)


def test_local_plan_raises() -> None:
    runner, _ = _runner()
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.LOCAL,
    )
    with pytest.raises(DisabledMappingError, match="[Dd]isabled"):
        runner.map(plan)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic result construction
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_json_builds_mapping_result() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert result is not None
    assert result.target_code == "LOINC:8480-6"
    assert result.target_term == "Systolic blood pressure"
    assert result.ontology == "LOINC"
    assert result.confidence == 0.65


def test_markdown_fenced_json_accepted() -> None:
    fenced = "```json\n" + _VALID_JSON + "\n```"
    runner, _ = _runner(fenced)
    result = runner.map(_disabled_plan())
    assert result.target_code == "LOINC:8480-6"


def test_malformed_json_raises() -> None:
    runner, _ = _runner("not valid json { at all")
    with pytest.raises(DisabledMappingError, match="malformed JSON"):
        runner.map(_disabled_plan())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Source fields pass through
# ─────────────────────────────────────────────────────────────────────────────


def test_source_term_and_label_preserved() -> None:
    runner, _ = _runner()
    plan = _disabled_plan(
        original_term="diastolic_bp",
        original_label="Diastolic blood pressure",
    )
    result = runner.map(plan)
    assert result.source_term == "diastolic_bp"
    assert result.source_label == "Diastolic blood pressure"


def test_source_type_preserved_when_provided() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan(), source_type="radio")
    assert result.source_type == "radio"


def test_source_type_default_none() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert result.source_type is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Logic type
# ─────────────────────────────────────────────────────────────────────────────


def test_logic_type_is_llm() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert result.logic_type == LogicType.LLM


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline metadata in rag_debug.candidates_retrieved[0]
# ─────────────────────────────────────────────────────────────────────────────


def _meta(result) -> dict:
    """Extract pipeline metadata dict from the standard storage location."""
    return result.metadata.rag_debug.candidates_retrieved[0]


def test_metadata_includes_retrieval_mode_disabled() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["retrieval_mode"] == "disabled"


def test_metadata_is_grounded_false() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["is_grounded"] is False


def test_metadata_grounding_source_none() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["grounding_source"] == "none"


def test_metadata_policy_disabled_llm_only() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["policy"] == "disabled_llm_only"


def test_metadata_retrieval_skipped_true() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["retrieval_skipped"] is True


def test_metadata_includes_retrieval_disabled_reason() -> None:
    runner, _ = _runner()
    plan = _disabled_plan(retrieval_disabled_reason="User explicitly disabled retrieval.")
    result = runner.map(plan)
    assert "retrieval_disabled_reason" in _meta(result)
    assert "disabled" in _meta(result)["retrieval_disabled_reason"].lower()


def test_metadata_includes_query_plan_summary() -> None:
    runner, _ = _runner()
    plan = _disabled_plan(
        original_term="bp_sys",
        inferred_meaning="Systolic BP",
        semantic_type="measurement",
    )
    result = runner.map(plan)
    qp = _meta(result)["query_plan"]
    assert qp["original_term"] == "bp_sys"
    assert qp["inferred_meaning"] == "Systolic BP"
    assert qp["semantic_type"] == "measurement"


def test_metadata_candidate_count_zero() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert _meta(result)["candidate_count"] == 0


def test_metadata_model_comes_from_provider() -> None:
    runner, provider = _runner()
    result = runner.map(_disabled_plan())
    assert result.metadata.model == provider.model


def test_metadata_top_k_zero() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert result.metadata.rag_debug.top_k == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. target_ontology_constraint enforcement
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_constraint_accepted_when_matching() -> None:
    runner, _ = _runner()
    plan = _disabled_plan(target_ontology_constraint="LOINC")
    result = runner.map(plan)
    assert result.ontology == "LOINC"


def test_target_ontology_constraint_mismatch_raises() -> None:
    hpo_json = """{
      "target_code": "HP:0012735",
      "target_term": "Cough",
      "ontology": "HPO",
      "confidence": 0.7,
      "notes": "HPO mapping.",
      "alternatives": []
    }"""
    runner, _ = _runner(hpo_json)
    plan = _disabled_plan(target_ontology_constraint="LOINC")
    with pytest.raises(DisabledMappingError, match="LOINC"):
        runner.map(plan)


def test_target_ontology_mismatch_does_not_raise_for_unmapped() -> None:
    runner, _ = _runner(_UNMAPPED_JSON)
    plan = _disabled_plan(target_ontology_constraint="LOINC")
    result = runner.map(plan)
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_allowed_target_ontologies_mismatch_becomes_unmapped() -> None:
    hpo_json = """{
      "target_code": "HP:0012735",
      "target_term": "Cough",
      "ontology": "HPO",
      "confidence": 0.7,
      "notes": "HPO mapping.",
      "alternatives": []
    }"""
    runner, _ = _runner(hpo_json)
    plan = _disabled_plan(allowed_target_ontologies=["LOINC", "MONDO"])

    result = runner.map(plan)

    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"


def test_allowed_target_ontologies_none_applies_no_disabled_filter() -> None:
    hpo_json = """{
      "target_code": "HP:0012735",
      "target_term": "Cough",
      "ontology": "HPO",
      "confidence": 0.7,
      "notes": "HPO mapping.",
      "alternatives": []
    }"""
    runner, _ = _runner(hpo_json)

    result = runner.map(_disabled_plan(allowed_target_ontologies=None))

    assert result.target_code == "HP:0012735"
    assert result.ontology == "HPO"


# ─────────────────────────────────────────────────────────────────────────────
# 7. UNMAPPED convention
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_json_produces_convention() -> None:
    runner, _ = _runner(_UNMAPPED_JSON)
    result = runner.map(_disabled_plan())
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.target_term == "UNMAPPED"
    assert result.ontology == "UNKNOWN"
    assert result.confidence == 0.0


def test_empty_target_code_treated_as_unmapped() -> None:
    empty_code_json = """{
      "target_code": "",
      "target_term": "",
      "ontology": "",
      "confidence": 0.0,
      "notes": "Empty response.",
      "alternatives": []
    }"""
    runner, _ = _runner(empty_code_json)
    result = runner.map(_disabled_plan())
    assert result.target_code == "UNKNOWN:UNMAPPED"
    assert result.ontology == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Confidence validation
# ─────────────────────────────────────────────────────────────────────────────


def test_confidence_above_one_raises() -> None:
    bad_conf_json = """{
      "target_code": "LOINC:8480-6",
      "target_term": "Systolic BP",
      "ontology": "LOINC",
      "confidence": 1.5,
      "notes": "Overconfident.",
      "alternatives": []
    }"""
    runner, _ = _runner(bad_conf_json)
    with pytest.raises(DisabledMappingError, match="confidence"):
        runner.map(_disabled_plan())


def test_confidence_below_zero_raises() -> None:
    bad_conf_json = """{
      "target_code": "LOINC:8480-6",
      "target_term": "Systolic BP",
      "ontology": "LOINC",
      "confidence": -0.1,
      "notes": "Negative.",
      "alternatives": []
    }"""
    runner, _ = _runner(bad_conf_json)
    with pytest.raises(DisabledMappingError, match="confidence"):
        runner.map(_disabled_plan())


def test_confidence_boundary_zero_accepted() -> None:
    zero_json = """{
      "target_code": "LOINC:8480-6",
      "target_term": "Systolic BP",
      "ontology": "LOINC",
      "confidence": 0.0,
      "notes": "Zero confidence.",
      "alternatives": []
    }"""
    runner, _ = _runner(zero_json)
    result = runner.map(_disabled_plan())
    assert result.confidence == 0.0


def test_confidence_boundary_one_accepted() -> None:
    one_json = """{
      "target_code": "LOINC:8480-6",
      "target_term": "Systolic BP",
      "ontology": "LOINC",
      "confidence": 1.0,
      "notes": "Max confidence.",
      "alternatives": []
    }"""
    runner, _ = _runner(one_json)
    result = runner.map(_disabled_plan())
    assert result.confidence == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. No candidates, no grounded components
# ─────────────────────────────────────────────────────────────────────────────


def test_no_candidates_created() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    assert result.alternatives == []
    assert _meta(result)["candidate_count"] == 0


def test_provider_called_exactly_once() -> None:
    """Reranker, normalizer, and merger would add more provider calls; only one here."""
    runner, provider = _runner()
    runner.map(_disabled_plan())
    assert len(provider.calls) == 1


def test_result_has_no_normalizedcandidate_objects() -> None:
    """Verify no NormalizedCandidate is embedded anywhere in the result."""
    from llm_ontology_mapper.models import NormalizedCandidate

    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    # No NormalizedCandidates in alternatives
    for alt in result.alternatives:
        assert not isinstance(alt, NormalizedCandidate)


# ─────────────────────────────────────────────────────────────────────────────
# 10. No both mode
# ─────────────────────────────────────────────────────────────────────────────


def test_no_both_retrieval_mode() -> None:
    values = {m.value for m in RetrievalMode}
    assert "both" not in values
    assert len(values) == 3  # public, local, disabled


# ─────────────────────────────────────────────────────────────────────────────
# 11. Serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_result_serializes_cleanly() -> None:
    runner, _ = _runner()
    result = runner.map(_disabled_plan())
    dumped = result.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert dumped["logic_type"] == "llm"
    assert dumped["target_code"] == "LOINC:8480-6"
    rag = dumped["metadata"]["rag_debug"]
    assert rag["candidates_retrieved"][0]["retrieval_mode"] == "disabled"
    assert rag["candidates_retrieved"][0]["is_grounded"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. Public package surface
# ─────────────────────────────────────────────────────────────────────────────


def test_imports_available_from_package() -> None:
    from llm_ontology_mapper import DisabledMappingError, DisabledMappingRunner  # noqa: F401

    assert DisabledMappingRunner is not None
    assert DisabledMappingError is not None


# ─────────────────────────────────────────────────────────────────────────────
# 13. Existing phase components remain importable and unmodified
# ─────────────────────────────────────────────────────────────────────────────


def test_mapping_result_builder_still_importable() -> None:
    from llm_ontology_mapper import MappingResultBuilder, MappingResultBuilderError  # noqa: F401

    assert MappingResultBuilder is not None


def test_llm_reranker_still_importable() -> None:
    from llm_ontology_mapper import LLMReranker, LLMRerankerError  # noqa: F401

    assert LLMReranker is not None


def test_candidate_merger_still_importable() -> None:
    from llm_ontology_mapper import CandidateMergeError, CandidateMerger  # noqa: F401

    assert CandidateMerger is not None


def test_candidate_normalizer_still_importable() -> None:
    from llm_ontology_mapper import CandidateNormalizationError, CandidateNormalizer  # noqa: F401

    assert CandidateNormalizer is not None


def test_retrieval_router_still_importable() -> None:
    from llm_ontology_mapper import RetrievalRouter  # noqa: F401

    assert RetrievalRouter is not None


def test_query_planner_still_importable() -> None:
    from llm_ontology_mapper import QueryPlanner, QueryPlanningError  # noqa: F401

    assert QueryPlanner is not None


def test_model_classes_still_importable() -> None:
    from llm_ontology_mapper import (  # noqa: F401
        MappingResult,
        NormalizedCandidate,
        QueryPlan,
        RerankDecision,
        RetrievalTrace,
    )

    assert MappingResult is not None
