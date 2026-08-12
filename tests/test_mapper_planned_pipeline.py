"""
Unit tests for OntologyMapper optional PlannedPipeline integration (Phase 10).

All planned-pipeline behavior is tested with injected fakes. No live LLM,
public API, or SapBERT service is called.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.models import (
    LogicType,
    MappingBatch,
    MappingResult,
    RetrievalMode,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse

pytestmark = pytest.mark.unit


class _StubProvider(BaseLLMProvider):
    RESPONSE_JSON = """{
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "confidence": 0.93,
        "alternatives": [],
        "notes": null
    }"""

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(list(messages))
        return CompletionResponse(
            content=self.RESPONSE_JSON,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
        )


class _FakePlannedPipeline:
    def __init__(self, result: MappingResult | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or _planned_result("planned")

    def map_term(self, **kwargs: Any) -> MappingResult:
        self.calls.append(kwargs)
        return self.result


class _ForbiddenPlannedPipeline:
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    def map_term(self, **kwargs: Any) -> MappingResult:
        raise AssertionError("PlannedPipeline should not be called")


def _planned_result(source_term: str = "sys_bp") -> MappingResult:
    return MappingResult(
        source_term=source_term,
        source_label="Systolic blood pressure",
        source_type="integer",
        target_code="LOINC:8480-6",
        target_term="Systolic blood pressure",
        ontology="LOINC",
        confidence=0.91,
        logic_type=LogicType.RAG,
    )


def test_default_constructor_uses_legacy_path() -> None:
    provider = _StubProvider()
    forbidden = _ForbiddenPlannedPipeline()
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=forbidden)

    result = mapper.map_term("cough", source_label="Do you have a cough?")

    assert result.target_code == "HP:0012735"
    assert provider.calls
    assert forbidden.calls == []


def test_map_term_default_still_uses_legacy_path() -> None:
    provider = _StubProvider()
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=fake)

    result = mapper.map_term("cough")

    assert result.logic_type == LogicType.LLM
    assert provider.calls
    assert fake.calls == []


def test_planned_flag_enables_planned_pipeline_delegation() -> None:
    provider = _StubProvider()
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=provider,
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    result = mapper.map_term("sys_bp")

    assert result is fake.result
    assert len(fake.calls) == 1
    assert provider.calls == []


def test_map_term_override_enables_planned_pipeline_delegation() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(llm_provider=_StubProvider(), planned_pipeline=fake)

    mapper.map_term("sys_bp", use_planned_pipeline=True)

    assert len(fake.calls) == 1


def test_planned_mode_passes_source_term_label_and_type() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term(
        "sys_bp",
        source_label="Systolic BP",
        source_type="integer",
    )

    call = fake.calls[0]
    assert call["source_term"] == "sys_bp"
    assert call["source_label"] == "Systolic BP"
    assert call["source_type"] == "integer"


def test_map_term_signature_adds_keyword_only_source_description() -> None:
    signature = inspect.signature(OntologyMapper.map_term)

    assert "source_description" in signature.parameters
    assert "source_data_type" not in signature.parameters
    assert signature.parameters["source_description"].kind is inspect.Parameter.KEYWORD_ONLY


def test_planned_mode_passes_source_description() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term(
        "creat",
        source_label="Serum creatinine",
        source_type="decimal",
        source_description="Most recent serum creatinine result collected at enrolment",
    )

    call = fake.calls[0]
    assert call["source_description"] == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert call["source_type"] == "decimal"


def test_existing_positional_arguments_keep_their_meanings() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp", "Systolic BP", "integer", "measurement")

    call = fake.calls[0]
    assert call["source_term"] == "sys_bp"
    assert call["source_label"] == "Systolic BP"
    assert call["source_type"] == "integer"
    assert call["clinical_area"] == "measurement"
    assert call["source_description"] is None


def test_planned_mode_passes_entity_type_as_clinical_area() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp", entity_type="cardiology")

    assert fake.calls[0]["clinical_area"] == "cardiology"


@pytest.mark.parametrize(
    "mode",
    [RetrievalMode.PUBLIC, RetrievalMode.LOCAL, RetrievalMode.DISABLED],
)
def test_planned_mode_passes_constructor_retrieval_mode(mode: RetrievalMode) -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        retrieval_mode=mode,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["retrieval_mode"] == mode


@pytest.mark.parametrize("mode", ["public", "local", "disabled"])
def test_planned_mode_accepts_retrieval_mode_override_strings(mode: str) -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp", retrieval_mode=mode)

    assert fake.calls[0]["retrieval_mode"] == RetrievalMode(mode)


def test_planned_mode_passes_single_explicit_target_ontology() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        ontologies=["loinc"],
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["target_ontology"] == "LOINC"
    assert fake.calls[0]["allowed_target_ontologies"] == ["LOINC"]


def test_planned_mode_passes_single_explicit_efo_target_ontology() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        ontologies=["EFO"],
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("disease")

    assert fake.calls[0]["target_ontology"] == "EFO"
    assert fake.calls[0]["allowed_target_ontologies"] == ["EFO"]


def test_planned_mode_without_explicit_ontologies_is_unrestricted() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        ontologies=None,
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["target_ontology"] is None
    assert fake.calls[0]["allowed_target_ontologies"] is None


def test_planned_mode_empty_explicit_ontologies_is_unrestricted() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        ontologies=[],
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["target_ontology"] is None
    assert fake.calls[0]["allowed_target_ontologies"] is None


def test_planned_mode_accepts_multiple_explicit_target_ontologies() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        ontologies=[" loinc ", "HPO", "HP", "", "MONDO"],
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["target_ontology"] is None
    assert fake.calls[0]["allowed_target_ontologies"] == ["LOINC", "HPO", "MONDO"]


def test_planned_mode_returns_mapping_result_from_planned_pipeline() -> None:
    expected = _planned_result("custom")
    fake = _FakePlannedPipeline(result=expected)
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    result = mapper.map_term("sys_bp")

    assert result is expected
    assert isinstance(result, MappingResult)


def test_legacy_map_data_dictionary_remains_unchanged_by_default() -> None:
    provider = _StubProvider()
    forbidden = _ForbiddenPlannedPipeline()
    mapper = OntologyMapper(llm_provider=provider, planned_pipeline=forbidden)

    batch = mapper.map_data_dictionary(
        [
            {"field_name": "cough", "field_label": "Cough?", "field_type": "radio"},
            {"field_name": "fever", "field_label": "Fever?", "field_type": "radio"},
        ],
        study_id="STUDY",
    )

    assert isinstance(batch, MappingBatch)
    assert len(batch.results) == 2
    assert provider.calls
    assert forbidden.calls == []


def test_planned_map_data_dictionary_uses_planned_pipeline_safely() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
    )

    batch = mapper.map_data_dictionary(
        [
            {"field_name": "sys_bp", "field_label": "Systolic BP", "field_type": "integer"},
            {"field_name": "dia_bp", "field_label": "Diastolic BP", "field_type": "integer"},
        ],
        entity_type="cardiology",
        study_id="STUDY",
    )

    assert len(batch.results) == 2
    assert len(fake.calls) == 2
    assert fake.calls[0]["clinical_area"] == "cardiology"


def test_planned_map_data_dictionary_forwards_row_source_context() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        planned_pipeline=fake,
    )

    batch = mapper.map_data_dictionary(
        [
            {
                "field_name": "creat",
                "field_label": "Serum creatinine",
                "field_description": ("Most recent serum creatinine result collected at enrolment"),
                "field_type": "decimal",
            }
        ],
        source_description_field="field_description",
        entity_type="measurement",
        use_planned_pipeline=True,
        retrieval_mode=RetrievalMode.PUBLIC,
    )

    assert len(batch.results) == 1
    assert len(fake.calls) == 1
    assert fake.calls[0]["source_term"] == "creat"
    assert fake.calls[0]["source_label"] == "Serum creatinine"
    assert fake.calls[0]["source_description"] == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert fake.calls[0]["source_type"] == "decimal"
    assert fake.calls[0]["clinical_area"] == "measurement"
    assert fake.calls[0]["retrieval_mode"] == RetrievalMode.PUBLIC


def test_retrieval_mode_is_rejected_for_legacy_map_term() -> None:
    mapper = OntologyMapper(llm_provider=_StubProvider())

    with pytest.raises(ValueError, match="use_planned_pipeline=True"):
        mapper.map_term("sys_bp", retrieval_mode="public")


def test_non_default_constructor_retrieval_mode_requires_planned_mode() -> None:
    with pytest.raises(ValueError, match="use_planned_pipeline=True"):
        OntologyMapper(llm_provider=_StubProvider(), retrieval_mode="local")


def test_no_both_mode_is_accepted_in_planned_mode() -> None:
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=_FakePlannedPipeline(),
    )

    with pytest.raises(ValueError):
        mapper.map_term("sys_bp", retrieval_mode="both")


def test_provider_configuration_still_builds_provider_for_planned_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper import mapper as mapper_module

    provider = _StubProvider()
    captured: dict[str, Any] = {}

    class _Factory:
        @staticmethod
        def from_config(**kwargs: Any) -> _StubProvider:
            captured["factory_kwargs"] = kwargs
            return provider

    class _PipelineFactory:
        def __init__(self, *, provider: BaseLLMProvider) -> None:
            captured["pipeline_provider"] = provider

        def map_term(self, **kwargs: Any) -> MappingResult:
            captured["pipeline_call"] = kwargs
            return _planned_result(kwargs["source_term"])

    monkeypatch.setattr(mapper_module, "LLMProviderFactory", _Factory)
    monkeypatch.setattr(mapper_module, "PlannedPipeline", _PipelineFactory)

    mapper = OntologyMapper(
        provider="openai",
        model="gpt-test",
        api_key="sk-test",
        use_planned_pipeline=True,
    )
    result = mapper.map_term("sys_bp")

    assert result.source_term == "sys_bp"
    assert captured["factory_kwargs"] == {
        "provider": "openai",
        "model": "gpt-test",
        "api_key": "sk-test",
    }
    assert captured["pipeline_provider"] is provider


def test_planned_mode_passes_max_candidates_and_max_alternatives() -> None:
    fake = _FakePlannedPipeline()
    mapper = OntologyMapper(
        llm_provider=_StubProvider(),
        use_planned_pipeline=True,
        planned_pipeline=fake,
        max_candidates=7,
        max_alternatives=3,
    )

    mapper.map_term("sys_bp")

    assert fake.calls[0]["max_candidates"] == 7
    assert fake.calls[0]["max_alternatives"] == 3
