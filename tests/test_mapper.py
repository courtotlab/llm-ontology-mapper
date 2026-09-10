"""
Unit tests for OntologyMapper (mapper.py).

All LLM provider calls are mocked — no API keys needed.

Run with:  pytest tests/test_mapper.py -v -m unit
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.models import LogicType, MappingBatch, MappingResult
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse

# ─────────────────────────────────────────────────────────────────────────────
# Shared mock provider fixture
# ─────────────────────────────────────────────────────────────────────────────


class _StubProvider(BaseLLMProvider):
    """Deterministic stub — returns a hard-coded JSON response."""

    RESPONSE_JSON = """{
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "confidence": 0.93,
        "alternatives": [],
        "notes": null
    }"""

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        return CompletionResponse(
            content=self.RESPONSE_JSON,
            model="stub-model",
            prompt_tokens=50,
            completion_tokens=30,
        )


@pytest.fixture()
def stub_provider() -> _StubProvider:
    return _StubProvider(model="stub-model")


# ─────────────────────────────────────────────────────────────────────────────
# OntologyMapper construction
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mapper_accepts_injected_provider(stub_provider: _StubProvider) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    assert mapper._llm is stub_provider


@pytest.mark.unit
def test_mapper_builds_provider_from_shorthand(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMProviderFactory.from_config() should be called once with the right args."""
    from llm_ontology_mapper import mapper as mapper_module
    from llm_ontology_mapper.mapper import OntologyMapper

    mock_factory = MagicMock(return_value=MagicMock(spec=BaseLLMProvider))
    monkeypatch.setattr(mapper_module, "LLMProviderFactory", MagicMock(from_config=mock_factory))

    OntologyMapper(provider="openai", model="gpt-4o", api_key="sk-test")
    mock_factory.assert_called_once_with(provider="openai", model="gpt-4o", api_key="sk-test")


@pytest.mark.unit
def test_use_planned_pipeline_false_raises(stub_provider: _StubProvider) -> None:
    """The legacy non-planned mapping path has been removed; the planned
    pipeline is the only supported mapping architecture."""
    from llm_ontology_mapper.mapper import OntologyMapper

    with pytest.raises(ValueError, match="no longer supported"):
        OntologyMapper(llm_provider=stub_provider, use_planned_pipeline=False)


# ─────────────────────────────────────────────────────────────────────────────
# map_data_dictionary
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_map_data_dictionary_returns_batch(
    stub_provider: _StubProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper
    from llm_ontology_mapper.models import LogicType, MappingResult

    mapper = OntologyMapper(llm_provider=stub_provider)

    # Patch map_term to avoid the NotImplementedError during the refactoring phase
    fake_result = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )
    monkeypatch.setattr(mapper, "map_term", lambda **kwargs: fake_result)

    records = [
        {"field_name": "cough", "field_label": "Do you have a cough?", "field_type": "radio"},
        {"field_name": "fever", "field_label": "Do you have a fever?", "field_type": "radio"},
    ]
    batch = mapper.map_data_dictionary(records, study_id="TEST_STUDY")

    assert isinstance(batch, MappingBatch)
    assert len(batch.results) == 2
    assert batch.study_id == "TEST_STUDY"


@pytest.mark.unit
def test_map_data_dictionary_signature_accepts_source_description_field() -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    signature = inspect.signature(OntologyMapper.map_data_dictionary)

    assert "source_description_field" in signature.parameters
    assert "source_data_type_field" not in signature.parameters
    assert "source_data_type" not in signature.parameters
    assert signature.parameters["source_description_field"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.unit
def test_map_data_dictionary_existing_positional_arguments_keep_meanings(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    calls: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> MappingResult:
        calls.append(kwargs)
        return MappingResult(
            source_term=kwargs["source_term"],
            target_code="HP:0012735",
            target_term="Cough",
            ontology="HPO",
            confidence=0.93,
            logic_type=LogicType.RAG,
        )

    monkeypatch.setattr(mapper, "map_term", _spy)
    records = [{"name": "cough", "label": "Cough?", "type": "radio"}]

    batch = mapper.map_data_dictionary(records, "name", "label", "type", "phenotype", "STUDY")

    assert batch.study_id == "STUDY"
    assert batch.entity_type == "phenotype"
    assert calls[0]["source_term"] == "cough"
    assert calls[0]["source_label"] == "Cough?"
    assert calls[0]["source_type"] == "radio"
    assert calls[0]["entity_type"] == "phenotype"
    assert calls[0]["source_description"] is None


@pytest.mark.unit
def test_map_data_dictionary_forwards_row_specific_description_and_type(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    calls: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> MappingResult:
        calls.append(kwargs)
        return MappingResult(
            source_term=kwargs["source_term"],
            source_label=kwargs.get("source_label"),
            source_type=kwargs.get("source_type"),
            target_code="HP:0012735",
            target_term="Cough",
            ontology="HPO",
            confidence=0.93,
            logic_type=LogicType.RAG,
        )

    monkeypatch.setattr(mapper, "map_term", _spy)
    records = [
        {
            "field_name": "creat",
            "field_label": "Serum creatinine",
            "field_description": " Most recent serum creatinine result collected at enrolment ",
            "field_type": " decimal ",
        },
        {
            "field_name": "smoking_status",
            "field_label": "Current smoking status",
            "field_description": "Participant-reported tobacco smoking category",
            "field_type": "categorical",
        },
    ]
    original_records = copy.deepcopy(records)

    batch = mapper.map_data_dictionary(
        records,
        source_description_field="field_description",
        entity_type="measurement",
        use_planned_pipeline=True,
        retrieval_mode="public",
    )

    assert [result.source_term for result in batch.results] == ["creat", "smoking_status"]
    assert records == original_records
    assert len(calls) == 2
    assert calls[0]["source_description"] == (
        "Most recent serum creatinine result collected at enrolment"
    )
    assert calls[1]["source_description"] == "Participant-reported tobacco smoking category"
    assert calls[0]["source_type"] == "decimal"
    assert calls[1]["source_type"] == "categorical"
    assert calls[0]["source_description"] != calls[1]["source_description"]
    assert calls[0]["entity_type"] == "measurement"
    assert calls[0]["use_planned_pipeline"] is True
    assert calls[0]["retrieval_mode"] == "public"


@pytest.mark.unit
def test_map_data_dictionary_missing_description_values_become_none(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    calls: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> MappingResult:
        calls.append(kwargs)
        return MappingResult(
            source_term=kwargs["source_term"],
            target_code="HP:0012735",
            target_term="Cough",
            ontology="HPO",
            confidence=0.93,
            logic_type=LogicType.RAG,
        )

    monkeypatch.setattr(mapper, "map_term", _spy)
    records = [
        {"field_name": "missing_key", "field_type": "integer"},
        {"field_name": "empty", "field_description": "", "field_type": ""},
        {"field_name": "blank", "field_description": "   ", "field_type": "   "},
        {"field_name": "nan", "field_description": math.nan, "field_type": math.nan},
        {"field_name": "literal_nan", "field_description": "nan", "field_type": "N/A"},
    ]

    mapper.map_data_dictionary(records, source_description_field="field_description")

    assert [call["source_description"] for call in calls] == [None, None, None, None, None]
    assert [call["source_type"] for call in calls] == ["integer", None, None, None, None]
    for call in calls:
        assert call["source_description"] not in {"None", "null", "N/A", "nan"}
        assert call["source_type"] not in {"None", "null", "N/A", "nan"}


@pytest.mark.unit
def test_map_data_dictionary_pandas_na_values_become_none_when_pandas_available(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mapper,
        "map_term",
        lambda **kwargs: (
            calls.append(kwargs)
            or MappingResult(
                source_term=kwargs["source_term"],
                target_code="HP:0012735",
                target_term="Cough",
                ontology="HPO",
                confidence=0.93,
                logic_type=LogicType.RAG,
            )
        ),
    )

    mapper.map_data_dictionary(
        [{"field_name": "creat", "field_description": pd.NA, "field_type": pd.NA}],
        source_description_field="field_description",
    )

    assert calls[0]["source_description"] is None
    assert calls[0]["source_type"] is None


@pytest.mark.unit
def test_map_data_dictionary_unconfigured_description_field_is_none(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mapper,
        "map_term",
        lambda **kwargs: (
            calls.append(kwargs)
            or MappingResult(
                source_term=kwargs["source_term"],
                target_code="HP:0012735",
                target_term="Cough",
                ontology="HPO",
                confidence=0.93,
                logic_type=LogicType.RAG,
            )
        ),
    )

    mapper.map_data_dictionary(
        [{"field_name": "creat", "field_description": "available but unconfigured"}],
        source_description_field=" ",
    )

    assert calls[0]["source_description"] is None


@pytest.mark.unit
def test_map_data_dictionary_missing_required_source_term_preserves_skip_behavior(
    stub_provider: _StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)

    def _raising_for_blank_source(**kwargs: Any) -> MappingResult:
        if not kwargs["source_term"]:
            raise ValueError("source term required")
        return MappingResult(
            source_term=kwargs["source_term"],
            target_code="HP:0012735",
            target_term="Cough",
            ontology="HPO",
            confidence=0.93,
            logic_type=LogicType.RAG,
        )

    monkeypatch.setattr(mapper, "map_term", _raising_for_blank_source)

    batch = mapper.map_data_dictionary([{}, {"field_name": "good"}])

    assert [result.source_term for result in batch.results] == ["good"]


@pytest.mark.unit
def test_map_data_dictionary_skips_failed_rows(
    stub_provider: _StubProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RuntimeError on one row must not abort the entire batch."""
    from llm_ontology_mapper.mapper import OntologyMapper

    mapper = OntologyMapper(llm_provider=stub_provider)

    call_count = 0

    def _flaky(**kwargs: Any) -> MappingResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated LLM error")
        return MappingResult(
            source_term=kwargs.get("source_term", "x"),
            target_code="HP:0000001",
            target_term="X",
            ontology="HPO",
            confidence=0.5,
            logic_type=LogicType.LLM,
        )

    monkeypatch.setattr(mapper, "map_term", _flaky)

    records = [{"field_name": "bad"}, {"field_name": "good"}]
    batch = mapper.map_data_dictionary(records)

    # Only the successful row makes it into the batch
    assert len(batch.results) == 1
