from __future__ import annotations

import json
import os

import pytest

from llm_ontology_mapper.agentic_mapper import AgenticMapper
from llm_ontology_mapper.providers import OllamaProvider, OpenAIProvider
from llm_ontology_mapper.search_tools import SearchTools


# ---- Edit these values for local experiments ----

RUN_THIS_TEST = False

AGENTIC_PROVIDER = "ollama"  # "openai" or "ollama"

OPENAI_MODEL = "gpt-4.1-mini"

OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

SOURCE_TERM = "cough"
SOURCE_LABEL = "Does the patient have a cough?"

TARGET_ONTOLOGY = "HPO"
CLINICAL_AREA = "phenotype"

EXPECT_PREFIX = "HP:"
EXPECT_ONTOLOGY = "HPO"
EXPECT_LOGIC_TYPE = "agentic"
REQUIRE_MAPPED = True


pytestmark = pytest.mark.live


def _optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _provider() -> OpenAIProvider | OllamaProvider:
    provider_name = AGENTIC_PROVIDER.strip().lower()
    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI experiment requires OPENAI_API_KEY")
        return OpenAIProvider(model=OPENAI_MODEL, api_key=api_key)
    if provider_name == "ollama":
        return OllamaProvider(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    pytest.fail(
        f"Unsupported AGENTIC_PROVIDER={AGENTIC_PROVIDER!r}; "
        "expected 'openai' or 'ollama'"
    )


@pytest.mark.skipif(
    not RUN_THIS_TEST,
    reason="set RUN_THIS_TEST = True in this file to enable the experiment",
)
def test_agentic_experiment_smoke() -> None:
    source_label = _optional(SOURCE_LABEL)
    target_ontology = _optional(TARGET_ONTOLOGY)
    clinical_area = _optional(CLINICAL_AREA)

    loinc_case = (
        (target_ontology or "").upper() == "LOINC"
        or EXPECT_ONTOLOGY.strip().upper() == "LOINC"
    )
    if loinc_case and (
        not os.environ.get("LOINC_USERNAME")
        or not os.environ.get("LOINC_PASSWORD")
    ):
        pytest.skip(
            "LOINC experiment requires LOINC_USERNAME and LOINC_PASSWORD"
        )

    provider = _provider()
    mapper = AgenticMapper(
        provider=provider,
        search_tools=SearchTools(api_timeout=15, request_delay=0.1),
    )

    model = OPENAI_MODEL if AGENTIC_PROVIDER.strip().lower() == "openai" else OLLAMA_MODEL
    print(
        json.dumps(
            {
                "provider": AGENTIC_PROVIDER,
                "model": model,
                "source_term": SOURCE_TERM,
                "source_label": source_label,
                "target_ontology": target_ontology,
                "clinical_area": clinical_area,
            },
            indent=2,
        )
    )

    result = mapper.map(
        source_term=SOURCE_TERM,
        source_label=source_label,
        target_ontology=target_ontology,
        clinical_area=clinical_area,
    )

    payload = (
        result.model_dump(mode="json")
        if hasattr(result, "model_dump")
        else result
    )
    print(json.dumps(payload, indent=2, default=str))

    assert result is not None
    assert result.target_code.strip()
    assert result.target_term.strip()
    if REQUIRE_MAPPED:
        assert result.target_code != "UNKNOWN:UNMAPPED"
    if EXPECT_PREFIX:
        assert result.target_code.startswith(EXPECT_PREFIX)
    if EXPECT_ONTOLOGY:
        assert result.ontology == EXPECT_ONTOLOGY
    if EXPECT_LOGIC_TYPE:
        logic_type = getattr(result.logic_type, "value", result.logic_type)
        assert str(logic_type) == EXPECT_LOGIC_TYPE
