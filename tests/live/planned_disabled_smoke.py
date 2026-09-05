"""Manual live smoke script for planned disabled-mode ontology mapping.

This is a direct runnable script, not a pytest test. It intentionally uses
OntologyMapper(use_planned_pipeline=True), not AgenticMapper.

Run with:
    uv run python tests/live/planned_disabled_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from _planned_smoke_helpers import (
    env_flag,
    extract_pipeline_metadata,
    print_alternatives_summary,
    print_full_debug_result,
    print_result_summary,
    print_trace_summary,
)
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.models import LogicType
from llm_ontology_mapper.providers import OllamaProvider, OpenAIProvider

__test__ = False


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================

PROVIDER = "openai"  # "openai" or "ollama"
OPENAI_MODEL = "gpt-4.1-mini"
OLLAMA_MODEL = "qwen2.5:14b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

SOURCE_TERM = "hr"
SOURCE_LABEL = "heart rate"
CLINICAL_AREA = "measurement"
TARGET_ONTOLOGY = "LOINC"
RETRIEVAL_MODE = "disabled"
SMOKE_DEBUG = env_flag("SMOKE_DEBUG")


def _optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _build_provider() -> OpenAIProvider | OllamaProvider | None:
    provider_name = PROVIDER.strip().lower()
    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("SKIP: OPENAI_API_KEY is required for PROVIDER='openai'.")
            return None
        return OpenAIProvider(model=OPENAI_MODEL, api_key=api_key)
    if provider_name == "ollama":
        return OllamaProvider(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    raise ValueError("PROVIDER must be 'openai' or 'ollama'.")


def _logic_value(result: object) -> str:
    logic_type = getattr(result, "logic_type")
    return str(getattr(logic_type, "value", logic_type))


def main() -> None:
    provider = _build_provider()
    if provider is None:
        sys.exit(0)

    provider_name = PROVIDER.strip().lower()
    model = OPENAI_MODEL if provider_name == "openai" else OLLAMA_MODEL
    context: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
        "source_term": SOURCE_TERM,
        "source_label": _optional(SOURCE_LABEL),
        "target_ontology": _optional(TARGET_ONTOLOGY),
        "clinical_area": _optional(CLINICAL_AREA),
        "retrieval_mode": RETRIEVAL_MODE,
        "smoke_debug": SMOKE_DEBUG,
    }
    if provider_name == "ollama":
        context["ollama_base_url"] = OLLAMA_BASE_URL
    print(json.dumps(context, indent=2))

    target_ontology = _optional(TARGET_ONTOLOGY)
    mapper = OntologyMapper(
        llm_provider=provider,
        ontologies=[target_ontology] if target_ontology else None,
        use_planned_pipeline=True,
        retrieval_mode=RETRIEVAL_MODE,
    )
    result = mapper.map_term(
        source_term=SOURCE_TERM,
        source_label=_optional(SOURCE_LABEL),
        entity_type=_optional(CLINICAL_AREA),
    )

    info = extract_pipeline_metadata(result) or {}
    print_result_summary(result)
    print_alternatives_summary(result, max_alternatives=0, always=True)
    print_trace_summary(result)
    print_full_debug_result(result, enabled=SMOKE_DEBUG)

    assert result.logic_type == LogicType.LLM, (
        f"Expected logic_type=LLM, got {_logic_value(result)!r}"
    )
    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    if target_ontology:
        expected_ontology = target_ontology.upper()
        assert result.ontology in {expected_ontology, "UNKNOWN"}, (
            f"Expected {expected_ontology!r} or UNKNOWN due to hard target, "
            f"got {result.ontology!r}"
        )
    else:
        # No target constraint: pipeline picks freely; accept any non-empty ontology.
        assert result.ontology, (
            f"result.ontology must be non-empty in unconstrained mode, "
            f"got {result.ontology!r}"
        )
    assert result.alternatives == [], "disabled mode should not return alternatives"

    assert info, "Expected disabled-mode pipeline metadata"
    assert info.get("retrieval_mode") == "disabled", info
    assert info.get("is_grounded") is False, info
    assert info.get("grounding_source") == "none", info
    assert info.get("policy") == "disabled_llm_only", info
    assert info.get("retrieval_skipped") is True, info

    print("PASS: planned disabled smoke completed")


if __name__ == "__main__":
    main()
