"""Manual live smoke script for planned public ontology mapping.

This is a direct runnable script, not a pytest test. It intentionally uses
OntologyMapper(use_planned_pipeline=True), not AgenticMapper.

Run with:
    uv run python tests/live/planned_public_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.planned_pipeline import PlannedPipeline
from llm_ontology_mapper.providers import OllamaProvider, OpenAIProvider
from llm_ontology_mapper.public_retriever import PublicOntologyRetriever
from llm_ontology_mapper.search_tools import SearchTools

__test__ = False


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================

PROVIDER = "openai"  # "openai" or "ollama"
OPENAI_MODEL = "gpt-4.1-mini"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

SOURCE_TERM = "sys_bp"
SOURCE_LABEL = ""
CLINICAL_AREA = "measurement"
TARGET_ONTOLOGY = "LOINC"
RETRIEVAL_MODE = "public"

MAX_RESULTS_PER_QUERY = 10

RUN_TARGET_OVERRIDE_CASE = True
OVERRIDE_TARGET_ONTOLOGY = "HPO"


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


def _result_payload(result: object) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


def _pipeline_info(result: object) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None)
    rag_debug = getattr(metadata, "rag_debug", None)
    candidates = getattr(rag_debug, "candidates_retrieved", None) or []
    if candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def _is_unmapped(result: object) -> bool:
    code = str(getattr(result, "target_code", "")).upper()
    ontology = str(getattr(result, "ontology", "")).upper()
    return code in {"UNMAPPED", "UNKNOWN:UNMAPPED"} or ontology == "UNKNOWN"


def _make_mapper(
    provider: OpenAIProvider | OllamaProvider,
    *,
    target_ontology: str | None,
) -> OntologyMapper:
    # Provide explicit blank LOINC credentials so this script does not read
    # LOINC_* environment variables. LOINC public retrieval may therefore
    # return no candidates unless credentials are added in code for local use.
    search_tools = SearchTools(
        api_timeout=15,
        request_delay=0.1,
        loinc_username="",
        loinc_password="",
    )
    pipeline = PlannedPipeline(
        provider=provider,
        public_retriever=PublicOntologyRetriever(search_tools=search_tools),
    )
    return OntologyMapper(
        llm_provider=provider,
        ontologies=[target_ontology] if target_ontology else None,
        use_planned_pipeline=True,
        retrieval_mode=RETRIEVAL_MODE,
        planned_pipeline=pipeline,
        rag_top_k=MAX_RESULTS_PER_QUERY,
    )


def _print_context(
    *,
    provider_name: str,
    model: str,
    target_ontology: str | None,
) -> None:
    context: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
        "source_term": SOURCE_TERM,
        "source_label": _optional(SOURCE_LABEL),
        "target_ontology": target_ontology,
        "clinical_area": _optional(CLINICAL_AREA),
        "retrieval_mode": RETRIEVAL_MODE,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
    }
    if provider_name == "ollama":
        context["ollama_base_url"] = OLLAMA_BASE_URL
    print(json.dumps(context, indent=2))


def _run_case(
    provider: OpenAIProvider | OllamaProvider,
    *,
    target_ontology: str | None,
    override_case: bool = False,
) -> None:
    provider_name = PROVIDER.strip().lower()
    model = OPENAI_MODEL if provider_name == "openai" else OLLAMA_MODEL
    _print_context(
        provider_name=provider_name,
        model=model,
        target_ontology=target_ontology,
    )

    mapper = _make_mapper(provider, target_ontology=target_ontology)
    result = mapper.map_term(
        source_term=SOURCE_TERM,
        source_label=_optional(SOURCE_LABEL),
        entity_type=_optional(CLINICAL_AREA),
    )

    payload = _result_payload(result)
    info = _pipeline_info(result)
    print(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"selected_pipeline_metadata": info}, indent=2, default=str))

    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"

    if override_case:
        assert result.ontology in {"HPO", "HP", "UNKNOWN"}, (
            f"Expected HPO/HP/UNKNOWN in override case, got {result.ontology!r}"
        )
        assert result.ontology != "LOINC", "HPO override case must not return LOINC"
    else:
        assert result.ontology in {"LOINC", "UNKNOWN"}, (
            f"Expected LOINC or UNKNOWN, got {result.ontology!r}"
        )
        if not _is_unmapped(result):
            term = result.target_term.lower()
            assert (
                result.target_code.startswith("LOINC:")
                or "8480-6" in result.target_code
                or ("systolic" in term and "blood pressure" in term)
            ), "Mapped public result does not look like a systolic BP LOINC result"

    if info:
        assert info.get("retrieval_mode") == "public", info
        if not _is_unmapped(result) and "is_grounded" in info:
            assert info["is_grounded"] is True, info


def main() -> None:
    provider = _build_provider()
    if provider is None:
        sys.exit(0)

    _run_case(provider, target_ontology=_optional(TARGET_ONTOLOGY))
    if RUN_TARGET_OVERRIDE_CASE:
        print("\n--- Running target ontology override case ---")
        _run_case(
            provider,
            target_ontology=_optional(OVERRIDE_TARGET_ONTOLOGY),
            override_case=True,
        )
    print("PASS: planned public smoke completed")


if __name__ == "__main__":
    main()
