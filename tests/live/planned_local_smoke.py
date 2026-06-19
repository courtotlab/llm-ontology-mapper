"""Manual live smoke script for planned local SapBERT ontology mapping.

This is a direct runnable script, not a pytest test. It intentionally uses
OntologyMapper(use_planned_pipeline=True), not AgenticMapper.

Run with:
    uv run python tests/live/planned_local_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from llm_ontology_mapper.local_retriever import LocalRetrievalError, LocalSemanticRetriever
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.planned_pipeline import PlannedPipeline, PlannedPipelineError
from llm_ontology_mapper.providers import OllamaProvider, OpenAIProvider

__test__ = False


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================

PROVIDER = "openai"  # "openai" or "ollama"
OPENAI_MODEL = "gpt-4.1-mini"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

SAPBERT_URL = "http://localhost:8000"

SOURCE_TERM = "sys_bp"
SOURCE_LABEL = ""
CLINICAL_AREA = "measurement"
TARGET_ONTOLOGY = "LOINC"
RETRIEVAL_MODE = "local"

MAX_RESULTS_PER_QUERY = 10


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


def _make_mapper(provider: OpenAIProvider | OllamaProvider) -> OntologyMapper:
    pipeline = PlannedPipeline(
        provider=provider,
        local_retriever=LocalSemanticRetriever(sapbert_url=SAPBERT_URL),
    )
    target_ontology = _optional(TARGET_ONTOLOGY)
    return OntologyMapper(
        llm_provider=provider,
        ontologies=[target_ontology] if target_ontology else None,
        use_planned_pipeline=True,
        retrieval_mode=RETRIEVAL_MODE,
        planned_pipeline=pipeline,
        rag_top_k=MAX_RESULTS_PER_QUERY,
    )


def main() -> None:
    if not _optional(SAPBERT_URL):
        print("SKIP: SAPBERT_URL is blank. Edit SAPBERT_URL at the top of this file.")
        sys.exit(0)

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
        "sapbert_url": SAPBERT_URL,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
    }
    if provider_name == "ollama":
        context["ollama_base_url"] = OLLAMA_BASE_URL
    print(json.dumps(context, indent=2))

    mapper = _make_mapper(provider)
    try:
        result = mapper.map_term(
            source_term=SOURCE_TERM,
            source_label=_optional(SOURCE_LABEL),
            entity_type=_optional(CLINICAL_AREA),
        )
    except PlannedPipelineError as exc:
        cause = exc.__cause__
        if isinstance(cause, LocalRetrievalError):
            raise RuntimeError(
                f"Local SapBERT retrieval failed for SAPBERT_URL={SAPBERT_URL!r}. "
                "Confirm the local service is running and implements POST /search."
            ) from exc
        raise

    payload = _result_payload(result)
    info = _pipeline_info(result)
    print(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"selected_pipeline_metadata": info}, indent=2, default=str))

    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    assert result.ontology in {"LOINC", "UNKNOWN"}, (
        f"Expected LOINC or UNKNOWN, got {result.ontology!r}"
    )
    if info:
        assert info.get("retrieval_mode") == "local", info
        if "grounding_source" in info:
            assert info["grounding_source"] in {"local_sapbert", "none"}, info
        if not _is_unmapped(result) and "is_grounded" in info:
            assert info["is_grounded"] is True, info

    print("PASS: planned local smoke completed")


if __name__ == "__main__":
    main()
