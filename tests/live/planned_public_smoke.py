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

from _planned_smoke_helpers import (
    env_flag,
    extract_pipeline_metadata,
    print_alternatives_summary,
    print_full_debug_result,
    print_result_summary,
    print_trace_summary,
)
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
OPENAI_MODEL = "gpt-5"
OLLAMA_MODEL = "gpt-oss:120b"
OLLAMA_BASE_URL = "http://localhost:11528"

SOURCE_TERM = "sys_bp"
SOURCE_LABEL = ""
CLINICAL_AREA = "measurement"
TARGET_ONTOLOGY = "LOINC"
RETRIEVAL_MODE = "public"

MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "6"))
MAX_ALTERNATIVES = int(os.environ.get("MAX_ALTERNATIVES", "5"))
SMOKE_DEBUG = env_flag("SMOKE_DEBUG")

RUN_TARGET_OVERRIDE_CASE = False
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


def _is_unmapped(result: object) -> bool:
    code = str(getattr(result, "target_code", "")).upper()
    ontology = str(getattr(result, "ontology", "")).upper()
    return code in {"UNMAPPED", "UNKNOWN:UNMAPPED"} or ontology == "UNKNOWN"


# All code prefixes that the public retrieval routes can legitimately emit, derived
# from search_tools._normalize_code prefix_map and OLS_ONTOLOGY_MAP.
_KNOWN_CODE_PREFIXES: tuple[str, ...] = (
    "CHEBI:", "DOID:", "GO:", "HP:", "ICD10:", "LOINC:",
    "MESH:", "MONDO:", "NCIT:", "RXNORM:", "SNOMEDCT:",
    "UBERON:", "UO:",
)


def _validate_alternatives(result: object) -> None:
    alternatives = getattr(result, "alternatives", [])
    assert len(alternatives) <= MAX_ALTERNATIVES

    for alt in alternatives:
        assert alt.code
        # Guard: bare UMLS CUIs (C0000000) must not leak as ontology codes.
        # Positive check: every legitimate code carries a known ontology prefix.
        assert alt.code.startswith(_KNOWN_CODE_PREFIXES), (
            f"alt.code lacks a known ontology prefix (raw CUI leak or unknown route?): "
            f"{alt.code!r}"
        )
        # Guard: doubled/embedded namespace (e.g. "SNOMEDCT:SNOMED:768500006") must
        # not appear — the code body after the first ":" must contain no further ":".
        _, _, _body = alt.code.partition(":")
        assert ":" not in _body, (
            f"doubled/embedded namespace in alt.code: {alt.code!r}"
        )
        assert alt.term
        assert alt.ontology
        assert 0.0 <= alt.confidence <= 1.0
        assert getattr(alt, "explanation", None), (
            "each alternative should include an explanation"
        )
        if not _is_unmapped(result):
            assert alt.code != result.target_code


def _make_mapper(
    provider: OpenAIProvider | OllamaProvider,
    *,
    target_ontology: str | None,
) -> OntologyMapper:
    search_tools = SearchTools(
        api_timeout=15,
        request_delay=0.1,
        loinc_username=os.environ.get("LOINC_USERNAME"),
        loinc_password=os.environ.get("LOINC_PASSWORD"),
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
        max_alternatives=MAX_ALTERNATIVES,
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
        "max_alternatives": MAX_ALTERNATIVES,
        "smoke_debug": SMOKE_DEBUG,
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

    info = extract_pipeline_metadata(result) or {}
    print_result_summary(result)
    print_alternatives_summary(
        result,
        max_alternatives=MAX_ALTERNATIVES,
        always=True,
    )
    print_trace_summary(result)
    print_full_debug_result(result, enabled=SMOKE_DEBUG)

    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    _validate_alternatives(result)

    if override_case:
        assert result.ontology in {"HPO", "HP", "UNKNOWN"}, (
            f"Expected HPO/HP/UNKNOWN in override case, got {result.ontology!r}"
        )
        assert result.ontology != "LOINC", "HPO override case must not return LOINC"
    elif target_ontology:
        expected_ontology = target_ontology.upper()
        assert result.ontology in {expected_ontology, "UNKNOWN"}, (
            f"Expected {expected_ontology!r} or UNKNOWN, got {result.ontology!r}"
        )
        if not _is_unmapped(result) and expected_ontology == "LOINC":
            term = result.target_term.lower()
            assert (
                result.target_code.startswith("LOINC:")
                or "8480-6" in result.target_code
                or ("systolic" in term and "blood pressure" in term)
            ), "Mapped public result does not look like a systolic BP LOINC result"
    else:
        # No target constraint: the pipeline picks the best-matching ontology freely.
        # Assert the result carries a non-empty ontology string — we don't pin the value.
        assert result.ontology, (
            f"result.ontology must be non-empty in unconstrained mode, got {result.ontology!r}"
        )

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
