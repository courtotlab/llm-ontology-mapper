"""Manual live smoke script for planned local SapBERT ontology mapping.

This is a direct runnable script, not a pytest test. It intentionally uses
OntologyMapper(use_planned_pipeline=True), not AgenticMapper.

Run with:
    uv run python tests/live/planned_local_smoke.py

TARGET_ONTOLOGY accepts one ontology or a comma-separated allow-list, e.g.
    TARGET_ONTOLOGY = "LOINC,HPO,MONDO"
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

from llm_ontology_mapper.local_retriever import LocalRetrievalError, LocalSemanticRetriever
from llm_ontology_mapper.mapper import OntologyMapper
from llm_ontology_mapper.planned_pipeline import PlannedPipeline, PlannedPipelineError
from llm_ontology_mapper.providers import OllamaProvider, OpenAIProvider

__test__ = False


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================

PROVIDER = "openai"  # "openai" or "ollama"
OPENAI_MODEL = "gpt-5.6-luna"
OLLAMA_MODEL = "gpt-oss:120b"
OLLAMA_BASE_URL = "http://localhost:11528"

SAPBERT_URL = "http://localhost:8765"

SOURCE_TERM = "sinus_pain"
SOURCE_LABEL = "sinus pain/congestion"
SOURCE_DESCRIPTION = ""
SOURCE_TYPE = ""
CLINICAL_AREA = ""
TARGET_ONTOLOGY = "HPO"
RETRIEVAL_MODE = "local"

MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "10"))
MAX_ALTERNATIVES = int(os.environ.get("MAX_ALTERNATIVES", "5"))
SMOKE_DEBUG = env_flag("SMOKE_DEBUG")


def _optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _target_ontologies(value: str | None) -> list[str] | None:
    if value is None:
        return None
    ontologies: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        ontology = item.strip().upper()
        if ontology and ontology not in seen:
            ontologies.append(ontology)
            seen.add(ontology)
    return ontologies or None


def _allowed_ontology_set(target_ontologies: list[str] | None) -> set[str] | None:
    if target_ontologies is None:
        return None
    allowed = {ontology.upper().strip() for ontology in target_ontologies if ontology.strip()}
    return allowed or None


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


# All code prefixes that local retrieval routes can legitimately emit, derived from
# search_tools._normalize_code prefix_map and OLS_ONTOLOGY_MAP.
_KNOWN_CODE_PREFIXES: tuple[str, ...] = (
    "CHEBI:",
    "DOID:",
    "EFO:",
    "GO:",
    "HP:",
    "ICD10:",
    "LOINC:",
    "MESH:",
    "MONDO:",
    "NCIT:",
    "RXNORM:",
    "SNOMED-CT:",
    "UBERON:",
    "UO:",
)


def _is_unmapped(result: object) -> bool:
    code = str(getattr(result, "target_code", "")).upper()
    ontology = str(getattr(result, "ontology", "")).upper()
    return code in {"UNMAPPED", "UNKNOWN:UNMAPPED"} or ontology == "UNKNOWN"


def _validate_alternatives(
    result: object,
    *,
    target_ontologies: list[str] | None,
) -> None:
    alternatives = getattr(result, "alternatives", [])
    assert len(alternatives) <= MAX_ALTERNATIVES
    allowed = _allowed_ontology_set(target_ontologies)

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
        if allowed is not None:
            alt_ontology = str(alt.ontology).upper().strip()
            if "EFO" not in allowed:
                assert alt_ontology in allowed, (
                    f"Alternative ontology {alt_ontology!r} is outside "
                    f"TARGET_ONTOLOGY allow-list {sorted(allowed)!r}"
                )


def _make_mapper(
    provider: OpenAIProvider | OllamaProvider,
    *,
    target_ontologies: list[str] | None,
) -> OntologyMapper:
    pipeline = PlannedPipeline(
        provider=provider,
        local_retriever=LocalSemanticRetriever(sapbert_url=SAPBERT_URL),
    )
    return OntologyMapper(
        llm_provider=provider,
        ontologies=target_ontologies,
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
    target_ontologies: list[str] | None,
) -> None:
    context: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
        "source_term": SOURCE_TERM,
        "source_label": _optional(SOURCE_LABEL),
        "source_description": _optional(SOURCE_DESCRIPTION),
        "source_type": _optional(SOURCE_TYPE),
        "target_ontology": ",".join(target_ontologies) if target_ontologies else None,
        "target_ontologies": target_ontologies,
        "hard_filter_active": target_ontologies is not None,
        "clinical_area": _optional(CLINICAL_AREA),
        "retrieval_mode": RETRIEVAL_MODE,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
        "max_alternatives": MAX_ALTERNATIVES,
        "sapbert_url": SAPBERT_URL,
        "smoke_debug": SMOKE_DEBUG,
    }
    if provider_name == "ollama":
        context["ollama_base_url"] = OLLAMA_BASE_URL
    print(json.dumps(context, indent=2))


def _run_case(
    provider: OpenAIProvider | OllamaProvider,
    *,
    target_ontologies: list[str] | None,
) -> None:
    provider_name = PROVIDER.strip().lower()
    model = OPENAI_MODEL if provider_name == "openai" else OLLAMA_MODEL
    _print_context(
        provider_name=provider_name,
        model=model,
        target_ontologies=target_ontologies,
    )

    mapper = _make_mapper(provider, target_ontologies=target_ontologies)
    try:
        result = mapper.map_term(
            source_term=SOURCE_TERM,
            source_label=_optional(SOURCE_LABEL),
            source_type=_optional(SOURCE_TYPE),
            entity_type=_optional(CLINICAL_AREA),
            source_description=_optional(SOURCE_DESCRIPTION),
        )
    except PlannedPipelineError as exc:
        cause = exc.__cause__
        if isinstance(cause, LocalRetrievalError):
            raise RuntimeError(
                f"Local SapBERT retrieval failed for SAPBERT_URL={SAPBERT_URL!r}. "
                "Confirm the local service is running and implements POST /search."
            ) from exc
        raise

    info = extract_pipeline_metadata(result) or {}
    print_result_summary(result)
    print_alternatives_summary(
        result,
        max_alternatives=MAX_ALTERNATIVES,
        always=True,
    )
    print_trace_summary(result)
    print_full_debug_result(result, enabled=SMOKE_DEBUG)
    query_plan = _extract_actual_query_plan(info)
    if SMOKE_DEBUG:
        assert query_plan, "Expected actual QueryPlan debug metadata"
        assert query_plan.get("source_description") == _optional(SOURCE_DESCRIPTION), (
            "Expected source_description on the actual QueryPlan debug metadata"
        )
        assert query_plan.get("source_type") == _optional(SOURCE_TYPE), (
            "Expected source_type on the actual QueryPlan debug metadata"
        )
        assert query_plan.get("expanded_queries"), (
            "Expected non-empty expanded_queries on the actual QueryPlan debug metadata"
        )

    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    _validate_alternatives(result, target_ontologies=target_ontologies)
    assert not any(
        alternative.confidence > result.confidence for alternative in result.alternatives
    ), "alternative confidence must not exceed selected result confidence"

    if target_ontologies:
        allowed = _allowed_ontology_set(target_ontologies) or set()
        result_ontology = result.ontology.upper()

        # For normal target ontologies, the final native ontology must still
        # match the hard target exactly.
        #
        # EFO is special: an EFO-scoped search may legitimately select an
        # imported term whose native identifier belongs to MONDO/HPO/etc.
        if "EFO" not in allowed:
            assert result_ontology in {*allowed, "UNKNOWN"}, (
                f"Expected one of {sorted(allowed)!r} or UNKNOWN, got {result.ontology!r}"
            )
        else:
            assert result_ontology, "EFO result must carry a non-empty native ontology"

        if not _is_unmapped(result) and result_ontology == "LOINC":
            assert result.target_code.startswith("LOINC:"), (
                f"LOINC result should use a LOINC code prefix, got {result.target_code!r}"
            )
    else:
        # No target constraint: the pipeline picks the best-matching ontology freely.
        # Assert the result carries a non-empty ontology string — we don't pin the value.
        assert result.ontology, (
            f"result.ontology must be non-empty in unconstrained mode, got {result.ontology!r}"
        )

    if info:
        assert info.get("retrieval_mode") == "local", info
        if "grounding_source" in info:
            assert info["grounding_source"] in {"local_sapbert", "none"}, info
        if not _is_unmapped(result) and "is_grounded" in info:
            assert info["is_grounded"] is True, info
        _validate_local_retrieval_debug(
            info,
            result,
            target_ontologies=target_ontologies,
        )


def _extract_actual_query_plan(info: dict[str, Any]) -> dict[str, Any]:
    retrieval_trace = info.get("retrieval_trace", {}) or {}
    query_plan = retrieval_trace.get("query_plan", {}) or {}
    if isinstance(query_plan, dict):
        return query_plan
    return {}


def _validate_local_retrieval_debug(
    info: dict[str, Any],
    result: object,
    *,
    target_ontologies: list[str] | None,
) -> None:
    retrieval_trace = info.get("retrieval_trace", {}) or {}
    query_plan = retrieval_trace.get("query_plan", {}) or {}
    target_constraint = (
        query_plan.get("target_ontology_constraint")
        if isinstance(query_plan, dict)
        else None
    )
    route_calls = retrieval_trace.get("route_calls", []) or []
    for route_call in route_calls:
        if not isinstance(route_call, dict):
            continue
        assert route_call.get("route") == "local_sapbert", route_call
        assert route_call.get("query"), route_call
        if target_constraint:
            assert route_call.get("target_ontology") == target_constraint, route_call
        if target_ontologies is not None:
            allowed = _allowed_ontology_set(target_ontologies) or set()
            searched = {
                str(ontology).upper().strip()
                for ontology in route_call.get("candidate_ontologies", []) or []
                if str(ontology).strip()
            }
            assert searched.issubset(allowed), route_call

    raw_count = retrieval_trace.get("raw_candidate_count") or 0
    merged_count = retrieval_trace.get("merged_candidate_count") or 0
    if raw_count or merged_count:
        assert route_calls, "Expected local_sapbert route calls for retrieved candidates"

    if not _is_unmapped(result):
        assert info.get("grounding_source") == "local_sapbert", info
        provenance_items = info.get("candidate_score_provenance", []) or []
        assert provenance_items, "Expected candidate score provenance for grounded local result"
        for item in provenance_items:
            if not isinstance(item, dict):
                continue
            assert item.get("retrieval_mode") == "local", item
            assert item.get("source") in {"SapBERT", "local_sapbert"}, item


def main() -> None:
    if not _optional(SAPBERT_URL):
        print("SKIP: SAPBERT_URL is blank. Edit SAPBERT_URL at the top of this file.")
        sys.exit(0)

    provider = _build_provider()
    if provider is None:
        sys.exit(0)

    _run_case(provider, target_ontologies=_target_ontologies(TARGET_ONTOLOGY))
    print("PASS: planned local smoke completed")


if __name__ == "__main__":
    main()
