"""Manual live smoke script for planned public multi-ontology mapping.

This is a direct runnable script, not a pytest test. It intentionally uses
OntologyMapper(use_planned_pipeline=True), not AgenticMapper.

Run with:
    uv run python tests/live/planned_public_multi_ontology_smoke.py

Custom allow-list:
    SMOKE_ALLOWED_ONTOLOGIES=LOINC,HPO,MONDO \
    uv run python tests/live/planned_public_multi_ontology_smoke.py

Unrestricted planner mode:
    SMOKE_NO_ONTOLOGY_FILTER=1 \
    uv run python tests/live/planned_public_multi_ontology_smoke.py
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
OPENAI_MODEL = "gpt-5.5"
OLLAMA_MODEL = "gpt-oss:120b"
OLLAMA_BASE_URL = "http://localhost:11528"

RETRIEVAL_MODE = "public"
DEFAULT_ALLOWED_ONTOLOGIES = ["LOINC", "HPO", "MONDO"]

DEFAULT_CASES: list[dict[str, str]] = [
    # Expected LOINC measurement mappings
    {
        "source_term": "sys_bp",
        "source_label": "Systolic blood pressure",
        "clinical_area": "measurement",
    },
    {
        "source_term": "heart_rate",
        "source_label": "Heart rate",
        "clinical_area": "measurement",
    },
    {
        "source_term": "body_temperature",
        "source_label": "Body temperature",
        "clinical_area": "measurement",
    },

    # Expected HPO phenotype mappings
    {
        "source_term": "cough",
        "source_label": "Cough",
        "clinical_area": "phenotype",
    },
    {
        "source_term": "fever",
        "source_label": "Fever",
        "clinical_area": "phenotype",
    },
    {
        "source_term": "hearing_loss",
        "source_label": "Hearing loss",
        "clinical_area": "phenotype",
    },

    # Expected MONDO disease/diagnosis mappings
    {
        "source_term": "cystic_fibrosis",
        "source_label": "Cystic Fibrosis",
        "clinical_area": "diagnosis",
    },
    {
        "source_term": "type_2_diabetes",
        "source_label": "Type 2 diabetes mellitus",
        "clinical_area": "diagnosis",
    },
    {
        "source_term": "asthma",
        "source_label": "Asthma",
        "clinical_area": "diagnosis",
    },

    # Negative control:
    # The most appropriate ontology would normally be RxNorm,
    # which is not in DEFAULT_ALLOWED_ONTOLOGIES.
    {
        "source_term": "metformin",
        "source_label": "Metformin",
        "clinical_area": "medication",
    },
]

MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "6"))
MAX_ALTERNATIVES = int(os.environ.get("MAX_ALTERNATIVES", "5"))
SMOKE_DEBUG = env_flag("SMOKE_DEBUG")
SMOKE_NO_ONTOLOGY_FILTER = env_flag("SMOKE_NO_ONTOLOGY_FILTER")


def _optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def parse_allowed_ontologies(raw: str | None) -> list[str] | None:
    """Parse a comma-separated ontology list, preserving first-seen order."""
    if raw is None:
        return list(DEFAULT_ALLOWED_ONTOLOGIES)

    ontologies: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        ontology = item.strip().upper()
        if ontology and ontology not in seen:
            ontologies.append(ontology)
            seen.add(ontology)
    return ontologies or None


def resolve_allowed_ontologies(
    *,
    no_filter: bool,
    raw_allowed: str | None,
) -> list[str] | None:
    if no_filter:
        return None
    return parse_allowed_ontologies(raw_allowed)


def hard_filter_active(allowed_target_ontologies: list[str] | None) -> bool:
    return allowed_target_ontologies is not None


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


def _make_mapper(
    provider: OpenAIProvider | OllamaProvider,
    *,
    allowed_target_ontologies: list[str] | None,
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
        ontologies=allowed_target_ontologies,
        use_planned_pipeline=True,
        retrieval_mode=RETRIEVAL_MODE,
        planned_pipeline=pipeline,
        rag_top_k=MAX_RESULTS_PER_QUERY,
        max_alternatives=MAX_ALTERNATIVES,
    )


def validate_scope(
    *,
    result: object,
    allowed_target_ontologies: list[str] | None,
    pipeline_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return scope validation details and violations for live smoke assertions."""
    active = hard_filter_active(allowed_target_ontologies)
    allowed = {
        ontology.upper().strip()
        for ontology in (allowed_target_ontologies or [])
        if ontology.strip()
    }
    primary_ontology = str(getattr(result, "ontology", "")).upper().strip()
    alternatives = list(getattr(result, "alternatives", []) or [])
    alternative_ontologies = [
        str(getattr(alt, "ontology", "")).upper().strip() for alt in alternatives
    ]
    searched_ontologies = extract_searched_ontologies(pipeline_metadata or {})

    violations: list[str] = []
    if active:
        if primary_ontology not in allowed and primary_ontology != "UNKNOWN":
            violations.append(
                f"primary ontology {primary_ontology!r} is outside "
                f"allowed_target_ontologies={sorted(allowed)!r}"
            )
        for ontology in alternative_ontologies:
            if ontology not in allowed:
                violations.append(
                    f"alternative ontology {ontology!r} is outside "
                    f"allowed_target_ontologies={sorted(allowed)!r}"
                )
        extra_searched = sorted(searched_ontologies - allowed)
        if extra_searched:
            violations.append(
                f"searched ontologies {extra_searched!r} are outside "
                f"allowed_target_ontologies={sorted(allowed)!r}"
            )

    return {
        "hard_filter_active": active,
        "allowed_target_ontologies": list(allowed_target_ontologies)
        if allowed_target_ontologies is not None
        else None,
        "primary_ontology": primary_ontology,
        "primary_valid": (not active)
        or primary_ontology in allowed
        or primary_ontology == "UNKNOWN",
        "alternative_ontologies": alternative_ontologies,
        "alternatives_valid": (not active)
        or all(ontology in allowed for ontology in alternative_ontologies),
        "searched_ontologies": sorted(searched_ontologies),
        "searched_ontologies_exposed": bool(searched_ontologies),
        "searched_ontologies_valid": (not active)
        or not searched_ontologies
        or searched_ontologies.issubset(allowed),
        "violations": violations,
    }


def extract_searched_ontologies(pipeline_metadata: dict[str, Any]) -> set[str]:
    """Best-effort extraction of ontologies exposed in retrieval debug metadata."""
    retrieval_trace = pipeline_metadata.get("retrieval_trace", {}) or {}
    route_calls = retrieval_trace.get("route_calls", []) or []
    searched: set[str] = set()

    for route_call in route_calls:
        if not isinstance(route_call, dict):
            continue
        _add_ontology(searched, route_call.get("target_ontology"))
        for ontology in route_call.get("candidate_ontologies") or []:
            _add_ontology(searched, ontology)

    for item in pipeline_metadata.get("candidate_score_provenance", []) or []:
        if isinstance(item, dict):
            _add_ontology(searched, item.get("ontology"))

    return searched


def _add_ontology(target: set[str], value: object) -> None:
    if value is None:
        return
    text = str(value).upper().strip()
    if text:
        target.add(text)


def _planner_retrieval_debug(info: dict[str, Any]) -> dict[str, Any]:
    retrieval_trace = info.get("retrieval_trace", {}) or {}
    query_plan = retrieval_trace.get("query_plan", {}) or info.get("query_plan", {}) or {}
    return {
        "query_plan": query_plan,
        "retrieval_mode": info.get("retrieval_mode"),
        "grounding_source": info.get("grounding_source"),
        "is_grounded": info.get("is_grounded"),
        "candidate_count": info.get("candidate_count"),
        "raw_candidate_count": retrieval_trace.get("raw_candidate_count"),
        "merged_candidate_count": retrieval_trace.get("merged_candidate_count"),
        "route_calls": retrieval_trace.get("route_calls", []),
        "errors": retrieval_trace.get("errors", []),
    }


def _print_context(
    *,
    provider_name: str,
    model: str,
    allowed_target_ontologies: list[str] | None,
) -> None:
    context: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
        "retrieval_mode": RETRIEVAL_MODE,
        "allowed_target_ontologies": allowed_target_ontologies,
        "hard_filter_active": hard_filter_active(allowed_target_ontologies),
        "cases": DEFAULT_CASES,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
        "max_alternatives": MAX_ALTERNATIVES,
        "smoke_debug": SMOKE_DEBUG,
    }
    if provider_name == "ollama":
        context["ollama_base_url"] = OLLAMA_BASE_URL
    print(json.dumps(context, indent=2))


def _run_case(
    *,
    mapper: OntologyMapper,
    case: dict[str, str],
    allowed_target_ontologies: list[str] | None,
) -> None:
    print(json.dumps({"input": case}, indent=2))

    result = mapper.map_term(
        source_term=case["source_term"],
        source_label=_optional(case["source_label"]),
        entity_type=_optional(case["clinical_area"]),
    )

    info = extract_pipeline_metadata(result) or {}
    validation = validate_scope(
        result=result,
        allowed_target_ontologies=allowed_target_ontologies,
        pipeline_metadata=info,
    )

    print_result_summary(result)
    print_alternatives_summary(
        result,
        max_alternatives=MAX_ALTERNATIVES,
        always=True,
    )
    print_trace_summary(result)
    print(json.dumps({"scope_validation": validation}, indent=2, default=str))
    print(
        json.dumps(
            {"planner_retrieval_debug": _planner_retrieval_debug(info)},
            indent=2,
            default=str,
        )
    )
    print_full_debug_result(result, enabled=SMOKE_DEBUG)

    assert 0.0 <= result.confidence <= 1.0, "confidence must be in [0, 1]"
    if validation["violations"]:
        raise AssertionError("; ".join(validation["violations"]))

    if info:
        assert info.get("retrieval_mode") == "public", info
        if not _is_unmapped(result) and "is_grounded" in info:
            assert info["is_grounded"] is True, info


def main() -> None:
    provider = _build_provider()
    if provider is None:
        sys.exit(0)

    allowed_target_ontologies = resolve_allowed_ontologies(
        no_filter=SMOKE_NO_ONTOLOGY_FILTER,
        raw_allowed=os.environ.get("SMOKE_ALLOWED_ONTOLOGIES"),
    )

    provider_name = PROVIDER.strip().lower()
    model = OPENAI_MODEL if provider_name == "openai" else OLLAMA_MODEL
    _print_context(
        provider_name=provider_name,
        model=model,
        allowed_target_ontologies=allowed_target_ontologies,
    )
    mapper = _make_mapper(
        provider,
        allowed_target_ontologies=allowed_target_ontologies,
    )

    for case in DEFAULT_CASES:
        print("\n--- Running multi-ontology planned public case ---")
        _run_case(
            mapper=mapper,
            case=case,
            allowed_target_ontologies=allowed_target_ontologies,
        )

    print("PASS: planned public multi-ontology smoke completed")


if __name__ == "__main__":
    main()
