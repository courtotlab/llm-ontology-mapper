"""Shared output helpers for planned live smoke scripts."""

from __future__ import annotations

import json
import os
from typing import Any


def env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback) == "1"


def extract_pipeline_metadata(result: object) -> dict[str, Any] | None:
    try:
        metadata = result.metadata
        rag_debug = metadata.rag_debug
        candidates = rag_debug.candidates_retrieved or []
        if candidates and isinstance(candidates[0], dict):
            return candidates[0]
    except Exception:
        return None
    return None


def print_result_summary(result: object) -> None:
    alternatives = getattr(result, "alternatives", [])
    print(json.dumps({
        "result_summary": {
            "source_term": getattr(result, "source_term", None),
            "source_label": getattr(result, "source_label", None),
            "target_code": getattr(result, "target_code", None),
            "target_term": getattr(result, "target_term", None),
            "ontology": getattr(result, "ontology", None),
            "confidence": getattr(result, "confidence", None),
            "logic_type": _string_value(getattr(result, "logic_type", None)),
            "notes": getattr(result, "notes", None),
            "alternative_count": len(alternatives),
        },
    }, indent=2, default=str))


def print_alternatives_summary(
    result: object,
    *,
    max_alternatives: int | None = None,
    always: bool = False,
) -> None:
    alternatives = getattr(result, "alternatives", [])
    if not alternatives and not always:
        return

    print(json.dumps({
        "alternatives": [
            {
                "code": alt.code,
                "term": alt.term,
                "ontology": alt.ontology,
                "confidence": alt.confidence,
                "explanation": getattr(alt, "explanation", None),
            }
            for alt in alternatives
        ],
        "alternative_count": len(alternatives),
        "max_alternatives": max_alternatives,
    }, indent=2, default=str))


def print_trace_summary(result: object) -> None:
    """Print a trace summary that keeps two deliberately different grounding
    concepts visually distinct instead of printing them under identical
    "is_grounded"/"grounding_source" labels:

    - result_is_grounded / result_grounding_source come from the reranker's
      final decision (MappingResultBuilder's top-level pipeline metadata,
      itself copied from RerankDecision -- see llm_reranker.py). This means
      "the final selected mapping came from a retrieved candidate." It is
      False whenever the reranker abstains (is_unmapped=True), even if
      retrieval itself succeeded and returned candidates.

    - retrieval_is_grounded / retrieval_grounding_source come from the
      nested RetrievalTrace (planned_pipeline.py's _build_trace). This means
      "retrieval was attempted for this mode and produced merged
      candidates," independent of whether the reranker went on to select
      one of them.

    It is valid and expected for these to disagree: retrieval can succeed
    (retrieval_is_grounded=True, retrieval_grounding_source=public_api) while
    the reranker still abstains (result_is_grounded=False,
    result_grounding_source=none), producing an UNKNOWN:UNMAPPED result. See
    RerankDecision.is_grounded and RetrievalTrace.is_grounded in models.py.
    """
    pipeline_meta = extract_pipeline_metadata(result)
    if not pipeline_meta:
        return

    retrieval_trace = pipeline_meta.get("retrieval_trace", {}) or {}
    query_plan = (
        retrieval_trace.get("query_plan", {})
        or pipeline_meta.get("query_plan", {})
        or {}
    )

    print(json.dumps({
        "trace_summary": {
            "retrieval_mode": pipeline_meta.get("retrieval_mode"),
            "result_is_grounded": pipeline_meta.get("is_grounded"),
            "result_grounding_source": pipeline_meta.get("grounding_source"),
            "retrieval_is_grounded": retrieval_trace.get("is_grounded"),
            "retrieval_grounding_source": retrieval_trace.get("grounding_source"),
            "policy": pipeline_meta.get("policy"),
            "retrieval_skipped": pipeline_meta.get("retrieval_skipped"),
            "retrieval_disabled_reason": pipeline_meta.get(
                "retrieval_disabled_reason"
            ),
            "candidate_count": pipeline_meta.get("candidate_count"),
            "normalized_term": query_plan.get("normalized_term"),
            "semantic_type": query_plan.get("semantic_type"),
            "expanded_queries": query_plan.get("expanded_queries"),
            "candidate_ontologies": query_plan.get("candidate_ontologies"),
            "preferred_ontology": query_plan.get("preferred_ontology"),
            "target_ontology_constraint": query_plan.get(
                "target_ontology_constraint"
            ),
            "route_call_count": len(retrieval_trace.get("route_calls", []) or []),
            "raw_candidate_count": retrieval_trace.get("raw_candidate_count"),
            "merged_candidate_count": retrieval_trace.get("merged_candidate_count"),
            "selected_candidate_code": retrieval_trace.get("selected_candidate_code"),
            "errors": retrieval_trace.get("errors", []),
        },
    }, indent=2, default=str))


def print_full_debug_result(result: object, *, enabled: bool) -> None:
    if not enabled:
        return

    print("--- FULL DEBUG RESULT ---")
    if hasattr(result, "model_dump_json"):
        print(result.model_dump_json(indent=2))  # type: ignore[attr-defined]
        return
    if hasattr(result, "model_dump"):
        print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))  # type: ignore[attr-defined]
        return
    print(json.dumps(result, indent=2, default=str))


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))
