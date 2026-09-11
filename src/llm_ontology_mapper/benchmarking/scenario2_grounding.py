"""
Scenario 2 (retrieval-mode ablation) grounding extraction (Part 19).

Determines whether a mapped Top-1 prediction's code actually came from the
candidate set supplied to the grounded reranker, using the REAL
candidate/provenance evidence MappingResultBuilder / DisabledMappingRunner
already attach to MappingResult.metadata.rag_debug.candidates_retrieved[0]
(see mapping_result_builder._build_metadata / disabled_mapping._build_metadata)
-- never inferred from retrieval_mode != "disabled" or logic_type == "rag".

For public/local grounded modes, candidates_retrieved[0]["candidate_score_
provenance"] is the exact list of NormalizedCandidate records that were
merged and handed to LLMReranker (mapping_result_builder._candidate_score_
provenance); the selected code is checked against that list by code identity.

For disabled mode, DisabledMappingRunner never populates candidate_score_
provenance (there is no retrieval), so selected_code_was_retrieved is
mechanically False and grounding_source is "none" -- derived from the
evidence, not hardcoded as a special case.

Pure logic -- no network. Called at execution time (the evidence only exists
on the live MappingResult; it is not re-derivable from predictions.csv alone,
so the extracted fields are persisted as their own columns, Part 20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_ontology_mapper.models import MappingResult
from llm_ontology_mapper.ontology_identity import normalize_code_for_ontology

GROUNDING_SOURCE_NONE = "none"


@dataclass(frozen=True)
class GroundingInfo:
    is_grounded: bool
    grounding_source: str
    retrieval_skipped: bool
    selected_code_was_retrieved: bool
    candidate_count: int


def _pipeline_info(result: MappingResult) -> dict[str, Any]:
    metadata = result.metadata
    rag_debug = metadata.rag_debug if metadata is not None else None
    candidates_retrieved = rag_debug.candidates_retrieved if rag_debug is not None else []
    if candidates_retrieved and isinstance(candidates_retrieved[0], dict):
        return candidates_retrieved[0]
    return {}


def _retrieval_skipped(info: dict[str, Any]) -> bool:
    # Grounded modes (MappingResultBuilder) nest retrieval_skipped inside
    # retrieval_trace; disabled mode (DisabledMappingRunner) sets it at the
    # top level of pipeline_info directly. Check both without assuming which
    # shape is present.
    if "retrieval_skipped" in info:
        return bool(info["retrieval_skipped"])
    retrieval_trace = info.get("retrieval_trace") or {}
    return bool(retrieval_trace.get("retrieval_skipped", False))


def _candidate_codes(info: dict[str, Any]) -> list[str]:
    provenance = info.get("candidate_score_provenance") or []
    codes: list[str] = []
    for entry in provenance:
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        ontology = entry.get("ontology")
        if code:
            codes.append(normalize_code_for_ontology(code, ontology))
    return codes


def extract_grounding_info(
    result: MappingResult,
    *,
    is_mapped: bool,
    mapped_code_normalized: str | None,
) -> GroundingInfo:
    """Extract grounding evidence for one successfully-executed (non-error)
    mapping call. Must be called with the live MappingResult -- the evidence
    it reads is not reconstructable from predictions.csv alone."""
    info = _pipeline_info(result)
    is_grounded = bool(info.get("is_grounded", False))
    grounding_source = str(info.get("grounding_source") or GROUNDING_SOURCE_NONE)
    retrieval_skipped = _retrieval_skipped(info)
    candidate_codes = _candidate_codes(info)
    candidate_count = int(info.get("candidate_count", len(candidate_codes)))

    if not is_mapped or not mapped_code_normalized:
        selected_code_was_retrieved = False
    else:
        selected_code_was_retrieved = mapped_code_normalized in candidate_codes

    return GroundingInfo(
        is_grounded=is_grounded,
        grounding_source=grounding_source,
        retrieval_skipped=retrieval_skipped,
        selected_code_was_retrieved=selected_code_was_retrieved,
        candidate_count=candidate_count,
    )


def grounding_rate(rows: list[GroundingInfo]) -> float | None:
    """mapped predictions whose selected code was retrieved / mapped predictions.
    None when there are no mapped predictions (never fabricated as 0)."""
    if not rows:
        return None
    return sum(1 for r in rows if r.selected_code_was_retrieved) / len(rows)
