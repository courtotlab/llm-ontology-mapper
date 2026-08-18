"""Shared target-ontology eligibility rules for grounded candidates."""

from __future__ import annotations

from collections.abc import Iterable

from llm_ontology_mapper.models import NormalizedCandidate
from llm_ontology_mapper.ontology_identity import canonical_ontology


def candidate_allowed_for_targets(
    candidate: NormalizedCandidate,
    target_ontologies: Iterable[str] | None,
    *,
    strict_target_ontology: bool = False,
) -> bool:
    """Return whether a candidate satisfies the active target ontology constraint.

    Native ontology matches retain the historical hard-filter behavior. EFO has
    one additional search-space rule: candidates retrieved from an EFO-scoped
    route are eligible when EFO is among the requested targets, without
    relabeling the candidate's native ontology.

    strict_target_ontology=True disables that EFO provenance exception: only a
    candidate whose canonical native ontology is itself a member of the target
    set is eligible, regardless of retrieval provenance.
    """
    targets = _canonical_ontology_set(target_ontologies)
    if targets is None:
        return True

    native = canonical_ontology(candidate.ontology) or candidate.ontology.upper().strip()
    if native in targets:
        return True

    if strict_target_ontology:
        return False

    return "EFO" in targets and "EFO" in _canonical_ontology_set(
        candidate.retrieved_from_ontologies
    )


def _canonical_ontology_set(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    result = {
        canonical_ontology(value) or str(value or "").upper().strip()
        for value in values
        if str(value or "").strip()
    }
    return result or set()
