"""
Unit tests for the shared target-eligibility helper (target_eligibility.py).

No external APIs are called — all inputs are NormalizedCandidate objects
constructed directly from field values.

Run with:  pytest tests/test_target_eligibility.py -v -m unit
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.models import NormalizedCandidate, RetrievalMode
from llm_ontology_mapper.target_eligibility import candidate_allowed_for_targets

# ─────────────────────────────────────────────────────────────────────────────
# Shared factory helper
# ─────────────────────────────────────────────────────────────────────────────


def _c(
    code: str = "HP:0012735",
    term: str = "Cough",
    ontology: str = "HPO",
    source: str = "OLS",
    matched_query: str = "cough",
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    retrieved_from_ontologies: list[str] | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        code=code,
        term=term,
        ontology=ontology,
        source=source,
        matched_query=matched_query,
        retrieval_mode=retrieval_mode,
        retrieved_from_ontologies=retrieved_from_ontologies or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lenient (default) behaviour — regression coverage
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_lenient_default_allows_imported_hpo_candidate_retrieved_from_efo() -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", retrieved_from_ontologies=["EFO"])

    assert candidate_allowed_for_targets(candidate, ["EFO"]) is True


@pytest.mark.unit
def test_lenient_explicit_false_allows_imported_hpo_candidate_retrieved_from_efo() -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", retrieved_from_ontologies=["EFO"])

    assert (
        candidate_allowed_for_targets(candidate, ["EFO"], strict_target_ontology=False) is True
    )


@pytest.mark.unit
def test_lenient_rejects_candidate_not_retrieved_from_efo() -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", retrieved_from_ontologies=["HPO"])

    assert candidate_allowed_for_targets(candidate, ["EFO"]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_strict_rejects_imported_hpo_candidate() -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", retrieved_from_ontologies=["EFO"])

    assert (
        candidate_allowed_for_targets(candidate, ["EFO"], strict_target_ontology=True) is False
    )


@pytest.mark.unit
def test_strict_rejects_imported_mondo_candidate() -> None:
    candidate = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["EFO"],
    )

    assert (
        candidate_allowed_for_targets(candidate, ["EFO"], strict_target_ontology=True) is False
    )


@pytest.mark.unit
def test_strict_allows_native_efo_candidate() -> None:
    candidate = _c(code="EFO:0000408", ontology="EFO", term="disease")

    assert candidate_allowed_for_targets(candidate, ["EFO"], strict_target_ontology=True) is True


@pytest.mark.unit
def test_strict_non_efo_native_target_match_is_unaffected() -> None:
    """Strict mode is a generic native-only rule — it must not change eligibility
    for ontologies that never had a provenance carve-out (proves the flag is not
    an accidental EFO-only special case)."""
    candidate = _c(code="HP:0012735", ontology="HPO")

    assert candidate_allowed_for_targets(candidate, ["HPO"], strict_target_ontology=True) is True


@pytest.mark.unit
def test_strict_rejects_native_mismatch_even_with_matching_provenance() -> None:
    """A fabricated non-EFO 'carve-out-shaped' provenance value must not grant
    eligibility under strict mode; only the native-ontology branch may fire."""
    candidate = _c(
        code="HP:0012735",
        ontology="HPO",
        retrieved_from_ontologies=["LOINC"],
    )

    assert (
        candidate_allowed_for_targets(candidate, ["LOINC"], strict_target_ontology=True) is False
    )


@pytest.mark.unit
def test_strict_ignores_multi_route_provenance_including_efo() -> None:
    candidate = _c(
        code="HP:0002099",
        ontology="HPO",
        retrieved_from_ontologies=["EFO", "HPO"],
    )

    assert (
        candidate_allowed_for_targets(candidate, ["EFO"], strict_target_ontology=True) is False
    )


# ─────────────────────────────────────────────────────────────────────────────
# No targets — unrestricted regardless of strictness
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_targets_is_unrestricted_under_strict_mode() -> None:
    candidate = _c(code="HP:0012735", ontology="HPO")

    assert candidate_allowed_for_targets(candidate, None, strict_target_ontology=True) is True


@pytest.mark.unit
def test_no_targets_is_unrestricted_under_lenient_mode() -> None:
    candidate = _c(code="HP:0012735", ontology="HPO")

    assert candidate_allowed_for_targets(candidate, None) is True


# ─────────────────────────────────────────────────────────────────────────────
# Multi-target
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_strict_multi_target_allows_either_native_member() -> None:
    native_efo = _c(code="EFO:0000408", ontology="EFO")
    native_hpo = _c(code="HP:0012735", ontology="HPO")

    assert (
        candidate_allowed_for_targets(native_efo, ["EFO", "HPO"], strict_target_ontology=True)
        is True
    )
    assert (
        candidate_allowed_for_targets(native_hpo, ["EFO", "HPO"], strict_target_ontology=True)
        is True
    )


@pytest.mark.unit
def test_strict_multi_target_rejects_imported_mondo_retrieved_through_efo() -> None:
    imported_mondo = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["EFO"],
    )

    assert (
        candidate_allowed_for_targets(
            imported_mondo, ["EFO", "HPO"], strict_target_ontology=True
        )
        is False
    )


@pytest.mark.unit
def test_lenient_multi_target_still_allows_imported_mondo_retrieved_through_efo() -> None:
    imported_mondo = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["EFO"],
    )

    assert candidate_allowed_for_targets(imported_mondo, ["EFO", "HPO"]) is True
