"""
CandidateMerger — deduplicates and ranks NormalizedCandidate objects.

Pure and side-effect free.  No external services are called.

Phase 4B of the planned grounded mapping pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from llm_ontology_mapper.models import NormalizedCandidate
from llm_ontology_mapper.ontology_identity import canonical_ontology, validate_candidate_identity
from llm_ontology_mapper.target_eligibility import candidate_allowed_for_targets


class CandidateMergeError(Exception):
    """Raised when merger input is invalid."""


class CandidateMerger:
    """
    Deduplicate and rank NormalizedCandidate objects.

    The merger does not retrieve candidates.  It only merges candidate lists
    already produced by CandidateNormalizer.

    Deduplication key: (ontology, code.upper())  —  ontology is already
    uppercase from the NormalizedCandidate validator; code is case-folded
    for comparison only.

    Duplicate selection (descending priority):
      1. Higher normalized_score.
      2. Higher raw_score when normalized_score is equal or absent.
      3. Candidate with a definition when scores tie.
      4. Earlier input order when all else ties.

    Provenance from all duplicates is preserved in merged_candidates_provenance;
    all matched_query values are collected into matched_queries.

    Sorting order (descending priority):
      1. normalized_score descending (None treated as lower than any score).
      2. raw_score descending (None treated as lower than any score).
      3. term alphabetically as a stable tiebreaker.

    Usage::

        merger = CandidateMerger()
        merged = merger.merge(
            candidates,
            target_ontology_constraint="HPO",
            max_candidates=20,
        )
    """

    def merge(
        self,
        candidates: Iterable[NormalizedCandidate],
        *,
        target_ontology_constraint: str | None = None,
        allowed_target_ontologies: list[str] | None = None,
        max_candidates: int | None = None,
        strict_target_ontology: bool = False,
    ) -> list[NormalizedCandidate]:
        """
        Deduplicate, filter, sort, and optionally limit a candidate list.

        Steps (in order):
          1. Validate that every item is a NormalizedCandidate.
          2. Deduplicate by canonical (ontology, code.upper()) key.
          3. Apply target_ontology_constraint / allowed_target_ontologies as hard filters.
          4. Sort by score descending, then alphabetically.
          5. Apply max_candidates limit.

        Args:
            candidates:                  Iterable of NormalizedCandidate objects.
            target_ontology_constraint:  Hard ontology filter; case-insensitive.
            allowed_target_ontologies:   Hard ontology allow-list; None means unrestricted.
            max_candidates:              Maximum number to return (applied last).
            strict_target_ontology:      When True, require canonical native-ontology
                                          membership only; disables the EFO
                                          imported-term provenance exception.

        Returns:
            A sorted, deduplicated, optionally filtered list of NormalizedCandidate.

        Raises:
            CandidateMergeError: If any item is not a NormalizedCandidate.
        """
        candidate_list = list(candidates)
        for idx, item in enumerate(candidate_list):
            if not isinstance(item, NormalizedCandidate):
                raise CandidateMergeError(
                    f"Expected NormalizedCandidate at index {idx}, "
                    f"got {type(item).__name__}: {item!r}"
                )

        valid_candidates = _filter_consistent_candidates(candidate_list)
        deduped = self._deduplicate(valid_candidates)

        if target_ontology_constraint:
            deduped = [
                c
                for c in deduped
                if candidate_allowed_for_targets(
                    c,
                    [target_ontology_constraint],
                    strict_target_ontology=strict_target_ontology,
                )
            ]
        elif allowed_target_ontologies is not None:
            allowed = _normalize_ontology_set(allowed_target_ontologies)
            if allowed:
                deduped = [
                    c
                    for c in deduped
                    if candidate_allowed_for_targets(
                        c, allowed, strict_target_ontology=strict_target_ontology
                    )
                ]

        result = sorted(deduped, key=_sort_key)

        if max_candidates is not None:
            result = result[:max_candidates]

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _dedup_key(candidate: NormalizedCandidate) -> tuple[str, str]:
        return (candidate.ontology, candidate.code.upper())

    def _deduplicate(self, candidates: list[NormalizedCandidate]) -> list[NormalizedCandidate]:
        """Group by canonical key; resolve each group to one candidate."""
        groups: dict[tuple[str, str], list[NormalizedCandidate]] = {}
        for candidate in candidates:
            key = self._dedup_key(candidate)
            groups.setdefault(key, []).append(candidate)

        result: list[NormalizedCandidate] = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                result.append(_merge_group(group))

        return result


# ── Module-level helpers (no access to instance state needed) ─────────────────


def _sort_key(c: NormalizedCandidate) -> tuple:
    """Return a sort key that produces descending score, then alphabetical term."""
    ns = c.normalized_score
    rs = c.raw_score
    return (
        0 if ns is not None else 1,  # candidates with a score come first
        -(ns if ns is not None else 0.0),  # higher ns → more negative → earlier
        0 if rs is not None else 1,
        -(rs if rs is not None else 0.0),
        c.term,  # alphabetical tiebreaker
    )


def _normalize_ontology_set(ontologies: list[str]) -> set[str]:
    return {
        canonical_ontology(ontology) or str(ontology or "").upper().strip()
        for ontology in ontologies
        if str(ontology or "").strip()
    }


def _same_ontology(left: str, right: str) -> bool:
    left_key = canonical_ontology(left) or str(left or "").upper().strip()
    right_key = canonical_ontology(right) or str(right or "").upper().strip()
    return left_key == right_key


def _ontology_in_set(ontology: str, allowed: set[str]) -> bool:
    key = canonical_ontology(ontology) or str(ontology or "").upper().strip()
    return key in allowed


def _filter_consistent_candidates(
    candidates: list[NormalizedCandidate],
) -> list[NormalizedCandidate]:
    result: list[NormalizedCandidate] = []
    for candidate in candidates:
        validation = validate_candidate_identity(
            ontology=candidate.ontology,
            code=candidate.code,
        )
        if validation.valid:
            result.append(candidate)
    return result


def _winner_key(c: NormalizedCandidate) -> tuple:
    """Return a key for selecting the best duplicate (lower = better)."""
    ns = c.normalized_score
    rs = c.raw_score
    has_def = 1 if c.definition else 0
    return (
        0 if ns is not None else 1,
        -(ns if ns is not None else 0.0),
        0 if rs is not None else 1,
        -(rs if rs is not None else 0.0),
        -has_def,  # has definition → -1 → wins on tie
    )
    # Python's min() returns the first item with the minimum key on ties,
    # which preserves earlier-input-order as the final tiebreaker.


def _merge_group(group: list[NormalizedCandidate]) -> NormalizedCandidate:
    """
    Resolve a group of duplicate candidates to a single merged candidate.

    The winner's top-level fields are kept.  Provenance is replaced with a
    merged record that audits all duplicates and preserves all matched queries.
    """
    winner = min(group, key=_winner_key)

    all_matched_queries: list[str] = list(dict.fromkeys(c.matched_query for c in group))
    retrieved_from_ontologies: list[str] = list(
        dict.fromkeys(
            ontology
            for candidate in group
            for ontology in candidate.retrieved_from_ontologies
        )
    )
    merged_provenance: dict[str, Any] = {
        "merged": True,
        "duplicate_count": len(group),
        "matched_queries": all_matched_queries,
        "sources": [
            {
                "matched_query": c.matched_query,
                "provenance": c.provenance,
            }
            for c in group
        ],
    }

    return NormalizedCandidate(
        code=winner.code,
        term=winner.term,
        ontology=winner.ontology,
        definition=winner.definition,
        source=winner.source,
        matched_query=winner.matched_query,
        retrieval_mode=winner.retrieval_mode,
        raw_score=winner.raw_score,
        normalized_score=winner.normalized_score,
        provenance=merged_provenance,
        retrieved_from_ontologies=retrieved_from_ontologies,
    )
