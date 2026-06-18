"""
CandidateNormalizer — converts raw retrieval candidate dicts into NormalizedCandidate.

Pure and side-effect free.  No external services are called.

Phase 4A of the planned grounded mapping pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable

from llm_ontology_mapper.models import (
    NormalizedCandidate,
    RetrievalMode,
)


class CandidateNormalizationError(Exception):
    """Raised when a raw candidate dict cannot be normalized into a NormalizedCandidate."""


class CandidateNormalizer:
    """
    Convert raw retrieval candidate dicts into NormalizedCandidate objects.

    The normalizer does not retrieve candidates.  It normalizes candidate
    data already returned by a public or local retriever.

    Supports retrieval_mode=public and retrieval_mode=local only.
    retrieval_mode=disabled raises CandidateNormalizationError because
    disabled mode does not perform retrieval and produces no candidates.

    Usage::

        normalizer = CandidateNormalizer()
        candidate = normalizer.normalize(
            {"code": "HP:0012735", "term": "Cough", "score": 0.95},
            retrieval_mode=RetrievalMode.PUBLIC,
            matched_query="cough",
            default_ontology="HPO",
            default_source="OLS",
        )
    """

    _CODE_KEYS = ("code", "id", "curie", "iri", "concept_id", "loinc_num", "rxnorm_id", "icd10_code")
    _TERM_KEYS = ("term", "label", "name", "display", "preferred_label")
    _ONTOLOGY_KEYS = ("ontology", "ontology_id", "ontology_prefix", "source_ontology")
    _DEFINITION_KEYS = ("definition", "description", "meaning")
    _SOURCE_KEYS = ("source", "source_name", "endpoint", "retriever")
    _SCORE_KEYS = ("score", "raw_score", "similarity", "confidence")

    def normalize(
        self,
        raw_candidate: dict[str, Any],
        *,
        retrieval_mode: RetrievalMode | str,
        matched_query: str,
        default_ontology: str | None = None,
        default_source: str | None = None,
    ) -> NormalizedCandidate:
        """
        Normalize a raw retrieval candidate dict into a NormalizedCandidate.

        Args:
            raw_candidate:   Raw candidate dict from a public or local retriever.
            retrieval_mode:  Mode that produced this candidate (public or local).
            matched_query:   The expanded query that produced this candidate.
            default_ontology: Fallback ontology when the raw dict has none.
            default_source:  Fallback source name when the raw dict has none.

        Returns:
            A NormalizedCandidate.

        Raises:
            CandidateNormalizationError: If the candidate is invalid or
                retrieval_mode is disabled.
        """
        mode = self._resolve_mode(retrieval_mode)

        if not matched_query or not matched_query.strip():
            raise CandidateNormalizationError("matched_query must not be blank")

        code = self._extract_first(raw_candidate, self._CODE_KEYS)
        if not code:
            raise CandidateNormalizationError(
                f"raw_candidate missing a recognizable code field "
                f"(tried: {self._CODE_KEYS}): {raw_candidate!r}"
            )

        term = self._extract_first(raw_candidate, self._TERM_KEYS)
        if not term:
            raise CandidateNormalizationError(
                f"raw_candidate missing a recognizable term field "
                f"(tried: {self._TERM_KEYS}): {raw_candidate!r}"
            )

        ontology = self._extract_ontology(raw_candidate, default_ontology)
        if not ontology:
            raise CandidateNormalizationError(
                f"raw_candidate missing ontology and no default_ontology provided: "
                f"{raw_candidate!r}"
            )

        definition = self._extract_definition(raw_candidate)
        source = self._extract_source(raw_candidate, default_source, mode)
        raw_score = self._extract_score(raw_candidate)
        normalized_score = self._extract_normalized_score(raw_candidate, raw_score)

        provenance: dict[str, Any] = {
            "raw_candidate": raw_candidate,
            "retrieval_mode": mode.value,
            "matched_query": matched_query,
        }
        if default_ontology is not None:
            provenance["default_ontology"] = default_ontology
        if default_source is not None:
            provenance["default_source"] = default_source

        return NormalizedCandidate(
            code=code,
            term=term,
            ontology=ontology,
            definition=definition,
            source=source,
            matched_query=matched_query,
            retrieval_mode=mode,
            raw_score=raw_score,
            normalized_score=normalized_score,
            provenance=provenance,
        )

    def normalize_many(
        self,
        raw_candidates: Iterable[dict[str, Any]],
        *,
        retrieval_mode: RetrievalMode | str,
        matched_query: str,
        default_ontology: str | None = None,
        default_source: str | None = None,
    ) -> list[NormalizedCandidate]:
        """
        Normalize a sequence of raw retrieval candidate dicts.

        Fail-fast: if any candidate is invalid, CandidateNormalizationError
        is raised immediately.  Nothing is silently skipped.

        Returns:
            A list of NormalizedCandidate objects.
        """
        return [
            self.normalize(
                raw,
                retrieval_mode=retrieval_mode,
                matched_query=matched_query,
                default_ontology=default_ontology,
                default_source=default_source,
            )
            for raw in raw_candidates
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_mode(retrieval_mode: RetrievalMode | str) -> RetrievalMode:
        """Validate and coerce retrieval_mode; reject disabled."""
        if isinstance(retrieval_mode, RetrievalMode):
            mode = retrieval_mode
        else:
            try:
                mode = RetrievalMode(str(retrieval_mode).lower())
            except ValueError:
                raise CandidateNormalizationError(
                    f"Invalid retrieval_mode: {retrieval_mode!r}. "
                    "Must be 'public' or 'local'."
                )
        if mode == RetrievalMode.DISABLED:
            raise CandidateNormalizationError(
                "retrieval_mode='disabled' cannot produce NormalizedCandidate records; "
                "disabled mode does not perform retrieval."
            )
        return mode

    @staticmethod
    def _extract_first(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        """Try each key in order; return the first non-blank string value."""
        for key in keys:
            val = raw.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        return None

    @staticmethod
    def _extract_ontology(raw: dict[str, Any], default_ontology: str | None) -> str | None:
        """Extract and uppercase the ontology; fall back to default_ontology."""
        for key in ("ontology", "ontology_id", "ontology_prefix", "source_ontology"):
            val = raw.get(key)
            if val is not None:
                s = str(val).strip().upper()
                if s:
                    return s
        if default_ontology:
            s = default_ontology.strip().upper()
            if s:
                return s
        return None

    @staticmethod
    def _extract_definition(raw: dict[str, Any]) -> str | None:
        """Extract definition; handle a list value for the 'definitions' key."""
        for key in ("definition", "description", "meaning"):
            val = raw.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        definitions = raw.get("definitions")
        if isinstance(definitions, list):
            for item in definitions:
                s = str(item).strip()
                if s:
                    return s
        elif definitions is not None:
            s = str(definitions).strip()
            if s:
                return s
        return None

    @staticmethod
    def _extract_source(
        raw: dict[str, Any],
        default_source: str | None,
        mode: RetrievalMode,
    ) -> str:
        """Extract source; fall back to default_source, then infer from mode."""
        for key in ("source", "source_name", "endpoint", "retriever"):
            val = raw.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        if default_source:
            s = default_source.strip()
            if s:
                return s
        return "public_api" if mode == RetrievalMode.PUBLIC else "local_sapbert"

    @staticmethod
    def _extract_score(raw: dict[str, Any]) -> float | None:
        """Extract raw_score; coerce numeric strings to float."""
        for key in ("score", "raw_score", "similarity", "confidence"):
            val = raw.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_normalized_score(
        raw: dict[str, Any],
        raw_score: float | None,
    ) -> float | None:
        """
        Extract normalized_score.

        Prefer an explicit 'normalized_score' key; otherwise reuse raw_score
        when it already falls in [0, 1].  Never invent calibration.
        """
        val = raw.get("normalized_score")
        if val is not None:
            try:
                f = float(val)
                if 0.0 <= f <= 1.0:
                    return f
            except (TypeError, ValueError):
                pass
        if raw_score is not None and 0.0 <= raw_score <= 1.0:
            return raw_score
        return None
