"""
RetrievalSource — the contract every public ontology retrieval endpoint
adapter must implement.

This is the seam a contributor implements once to add a brand-new public
retrieval endpoint (CASE 2 of the retrieval extensibility refactor). It owns
endpoint URL interaction, endpoint-specific authentication, request
construction, response parsing, and transformation into the raw candidate
dict contract already consumed by CandidateNormalizer. It does NOT own
candidate canonicalization, deduplication, target eligibility, reranking, or
MappingResult construction -- those remain downstream, unchanged.

Deliberately no `supports(ontology)` method: which ontologies a source
serves is declared in ontology_config.yaml (see
llm_ontology_mapper.retrieval_sources.config), not decided by the adapter
itself.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RetrievalSource(Protocol):
    """Protocol for a public ontology retrieval endpoint adapter.

    Attributes:
        source_id:  Stable registry key for this source (e.g. "ols4"). Used
            to bind ontology_config.yaml's `retrieval.source` value to a
            concrete implementation. Must never change once shipped -- it
            may be surfaced in diagnostics.
        route_name: Stable, human-facing provenance label attached to every
            raw candidate dict's `route_name` field (e.g. "OLS"). Kept
            distinct from source_id so today's externally observable
            provenance strings are preserved unchanged by this refactor.
    """

    source_id: str
    route_name: str

    def search(
        self,
        query: str,
        *,
        ontology_id: str,
        top_k: int,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search this endpoint and return raw candidate dicts.

        Args:
            query:             Search string.
            ontology_id:       Endpoint-specific ontology identifier, resolved
                                from ontology_config.yaml's
                                `retrieval.source_ontology_id` (or the
                                ontology's own canonical config key when not
                                overridden). A source that only ever serves
                                one ontology (e.g. LOINC, RxNav, ICD10) may
                                ignore this argument.
            top_k:              Maximum number of candidates to return.
            route_diagnostics:  Optional sink populated in place with
                                retry/error telemetry, mirroring
                                SearchTools._get_with_retry's contract.

        Returns:
            A list of raw candidate dicts in the existing CandidateNormalizer
            contract (code/term/ontology/score/definition/source, tolerant
            of several alias key names). Never raises for "no results" --
            returns an empty list. May raise on a genuinely unexpected error;
            callers (PublicOntologyRetriever) wrap that in
            PublicRetrievalError.
        """
        ...
