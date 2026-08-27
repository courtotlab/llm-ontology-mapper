"""
PublicOntologyRetriever — thin wrapper around SearchTools for public-mode retrieval.

Accepts a QueryPlan with retrieval_mode=PUBLIC and returns raw candidate
dictionaries from OLS, LOINC, RxNav, and ICD public APIs.

No normalization, merging, or reranking is performed here.  Raw dicts are
enriched with provenance fields (matched_query, retrieval_mode, requested_ontology,
route_name) and returned as-is for downstream processing by CandidateNormalizer.

Phase 7 of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_ontology_mapper.models import (
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)
from llm_ontology_mapper.ontology_identity import canonical_ontology
from llm_ontology_mapper.search_tools import SearchTools

logger = logging.getLogger(__name__)

# Ontology families served by each public route.
# Keys are upper-cased ontology identifiers as callers/planners may supply them.
_LOINC_ONTOLOGIES: frozenset[str] = frozenset({"LOINC"})

_RXNORM_ONTOLOGIES: frozenset[str] = frozenset({"RXNORM", "RXNAV", "RXCUI"})

_ICD_ONTOLOGIES: frozenset[str] = frozenset({"ICD10", "ICD10CM", "ICD", "ICD-10", "ICD-10-CM"})

# OLS-routed ontologies are whatever SearchTools.OLS_ONTOLOGY_MAP supports.
# We reference the class attribute directly so additions to SearchTools propagate.
_OLS_ONTOLOGIES: frozenset[str] = frozenset(SearchTools.OLS_ONTOLOGY_MAP.keys())

_PUBLIC_ROUTE_ONTOLOGY_ALIASES: dict[str, str] = {
    "SNOMED-CT": "SNOMED",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class PublicRetrievalError(Exception):
    """
    Raised when the public retrieval path cannot proceed.

    Causes include:
      - retrieval_mode is not public (local or disabled)
      - no ontology route can be inferred (no constraint, no preferred, no candidates)
      - ontology is not supported by any known public route
      - an unexpected exception from a SearchTools call
    """


# ─────────────────────────────────────────────────────────────────────────────
# PublicOntologyRetriever
# ─────────────────────────────────────────────────────────────────────────────


class PublicOntologyRetriever:
    """
    Thin wrapper around SearchTools that consumes a QueryPlan and returns
    raw public candidate dictionaries.

    No normalization, deduplication, or reranking is performed.  Downstream
    pipeline stages (CandidateNormalizer → CandidateMerger → LLMReranker)
    handle those concerns.

    Supported public routes:
      - OLS  (EBI OLS4)     : HP/HPO, MONDO, NCIT, SNOMED, UBERON, CHEBI, …
      - LOINC Search API    : LOINC
      - RxNav               : RXNORM / RXNAV
      - NIH Clinical Tables : ICD10 / ICD10CM / ICD

    Usage::

        retriever = PublicOntologyRetriever(search_tools=fake_tools)
        candidates = retriever.retrieve(query_plan)
        # [{"code": "HP:0012735", "term": "Cough", "matched_query": "cough",
        #   "retrieval_mode": "public", "requested_ontology": "HPO",
        #   "route_name": "OLS", ...}, ...]
    """

    def __init__(self, search_tools: SearchTools | None = None) -> None:
        self._tools: SearchTools = search_tools if search_tools is not None else SearchTools()

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_plan: QueryPlan,
        *,
        route_plan: RetrievalRoutePlan | None = None,
        max_results_per_query: int = 10,
        route_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute public retrieval for the given QueryPlan.

        Args:
            query_plan:            Must have retrieval_mode=PUBLIC.
            route_plan:            Optional pre-computed RetrievalRoutePlan.
                                   When provided, its queries and
                                   candidate_ontologies take precedence over
                                   the equivalent fields derived from query_plan.
            max_results_per_query: Maximum candidates to fetch per (query, ontology)
                                   pair.  Defaults to 10.

        Returns:
            Flat list of raw candidate dicts, one entry per hit.  Each dict
            carries the original SearchTools fields plus:
              - matched_query      : the query string that produced the hit
              - retrieval_mode     : "public"
              - requested_ontology : the ontology name sent to the route
              - route_name         : the underlying search route identifier

        Raises:
            PublicRetrievalError: if retrieval_mode is not public, no ontology
                route can be inferred, an unsupported ontology is requested,
                or an unexpected SearchTools exception occurs.
        """
        if query_plan.retrieval_mode != RetrievalMode.PUBLIC:
            raise PublicRetrievalError(
                f"PublicOntologyRetriever only handles retrieval_mode=public. "
                f"Got retrieval_mode={query_plan.retrieval_mode.value!r}. "
                f"Use DisabledMappingRunner for disabled mode; "
                f"LocalSemanticRetriever for local mode."
            )

        queries = self._resolve_queries(query_plan, route_plan)
        ontologies = self._resolve_ontologies(query_plan, route_plan)

        if not ontologies:
            raise PublicRetrievalError(
                f"No ontology route could be inferred for "
                f"source_term={query_plan.original_term!r}. "
                f"Supply target_ontology_constraint, preferred_ontology, or "
                f"candidate_ontologies so the retriever knows where to search."
            )

        results: list[dict[str, Any]] = []
        for query in queries:
            for ontology in ontologies:
                route_ontology = public_route_ontology(ontology)
                logger.debug(
                    "PublicOntologyRetriever: query=%r ontology=%r",
                    query,
                    route_ontology,
                )
                route = _route_name(route_ontology)
                raw_candidates = self._call_route_timed(
                    query,
                    route_ontology,
                    max_results_per_query,
                    route_call_sink=route_calls,
                    route_call={
                        "route": "public_api",
                        "route_name": route,
                        "query": query,
                        "target_ontology": query_plan.target_ontology_constraint
                        or (route_plan.target_ontology_constraint if route_plan else None),
                        "allowed_target_ontologies": (
                            list(query_plan.allowed_target_ontologies)
                            if query_plan.allowed_target_ontologies is not None
                            else (
                                list(route_plan.allowed_target_ontologies)
                                if route_plan is not None
                                and route_plan.allowed_target_ontologies is not None
                                else None
                            )
                        ),
                        "candidate_ontologies": [route_ontology],
                    },
                )
                for candidate in raw_candidates:
                    enriched: dict[str, Any] = dict(candidate)
                    enriched.setdefault("matched_query", query)
                    enriched["retrieval_mode"] = RetrievalMode.PUBLIC.value
                    enriched["requested_ontology"] = route_ontology
                    enriched.setdefault("route_name", route)
                    results.append(enriched)

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_queries(
        query_plan: QueryPlan,
        route_plan: RetrievalRoutePlan | None,
    ) -> list[str]:
        """
        Return deduplicated, blank-filtered query list.

        If route_plan is provided and has queries, those take precedence.
        Otherwise queries are derived from query_plan with the standard
        fallback chain: expanded_queries → inferred_meaning → original_label
        → normalized_term → original_term.
        """
        if route_plan is not None and route_plan.queries:
            raw = [q.strip() for q in route_plan.queries if q and q.strip()]
            return list(dict.fromkeys(raw))

        raw = [q.strip() for q in query_plan.expanded_queries if q and q.strip()]

        if not raw:
            for candidate in (
                query_plan.inferred_meaning,
                query_plan.original_label,
                query_plan.normalized_term,
                query_plan.original_term,
            ):
                if candidate and candidate.strip():
                    raw = [candidate.strip()]
                    break

        return list(dict.fromkeys(raw))

    @staticmethod
    def _resolve_ontologies(
        query_plan: QueryPlan,
        route_plan: RetrievalRoutePlan | None,
    ) -> list[str]:
        """
        Return the ordered list of ontologies to search.

        target_ontology_constraint is always a hard constraint; it overrides
        everything else.  When absent, preferred_ontology comes first, then
        candidate_ontologies (deduplicated).

        allowed_target_ontologies is a hard allow-list. When present, only
        ontologies in that list are returned. If no preferred/candidate
        ontology survives the allow-list, the full allow-list is returned so
        all caller-selected ontologies remain searchable.

        If route_plan is provided, its target_ontology_constraint and
        candidate_ontologies take precedence over query_plan's equivalent fields
        (but query_plan.target_ontology_constraint is still the authoritative
        hard constraint).
        """
        # Hard constraint wins unconditionally
        constraint = query_plan.target_ontology_constraint or (
            route_plan.target_ontology_constraint if route_plan else None
        )
        allowed = _resolve_allowed_target_ontologies(query_plan, route_plan)
        if constraint:
            route_constraint = public_route_ontology(constraint)
            if allowed is not None and route_constraint not in allowed:
                return []
            return [route_constraint]

        ontologies: list[str] = []
        seen: set[str] = set()

        def _add(onto: str) -> None:
            key = public_route_ontology(onto)
            if key and key not in seen:
                ontologies.append(key)
                seen.add(key)

        # preferred_ontology from query_plan always comes first
        if query_plan.preferred_ontology:
            _add(query_plan.preferred_ontology)

        # candidate_ontologies: route_plan if available, else query_plan
        candidates_source = (
            route_plan.candidate_ontologies
            if route_plan is not None
            else query_plan.candidate_ontologies
        )
        for onto in candidates_source:
            _add(onto)

        if allowed is None:
            return ontologies

        filtered = [onto for onto in ontologies if onto in allowed]
        return filtered or list(allowed)

    def _call_route(
        self,
        query: str,
        ontology: str,
        top_k: int,
        *,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dispatch a single (query, ontology) pair to the correct SearchTools method.

        route_diagnostics, when provided, is forwarded to the SearchTools call
        and populated in place with retry/error telemetry -- see
        SearchTools._get_with_retry / _record_http_error_diagnostics.

        Raises:
            PublicRetrievalError: if the ontology is not supported, or if the
                SearchTools call raises an unexpected exception.
        """
        onto_upper = public_route_ontology(ontology)

        try:
            if onto_upper in _LOINC_ONTOLOGIES:
                return self._tools.search_loinc(
                    query, top_k=top_k, route_diagnostics=route_diagnostics
                )

            if onto_upper in _RXNORM_ONTOLOGIES:
                return self._tools.search_rxnorm(
                    query, top_k=top_k, route_diagnostics=route_diagnostics
                )

            if onto_upper in _ICD_ONTOLOGIES:
                return self._tools.search_icd10(
                    query, top_k=top_k, route_diagnostics=route_diagnostics
                )

            if onto_upper in _OLS_ONTOLOGIES:
                return self._tools.search_ols(
                    query,
                    ontology=onto_upper,
                    top_k=top_k,
                    route_diagnostics=route_diagnostics,
                )

        except Exception as exc:
            raise PublicRetrievalError(
                f"Unexpected error calling public route for ontology={ontology!r}, "
                f"query={query!r}: {exc}"
            ) from exc

        raise PublicRetrievalError(
            f"Ontology {ontology!r} is not supported by any known public route. "
            f"Supported OLS ontologies: {', '.join(sorted(_OLS_ONTOLOGIES))}; "
            f"also supported: LOINC, RXNORM/RXNAV, ICD10/ICD10CM."
        )

    def _call_route_timed(
        self,
        query: str,
        ontology: str,
        top_k: int,
        *,
        route_call_sink: list[dict[str, Any]] | None,
        route_call: dict[str, Any],
    ) -> list[dict[str, Any]]:
        start = time.monotonic()
        candidate_count: int | None = None
        diagnostics: dict[str, Any] = {}
        try:
            candidates = self._call_route(query, ontology, top_k, route_diagnostics=diagnostics)
            candidate_count = len(candidates)
            return candidates
        finally:
            if route_call_sink is not None:
                elapsed_ms = (time.monotonic() - start) * 1000
                timed_call = dict(route_call)
                timed_call["latency_ms"] = elapsed_ms
                if candidate_count is not None:
                    timed_call["candidate_count"] = candidate_count
                # attempts defaults to 1 when SearchTools never populated
                # diagnostics (e.g. an unsupported-ontology PublicRetrievalError
                # raised before any HTTP call was attempted).
                timed_call["retrieval_attempts"] = diagnostics.get("attempts", 1)
                final_error_type = diagnostics.get("final_error_type")
                if final_error_type:
                    timed_call["retrieval_final_error_type"] = final_error_type
                route_call_sink.append(timed_call)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _route_name(ontology: str) -> str:
    """Return the canonical route name for an ontology identifier."""
    onto_upper = public_route_ontology(ontology)
    if onto_upper in _LOINC_ONTOLOGIES:
        return "LOINC-Search-API"
    if onto_upper in _RXNORM_ONTOLOGIES:
        return "RxNav"
    if onto_upper in _ICD_ONTOLOGIES:
        return "NIH-ClinicalTables"
    return "OLS"


def public_route_ontology(ontology: str) -> str:
    """Return the public-retrieval route identifier for a user/planner ontology."""
    raw = str(ontology or "").upper().strip()
    canonical = canonical_ontology(raw) or raw
    return _PUBLIC_ROUTE_ONTOLOGY_ALIASES.get(canonical, canonical)


def _resolve_allowed_target_ontologies(
    query_plan: QueryPlan,
    route_plan: RetrievalRoutePlan | None,
) -> list[str] | None:
    raw = (
        query_plan.allowed_target_ontologies
        if query_plan.allowed_target_ontologies is not None
        else (route_plan.allowed_target_ontologies if route_plan is not None else None)
    )
    if raw is None:
        return None

    result: list[str] = []
    seen: set[str] = set()
    for ontology in raw:
        key = public_route_ontology(str(ontology or ""))
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result or None
