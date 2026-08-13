"""
LocalSemanticRetriever — strict local-only wrapper for SapBERT/FAISS retrieval.

Accepts a QueryPlan with retrieval_mode=LOCAL and returns raw candidate
dictionaries from a local SapBERT/FAISS semantic search service.

Strict local guarantees:
  - Does NOT call SearchTools, PublicOntologyRetriever, or OntologyRetriever.
  - Does NOT fall back to public APIs when the local service is unavailable.
  - Raises LocalRetrievalError on any failure; never silently returns [].
  - Returns [] only when the local service itself returns no results.

OntologyRetriever._search_sapbert was reviewed but NOT reused because
OntologyRetriever.search_ontology silently falls back to public APIs
(OLS/LOINC/RxNav/ICD) when SapBERT returns empty results.  That fallback
violates the strict local-only contract required for Phase 8.

Expected SapBERT/FAISS service protocol:
  POST {sapbert_url}/search
  Body : {"query": str, "ontology": str | None, "top_k": int}
  Reply: {"results": [{"code": str, "term": str, "score": float,
                        "definition": str}, ...]}

  If "ontology" is absent or null, the service performs broad semantic
  search across all loaded indices.

Phase 8 of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests  # type: ignore[import-untyped]

from llm_ontology_mapper.models import (
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)

logger = logging.getLogger(__name__)

# Ontology-name → SapBERT index key.
# Mirrors the fallback map inside OntologyRetriever._search_sapbert so that
# the same SapBERT service is compatible with both code paths.
# The user-facing requested_ontology is preserved separately in provenance.
_SAPBERT_INDEX_MAP: dict[str, str] = {
    "HP": "HPO",
    "HPO": "HPO",
    "MONDO": "MONDO",
    "NCIT": "NCIT",
    "LOINC": "LOINC",
    "SNOMED": "SNOMED",
    "SNOMEDCT": "SNOMED",
    "SNOMED-CT": "SNOMED",
    "RXNORM": "RXNORM",
    "RXCUI": "RXNORM",
    "RXNAV": "RXNORM",
    "ICD10": "ICD10CM",
    "ICD10CM": "ICD10CM",
    "ICD": "ICD10CM",
    "ICD-10": "ICD10CM",
    "ICD-10-CM": "ICD10CM",
    "CHEBI": "CHEBI",
    "UBERON": "UBERON",
    "GO": "GO",
    "DOID": "DOID",
    "MESH": "MESH",
    "UO": "UO",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class LocalRetrievalError(Exception):
    """
    Raised when the local retrieval path cannot proceed.

    Causes include:
      - retrieval_mode is not local (public or disabled)
      - no client or sapbert_url was configured
      - unexpected exception from the local client
      - malformed response from the local service
    """


# ─────────────────────────────────────────────────────────────────────────────
# SapBERTClient — default HTTP client for the local service
# ─────────────────────────────────────────────────────────────────────────────


class SapBERTClient:
    """
    Thin HTTP client for a SapBERT/FAISS local semantic search endpoint.

    Sends POST {sapbert_url}/search with a JSON payload and returns the
    parsed result list.  Raises LocalRetrievalError on HTTP errors, network
    failures, or malformed responses.  Never falls back to public APIs.

    Args:
        sapbert_url: Base URL of the local service, e.g. 'http://localhost:8080'.
        timeout:     Per-request timeout in seconds.
    """

    def __init__(self, sapbert_url: str, *, timeout: float = 10.0) -> None:
        self._url = sapbert_url.rstrip("/")
        self._timeout = timeout

    def search(
        self,
        query: str,
        ontology: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        POST /search and return parsed result list.

        Args:
            query:    The search string.
            ontology: SapBERT index key, or None for broad semantic search.
            top_k:    Maximum number of results to request.

        Returns:
            List of raw candidate dicts with at minimum: code, term, score,
            definition, source.

        Raises:
            LocalRetrievalError: on HTTP errors, network failures, or
                malformed response structure.
        """
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if ontology is not None:
            payload["ontology"] = ontology

        try:
            resp = requests.post(
                f"{self._url}/search",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise LocalRetrievalError(
                f"Local SapBERT service request failed "
                f"(url={self._url!r}, query={query!r}, ontology={ontology!r}): {exc}"
            ) from exc

        try:
            data: Any = resp.json()
        except Exception as exc:
            raise LocalRetrievalError(
                f"Local SapBERT response is not valid JSON (query={query!r}): {exc}"
            ) from exc

        if not isinstance(data, dict) or "results" not in data:
            raise LocalRetrievalError(
                f"Local SapBERT response missing 'results' key "
                f"(query={query!r}). Got: {str(data)[:200]}"
            )

        results = data["results"]
        if not isinstance(results, list):
            raise LocalRetrievalError(
                f"Local SapBERT 'results' is not a list "
                f"(query={query!r}). Got: {type(results).__name__}"
            )

        candidates: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    "code": item.get("code", ""),
                    "term": item.get("term", ""),
                    "score": float(item.get("score", 0.0)),
                    "definition": item.get("definition", ""),
                    "source": "SapBERT",
                }
            )
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# LocalSemanticRetriever
# ─────────────────────────────────────────────────────────────────────────────


class LocalSemanticRetriever:
    """
    Strict local-only wrapper that consumes a QueryPlan and returns raw
    candidate dictionaries from a SapBERT/FAISS semantic search service.

    Guarantees:
      - Only retrieval_mode=LOCAL is accepted.
      - Never calls SearchTools, PublicOntologyRetriever, or OntologyRetriever.
      - Never falls back to public APIs.
      - Raises LocalRetrievalError on failure; returns [] only when the
        local service itself produces no results.

    Ontology policy when no target_ontology_constraint is set:
      - preferred_ontology is searched first.
      - candidate_ontologies follow in order.
      - If no ontology is specified at all, the client performs broad
        semantic search (ontology=None in the request payload).

    Usage::

        retriever = LocalSemanticRetriever(sapbert_url="http://localhost:8080")
        candidates = retriever.retrieve(query_plan)
        # [{"code": "HP:0012735", "term": "Cough", "score": 0.93,
        #   "matched_query": "cough", "retrieval_mode": "local",
        #   "requested_ontology": "HPO", "route_name": "local_sapbert"}, ...]
    """

    def __init__(
        self,
        *,
        sapbert_url: str | None = None,
        client: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        if client is not None:
            self._client: Any = client
        elif sapbert_url:
            self._client = SapBERTClient(sapbert_url, timeout=timeout)
        else:
            self._client = None

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
        Execute local semantic retrieval for the given QueryPlan.

        Args:
            query_plan:            Must have retrieval_mode=LOCAL.
            route_plan:            Optional pre-computed RetrievalRoutePlan.
                                   When provided, its queries and
                                   candidate_ontologies take precedence.
            max_results_per_query: Maximum candidates per (query, ontology) pair.

        Returns:
            Flat list of raw candidate dicts.  Each dict carries the original
            local service fields plus provenance:
              - matched_query      : the query that produced the hit
              - retrieval_mode     : "local"
              - requested_ontology : user-facing ontology name (or None for broad)
              - route_name         : "local_sapbert"

        Raises:
            LocalRetrievalError: if retrieval_mode is not local, no client is
                configured, or the local service call fails.
        """
        if query_plan.retrieval_mode != RetrievalMode.LOCAL:
            raise LocalRetrievalError(
                f"LocalSemanticRetriever only handles retrieval_mode=local. "
                f"Got retrieval_mode={query_plan.retrieval_mode.value!r}. "
                f"Use PublicOntologyRetriever for public mode; "
                f"DisabledMappingRunner for disabled mode."
            )

        if self._client is None:
            raise LocalRetrievalError(
                "No local client configured. "
                "Provide sapbert_url or inject a client via the client= parameter."
            )

        queries = self._resolve_queries(query_plan, route_plan)
        ontologies = self._resolve_ontologies(query_plan, route_plan)

        results: list[dict[str, Any]] = []

        if not ontologies:
            # Broad semantic search: no ontology filter
            for query in queries:
                logger.debug("LocalSemanticRetriever: query=%r ontology=None (broad)", query)
                raw = self._call_client_timed(
                    query,
                    None,
                    max_results_per_query,
                    route_call_sink=route_calls,
                    route_call={
                        "route": "local_sapbert",
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
                        "candidate_ontologies": [],
                    },
                )
                for candidate in raw:
                    enriched: dict[str, Any] = dict(candidate)
                    enriched.setdefault("matched_query", query)
                    enriched["retrieval_mode"] = RetrievalMode.LOCAL.value
                    enriched["requested_ontology"] = None
                    enriched.setdefault("route_name", "local_sapbert")
                    results.append(enriched)
        else:
            for query in queries:
                for ontology in ontologies:
                    index_key = _SAPBERT_INDEX_MAP.get(ontology.upper(), ontology.upper())
                    logger.debug(
                        "LocalSemanticRetriever: query=%r ontology=%r index_key=%r",
                        query,
                        ontology,
                        index_key,
                    )
                    raw = self._call_client_timed(
                        query,
                        index_key,
                        max_results_per_query,
                        route_call_sink=route_calls,
                        route_call={
                            "route": "local_sapbert",
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
                            "candidate_ontologies": [ontology],
                        },
                    )
                    for candidate in raw:
                        enriched = dict(candidate)
                        enriched.setdefault("matched_query", query)
                        enriched["retrieval_mode"] = RetrievalMode.LOCAL.value
                        enriched["requested_ontology"] = ontology
                        enriched.setdefault("route_name", "local_sapbert")
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
        Return the ordered list of user-facing ontology names to search.

        target_ontology_constraint is always a hard constraint.
        When absent, preferred_ontology comes first, then candidate_ontologies.
        allowed_target_ontologies is a hard allow-list. When present, only
        ontologies in that list are returned; if no preferred/candidate
        ontology survives, the full allow-list is returned.
        Returns [] when no ontology can be inferred (signals broad search).
        """
        constraint = query_plan.target_ontology_constraint or (
            route_plan.target_ontology_constraint if route_plan else None
        )
        allowed = _resolve_allowed_target_ontologies(query_plan, route_plan)
        if constraint:
            constraint_upper = constraint.upper()
            if allowed is not None and constraint_upper not in allowed:
                return []
            return [constraint_upper]

        ontologies: list[str] = []
        seen: set[str] = set()

        def _add(onto: str) -> None:
            key = onto.upper()
            if key and key not in seen:
                ontologies.append(key)
                seen.add(key)

        if query_plan.preferred_ontology:
            _add(query_plan.preferred_ontology)

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

    def _call_client(
        self,
        query: str,
        ontology: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Invoke the local client, wrapping unexpected exceptions.

        LocalRetrievalError is re-raised as-is.  All other exceptions are
        wrapped in LocalRetrievalError so callers see a consistent error type.
        """
        try:
            return self._client.search(query, ontology, top_k)
        except LocalRetrievalError:
            raise
        except Exception as exc:
            raise LocalRetrievalError(
                f"Unexpected error from local semantic client "
                f"(query={query!r}, ontology={ontology!r}): {exc}"
            ) from exc

    def _call_client_timed(
        self,
        query: str,
        ontology: str | None,
        top_k: int,
        *,
        route_call_sink: list[dict[str, Any]] | None,
        route_call: dict[str, Any],
    ) -> list[dict[str, Any]]:
        start = time.monotonic()
        candidate_count: int | None = None
        try:
            candidates = self._call_client(query, ontology, top_k)
            candidate_count = len(candidates)
            return candidates
        finally:
            if route_call_sink is not None:
                elapsed_ms = (time.monotonic() - start) * 1000
                timed_call = dict(route_call)
                timed_call["latency_ms"] = elapsed_ms
                if candidate_count is not None:
                    timed_call["candidate_count"] = candidate_count
                route_call_sink.append(timed_call)


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
        key = str(ontology or "").upper().strip()
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result or None
