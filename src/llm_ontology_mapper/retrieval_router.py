"""
RetrievalRouter — converts a QueryPlan into a RetrievalRoutePlan.

The router decides what retrieval should happen based on the retrieval_mode
and constraints in the QueryPlan.  It does not perform actual retrieval, call
external APIs, or produce NormalizedCandidate records.

Phase 3 of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_ontology_mapper.models import (
    GroundingSource,
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """
    Plans retrieval routes from a QueryPlan.

    Supports exactly three modes:
      - public:   grounded via public ontology APIs
      - local:    grounded via local semantic index (SapBERT / FAISS)
      - disabled: ungrounded, LLM-only; route_calls is empty

    No external calls are made.  route_calls contains planned descriptions
    only, consumed by the concrete retriever in the next pipeline stage.

    Usage::

        from llm_ontology_mapper.retrieval_router import RetrievalRouter

        router = RetrievalRouter()
        route_plan = router.route(query_plan)
        print(route_plan.queries)        # ['systolic blood pressure', 'systolic BP']
        print(route_plan.grounding_source)  # GroundingSource.PUBLIC_API
    """

    def route(self, query_plan: QueryPlan) -> RetrievalRoutePlan:
        """
        Convert a QueryPlan into a RetrievalRoutePlan.

        Python-side rules applied here:
        - retrieval_mode determines grounding_source (1:1 mapping, no both mode).
        - target_ontology_constraint restricts candidate_ontologies.
        - expanded_queries are used as-is; fallback applied if empty.
        - Disabled mode produces retrieval_skipped=True, is_grounded_mode=False,
          grounding_source=none, and empty route_calls.

        No HTTP calls, no external service access.
        """
        mode = query_plan.retrieval_mode
        queries = self._resolve_queries(query_plan)
        candidate_ontologies = self._resolve_candidate_ontologies(query_plan)
        target_constraint = query_plan.target_ontology_constraint
        allowed_target_ontologies = query_plan.allowed_target_ontologies

        logger.debug(
            "RetrievalRouter.route: mode=%s queries=%r target=%r",
            mode.value,
            queries,
            target_constraint,
        )

        if mode == RetrievalMode.DISABLED:
            return RetrievalRoutePlan(
                retrieval_mode=mode,
                is_grounded_mode=False,
                retrieval_skipped=True,
                grounding_source=GroundingSource.NONE,
                queries=queries,
                target_ontology_constraint=target_constraint,
                allowed_target_ontologies=allowed_target_ontologies,
                candidate_ontologies=candidate_ontologies,
                route_calls=[],
                retrieval_disabled_reason=(
                    query_plan.retrieval_disabled_reason or "Retrieval disabled by caller"
                ),
            )

        grounding_source = (
            GroundingSource.PUBLIC_API
            if mode == RetrievalMode.PUBLIC
            else GroundingSource.LOCAL_SAPBERT
        )
        route_calls = self._build_route_calls(
            mode,
            queries,
            target_constraint,
            allowed_target_ontologies,
            candidate_ontologies,
        )

        return RetrievalRoutePlan(
            retrieval_mode=mode,
            is_grounded_mode=True,
            retrieval_skipped=False,
            grounding_source=grounding_source,
            queries=queries,
            target_ontology_constraint=target_constraint,
            allowed_target_ontologies=allowed_target_ontologies,
            candidate_ontologies=candidate_ontologies,
            route_calls=route_calls,
            retrieval_disabled_reason=None,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_queries(plan: QueryPlan) -> list[str]:
        """
        Return the query list, applying a fallback if expanded_queries is empty.

        Fallback priority: expanded_queries → inferred_meaning → original_label
                           → normalized_term → original_term.

        The result is deduplicated (order-preserving) and blank-filtered.
        """
        raw = [q.strip() for q in plan.expanded_queries if q and q.strip()]

        if not raw:
            for candidate in (
                plan.inferred_meaning,
                plan.original_label,
                plan.normalized_term,
                plan.original_term,
            ):
                if candidate and candidate.strip():
                    raw = [candidate.strip()]
                    break

        # Deduplicate preserving insertion order
        return list(dict.fromkeys(raw))

    @staticmethod
    def _resolve_candidate_ontologies(plan: QueryPlan) -> list[str]:
        """
        Respect target_ontology_constraint as a hard constraint.

        If set, restrict to that single ontology.
        If an allow-list is set, intersect the planner's candidates with it;
        fall back to the full allow-list when no planner candidate survives.
        Otherwise use the planner's candidate_ontologies list.
        """
        if plan.target_ontology_constraint:
            return [plan.target_ontology_constraint]
        if plan.allowed_target_ontologies:
            allowed = set(plan.allowed_target_ontologies)
            candidate_ontologies = [
                ontology for ontology in plan.candidate_ontologies if ontology in allowed
            ]
            return candidate_ontologies or list(plan.allowed_target_ontologies)
        return list(plan.candidate_ontologies)

    @staticmethod
    def _build_route_calls(
        mode: RetrievalMode,
        queries: list[str],
        target_ontology: str | None,
        allowed_target_ontologies: list[str] | None,
        candidate_ontologies: list[str],
    ) -> list[dict[str, Any]]:
        """
        Build one planned route-call descriptor per query.

        Each descriptor carries the route name, query, target ontology
        constraint, and the candidate ontologies to search.  These are
        descriptions only — no HTTP request is made.
        """
        route_name = "public_api" if mode == RetrievalMode.PUBLIC else "local_sapbert"
        return [
            {
                "route": route_name,
                "query": q,
                "target_ontology": target_ontology,
                "allowed_target_ontologies": (
                    list(allowed_target_ontologies) if allowed_target_ontologies else None
                ),
                "candidate_ontologies": list(candidate_ontologies),
            }
            for q in queries
        ]
