"""RxNavSource — RetrievalSource adapter for RxNav (RxNorm)."""

from __future__ import annotations

from typing import Any

from llm_ontology_mapper.search_tools import SearchTools


class RxNavSource:
    """Adapter over SearchTools.search_rxnorm.

    RxNav serves exactly one ontology, so `ontology_id` is accepted for
    protocol conformance but ignored. Atom/rxcui grouping, dedup, preferred
    naming, retry, and response parsing all remain inside SearchTools
    unchanged -- see SearchTools.search_rxnorm.
    """

    source_id = "rxnav"
    route_name = "RxNav"

    def __init__(self, tools: SearchTools) -> None:
        self._tools = tools

    def search(
        self,
        query: str,
        *,
        ontology_id: str,
        top_k: int,
        route_diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._tools.search_rxnorm(
            query,
            top_k=top_k,
            route_diagnostics=route_diagnostics,
        )
