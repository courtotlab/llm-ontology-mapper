"""LOINCSource — RetrievalSource adapter for the LOINC Search API."""

from __future__ import annotations

from typing import Any

from llm_ontology_mapper.search_tools import SearchTools


class LOINCSource:
    """Adapter over SearchTools.search_loinc.

    LOINC serves exactly one ontology, so `ontology_id` is accepted for
    protocol conformance but ignored. Authentication, the ACTIVE-status
    filter, retry, and response parsing all remain inside SearchTools
    unchanged -- see SearchTools.search_loinc.
    """

    source_id = "loinc"
    route_name = "LOINC-Search-API"

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
        return self._tools.search_loinc(
            query,
            top_k=top_k,
            route_diagnostics=route_diagnostics,
        )
