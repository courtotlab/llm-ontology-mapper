"""OLS4Source — RetrievalSource adapter for EBI OLS4 (multi-ontology search)."""

from __future__ import annotations

from typing import Any

from llm_ontology_mapper.search_tools import SearchTools


class OLS4Source:
    """Adapter over SearchTools.search_ols.

    HTTP interaction, retry, response parsing, and identity resolution all
    remain inside SearchTools unchanged -- see SearchTools.search_ols. This
    adapter's only job is to satisfy the RetrievalSource contract so
    PublicOntologyRetriever can dispatch to OLS4 through the registry
    instead of a hard-coded ontology check.
    """

    source_id = "ols4"
    route_name = "OLS"

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
        return self._tools.search_ols(
            query,
            ontology=ontology_id,
            top_k=top_k,
            route_diagnostics=route_diagnostics,
        )
