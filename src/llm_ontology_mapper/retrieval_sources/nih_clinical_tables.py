"""NIHClinicalTablesSource — RetrievalSource adapter for NIH Clinical Tables (ICD-10-CM)."""

from __future__ import annotations

from typing import Any

from llm_ontology_mapper.search_tools import SearchTools


class NIHClinicalTablesSource:
    """Adapter over SearchTools.search_icd10.

    NIH Clinical Tables serves exactly one ontology, so `ontology_id` is
    accepted for protocol conformance but ignored. Positional-array response
    parsing, retry, and scoring all remain inside SearchTools unchanged --
    see SearchTools.search_icd10.
    """

    source_id = "nih_clinical_tables"
    route_name = "NIH-ClinicalTables"

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
        return self._tools.search_icd10(
            query,
            top_k=top_k,
            route_diagnostics=route_diagnostics,
        )
