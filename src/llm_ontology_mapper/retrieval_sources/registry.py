"""
The one retrieval-source registration location.

To add a brand-new public retrieval endpoint (extensibility CASE 2):
  1. Implement a RetrievalSource adapter (see protocol.py; ols4.py is the
     reference implementation).
  2. Add one line to _SOURCE_CLASSES below.
  3. Point an ontology at it via ontology_config.yaml's `retrieval.source`.

Nothing else needs to change: PublicOntologyRetriever resolves
ontology -> source purely through ontology_config.yaml + this registry.
"""

from __future__ import annotations

from llm_ontology_mapper.retrieval_sources.loinc import LOINCSource
from llm_ontology_mapper.retrieval_sources.nih_clinical_tables import (
    NIHClinicalTablesSource,
)
from llm_ontology_mapper.retrieval_sources.ols4 import OLS4Source
from llm_ontology_mapper.retrieval_sources.protocol import RetrievalSource
from llm_ontology_mapper.retrieval_sources.rxnav import RxNavSource
from llm_ontology_mapper.search_tools import SearchTools

# ── THE one registration point ──────────────────────────────────────────────
_SOURCE_CLASSES: dict[str, type] = {
    "ols4": OLS4Source,
    "loinc": LOINCSource,
    "rxnav": RxNavSource,
    "nih_clinical_tables": NIHClinicalTablesSource,
}


def register_source(source_id: str, source_cls: type) -> None:
    """Register (or override, e.g. for tests) a RetrievalSource implementation.

    This is the same mechanism a real new endpoint is registered through --
    tests exercising extensibility CASE 2 call this directly with a fake
    source instead of editing _SOURCE_CLASSES above.
    """
    _SOURCE_CLASSES[source_id] = source_cls


def unregister_source(source_id: str) -> None:
    """Remove a previously registered source (test cleanup helper)."""
    _SOURCE_CLASSES.pop(source_id, None)


def known_source_ids() -> set[str]:
    return set(_SOURCE_CLASSES)


def route_name_for_source(source_id: str) -> str | None:
    cls = _SOURCE_CLASSES.get(source_id)
    return getattr(cls, "route_name", None) if cls is not None else None


def build_registry(tools: SearchTools) -> dict[str, RetrievalSource]:
    """Construct one bound adapter instance per registered source.

    Every adapter shares the same SearchTools instance, so credentials and
    base URLs injected into that instance (e.g. Bridge's LOINC credential
    wiring) apply uniformly across all sources.
    """
    return {source_id: cls(tools) for source_id, cls in _SOURCE_CLASSES.items()}
