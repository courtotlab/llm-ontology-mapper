"""
Public retrieval-source adapters for the planned pipeline's public retrieval
stage (PublicOntologyRetriever).

See registry.py for the one source-registration location, config.py for the
ontology -> source configuration resolution, and protocol.py for the
RetrievalSource contract a new endpoint adapter implements.
"""

from __future__ import annotations

from llm_ontology_mapper.retrieval_sources.config import (
    ResolvedRoute,
    RetrievalConfigError,
    resolve_retrieval_route,
    validate_retrieval_config,
)
from llm_ontology_mapper.retrieval_sources.loinc import LOINCSource
from llm_ontology_mapper.retrieval_sources.nih_clinical_tables import (
    NIHClinicalTablesSource,
)
from llm_ontology_mapper.retrieval_sources.ols4 import OLS4Source
from llm_ontology_mapper.retrieval_sources.protocol import RetrievalSource
from llm_ontology_mapper.retrieval_sources.registry import (
    build_registry,
    known_source_ids,
    register_source,
    route_name_for_source,
    unregister_source,
)
from llm_ontology_mapper.retrieval_sources.rxnav import RxNavSource

__all__ = [
    "RetrievalSource",
    "RetrievalConfigError",
    "ResolvedRoute",
    "resolve_retrieval_route",
    "validate_retrieval_config",
    "OLS4Source",
    "LOINCSource",
    "RxNavSource",
    "NIHClinicalTablesSource",
    "build_registry",
    "register_source",
    "unregister_source",
    "known_source_ids",
    "route_name_for_source",
]
