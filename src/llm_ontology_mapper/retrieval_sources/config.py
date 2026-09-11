"""
Ontology -> retrieval source resolution and validation.

The single authoritative home for "which public retrieval source serves
this ontology" is the `retrieval:` block on each ontology in
assets/ontology_config.yaml (see ontology_identity.get_ontology_config()).
This module turns that declarative data into a fast lookup and validates it
fails fast (at PublicOntologyRetriever construction time) rather than
degrading into "no candidates found" at request time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llm_ontology_mapper.ontology_identity import get_ontology_config


class RetrievalConfigError(Exception):
    """Raised when ontology_config.yaml's retrieval configuration is invalid.

    Causes include: an ontology's `retrieval.source` naming a source id that
    is not registered, a malformed `retrieval` block, or two ontologies
    declaring the same retrieval alias.
    """


@dataclass(frozen=True)
class ResolvedRoute:
    """Where a given (already public-route-normalized) ontology name routes."""

    config_key: str
    source_id: str
    source_ontology_id: str


def build_alias_index(ontologies_config: dict[str, Any]) -> dict[str, ResolvedRoute]:
    """Build an alias->ResolvedRoute index from an `ontologies:` config mapping.

    Every ontology with a `retrieval` block is indexed under its own
    (upper-cased) config key, plus every string listed in `retrieval.aliases`
    (also upper-cased). Ontologies without a `retrieval` block are not
    routable and are simply absent from the index -- looking one up returns
    None, the same "not supported" signal as today.

    Raises:
        RetrievalConfigError: a `retrieval` block is malformed, or two
            ontologies declare the same alias (including one ontology's own
            config key colliding with another ontology's alias).
    """
    index: dict[str, ResolvedRoute] = {}
    for config_key, meta in ontologies_config.items():
        if not isinstance(meta, dict):
            continue
        retrieval = meta.get("retrieval")
        if retrieval is None:
            continue
        if not isinstance(retrieval, dict):
            raise RetrievalConfigError(
                f"ontologies.{config_key}.retrieval must be a mapping, "
                f"got {type(retrieval).__name__}"
            )

        source_id = retrieval.get("source")
        if not source_id or not isinstance(source_id, str):
            raise RetrievalConfigError(
                f"ontologies.{config_key}.retrieval.source is required and must be "
                f"a non-empty string"
            )

        source_ontology_id = retrieval.get("source_ontology_id") or config_key
        if not isinstance(source_ontology_id, str):
            raise RetrievalConfigError(
                f"ontologies.{config_key}.retrieval.source_ontology_id must be a string"
            )

        raw_aliases = retrieval.get("aliases") or []
        if not isinstance(raw_aliases, list):
            raise RetrievalConfigError(
                f"ontologies.{config_key}.retrieval.aliases must be a list of strings"
            )

        route = ResolvedRoute(
            config_key=config_key,
            source_id=source_id,
            source_ontology_id=source_ontology_id,
        )

        for alias in (config_key, *raw_aliases):
            key = str(alias).upper().strip()
            if not key:
                continue
            existing = index.get(key)
            if existing is not None and existing.config_key != config_key:
                raise RetrievalConfigError(
                    f"retrieval alias {key!r} is declared by both "
                    f"ontologies.{existing.config_key} and ontologies.{config_key}; "
                    f"each retrieval alias must resolve to exactly one ontology"
                )
            index[key] = route

    return index


def validate_retrieval_config(
    ontologies_config: dict[str, Any],
    known_source_ids: set[str],
) -> None:
    """Fail fast on any retrieval config referencing an unregistered source.

    Raises:
        RetrievalConfigError: an ontology's `retrieval.source` is not one of
            known_source_ids, or the config itself is malformed (surfaced via
            build_alias_index).
    """
    index = build_alias_index(ontologies_config)
    for route in index.values():
        if route.source_id not in known_source_ids:
            raise RetrievalConfigError(
                f"ontologies.{route.config_key}.retrieval.source={route.source_id!r} "
                f"is not a registered retrieval source. Registered sources: "
                f"{sorted(known_source_ids)!r}"
            )


@lru_cache(maxsize=1)
def _default_alias_index() -> dict[str, ResolvedRoute]:
    return build_alias_index(get_ontology_config().get("ontologies", {}))


def resolve_retrieval_route(
    route_ontology: str,
    *,
    ontologies_config: dict[str, Any] | None = None,
) -> ResolvedRoute | None:
    """Resolve an already public-route-normalized ontology name to its source.

    Args:
        route_ontology:    Output of public_retriever.public_route_ontology()
                            -- NOT a raw caller-supplied ontology string.
        ontologies_config: Optional `ontologies:` config mapping to resolve
                            against instead of the real shipped
                            ontology_config.yaml. Intended for tests that
                            need to add a fake ontology/source pairing
                            without touching production configuration.

    Returns:
        The ResolvedRoute, or None when no ontology in the config claims
        this alias (i.e. genuinely unsupported).
    """
    key = route_ontology.upper().strip()
    if ontologies_config is not None:
        return build_alias_index(ontologies_config).get(key)
    return _default_alias_index().get(key)
