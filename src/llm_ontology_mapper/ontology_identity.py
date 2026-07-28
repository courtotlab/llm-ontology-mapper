"""Ontology namespace resolution and candidate identity validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "assets" / "ontology_config.yaml"
_CURIE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):(.+)$")
_OBO_SEGMENT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)[_:](.+)$")
_LOINC_LOCAL_RE = re.compile(r"^\d{1,7}-\d$")


@dataclass(frozen=True)
class CandidateIdentity:
    """Resolved candidate identity from authoritative code/IRI/ontology fields."""

    code: str
    ontology: str
    reason: str


@dataclass(frozen=True)
class IdentityValidation:
    """Result of ontology/code consistency validation."""

    valid: bool
    reason: str | None = None


@lru_cache(maxsize=1)
def _ontology_config() -> dict[str, Any]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _prefix_aliases() -> dict[str, str]:
    raw = _ontology_config().get("prefix_aliases", {})
    aliases: dict[str, str] = {}
    if isinstance(raw, dict):
        for alias, ontology in raw.items():
            alias_text = str(alias or "").upper().strip()
            canonical = str(ontology or "").upper().strip()
            if alias_text and canonical:
                aliases[alias_text] = canonical
    return aliases


@lru_cache(maxsize=1)
def _canonical_curie_prefixes() -> dict[str, str]:
    raw = _ontology_config().get("ontologies", {})
    prefixes: dict[str, str] = {}
    if isinstance(raw, dict):
        for ontology, meta in raw.items():
            canonical = canonical_ontology(str(ontology))
            if not canonical or not isinstance(meta, dict):
                continue
            prefix = str(meta.get("curie_prefix") or canonical).upper().strip()
            if prefix:
                prefixes[canonical] = prefix
    prefixes.setdefault("HPO", "HP")
    prefixes.setdefault("MONDO", "MONDO")
    prefixes.setdefault("LOINC", "LOINC")
    return prefixes


@lru_cache(maxsize=1)
def _known_prefixes() -> set[str]:
    prefixes = set(_prefix_aliases())
    prefixes.update(_canonical_curie_prefixes().values())
    return {prefix.upper() for prefix in prefixes if prefix}


def canonical_ontology(value: Any) -> str | None:
    """Return the canonical ontology key for an alias, or None for blank input."""
    text = str(value or "").strip()
    if not text:
        return None
    return _canonical_ontology_text(text)


def ontology_from_code_namespace(code: Any) -> str | None:
    """Resolve ontology from a CURIE/code namespace when one is present."""
    prefix = _code_prefix(code)
    if not prefix:
        return None
    return canonical_ontology(prefix)


def ontology_from_iri_namespace(iri: Any) -> str | None:
    """Resolve ontology from an OBO-style IRI namespace when one is present."""
    curie = curie_from_iri(iri)
    return ontology_from_code_namespace(curie)


def curie_from_iri(iri: Any) -> str | None:
    """Extract a CURIE from the final segment of an OBO-style IRI."""
    text = str(iri or "").strip()
    if not text:
        return None
    segment = re.split(r"[/#]", text.rstrip("/#"))[-1]
    match = _OBO_SEGMENT_RE.match(segment)
    if not match:
        return None
    prefix, local = match.groups()
    ontology = canonical_ontology(prefix)
    if not ontology:
        return None
    return f"{curie_prefix_for_ontology(ontology)}:{local.strip()}"


def curie_prefix_for_ontology(ontology: Any) -> str:
    """Return the configured CURIE prefix for an ontology key or alias."""
    canonical = canonical_ontology(ontology)
    if not canonical:
        return str(ontology or "").upper().strip()
    return _canonical_curie_prefixes().get(canonical, canonical)


def normalize_code_for_ontology(raw_code: Any, ontology: Any) -> str:
    """Normalize a candidate code without overriding an existing authoritative namespace."""
    text = str(raw_code or "").strip().replace("_", ":")
    if not text:
        return text

    iri_curie = curie_from_iri(text)
    if iri_curie:
        return iri_curie

    match = _CURIE_RE.match(text)
    if match:
        prefix, local = match.groups()
        code_ontology = canonical_ontology(prefix)
        if code_ontology:
            return f"{curie_prefix_for_ontology(code_ontology)}:{local.strip()}"
        return f"{prefix.upper().strip()}:{local.strip()}"

    prefix = curie_prefix_for_ontology(ontology)
    return f"{prefix}:{text}" if prefix else text


def resolve_candidate_identity(
    *,
    code: Any,
    iri: Any = None,
    explicit_ontology: Any = None,
    fallback_ontology: Any = None,
) -> CandidateIdentity | None:
    """Resolve candidate ontology/code from code, IRI, explicit ontology, then fallback."""
    raw_code = str(code or "").strip()
    raw_iri = str(iri or "").strip()
    iri_curie = curie_from_iri(raw_iri)

    code_for_resolution = curie_from_iri(raw_code) or raw_code.replace("_", ":")
    code_ontology = ontology_from_code_namespace(code_for_resolution)
    iri_ontology = ontology_from_code_namespace(iri_curie)

    if code_ontology and iri_ontology and code_ontology != iri_ontology:
        return None

    explicit = canonical_ontology(explicit_ontology)
    fallback = canonical_ontology(fallback_ontology)
    ontology = code_ontology or iri_ontology or explicit or fallback
    if not ontology:
        return None

    if code_ontology:
        normalized_code = normalize_code_for_ontology(code_for_resolution, ontology)
        reason = "code_namespace"
    elif iri_curie:
        normalized_code = iri_curie
        reason = "iri_namespace"
    elif raw_code:
        normalized_code = normalize_code_for_ontology(raw_code, ontology)
        reason = "explicit_ontology" if explicit else "fallback_ontology"
    else:
        return None

    validation = validate_candidate_identity(
        ontology=ontology,
        code=normalized_code,
        iri=raw_iri or None,
    )
    if not validation.valid:
        return None

    return CandidateIdentity(code=normalized_code, ontology=ontology, reason=reason)


def validate_candidate_identity(
    *,
    ontology: Any,
    code: Any,
    iri: Any = None,
) -> IdentityValidation:
    """Validate that ontology, code, and optional IRI describe one namespace."""
    canonical = canonical_ontology(ontology)
    code_text = str(code or "").strip()
    if not canonical:
        return IdentityValidation(False, "missing ontology")
    if canonical == "UNKNOWN" and code_text.upper() == "UNKNOWN:UNMAPPED":
        return IdentityValidation(True)
    if not code_text:
        return IdentityValidation(False, "missing code")

    if _has_nested_curie_prefix(code_text):
        return IdentityValidation(False, "nested CURIE namespace")

    code_ontology = ontology_from_code_namespace(code_text)
    iri_ontology = ontology_from_iri_namespace(iri)
    if code_ontology and iri_ontology and code_ontology != iri_ontology:
        return IdentityValidation(False, "code namespace conflicts with IRI namespace")
    if iri_ontology and iri_ontology != canonical:
        return IdentityValidation(False, "IRI namespace conflicts with ontology")

    if canonical == "HPO":
        if code_ontology != "HPO" or _code_prefix(code_text) != "HP":
            return IdentityValidation(False, "HPO candidates require HP codes")
    elif canonical == "MONDO":
        if code_ontology != "MONDO" or _code_prefix(code_text) != "MONDO":
            return IdentityValidation(False, "MONDO candidates require MONDO codes")
    elif canonical == "LOINC":
        local = code_text.split(":", 1)[1] if code_ontology else code_text
        if code_ontology not in (None, "LOINC") or not _LOINC_LOCAL_RE.match(local):
            return IdentityValidation(False, "LOINC candidates require valid LOINC identifiers")
    elif code_ontology and code_ontology != canonical:
        return IdentityValidation(False, "code namespace conflicts with ontology")

    return IdentityValidation(True)


def _canonical_ontology_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    return _prefix_aliases().get(upper, upper)


def _code_prefix(code: Any) -> str | None:
    text = str(code or "").strip().replace("_", ":")
    match = _CURIE_RE.match(text)
    if not match:
        return None
    prefix = match.group(1).upper().strip()
    return prefix or None


def _has_nested_curie_prefix(code: str) -> bool:
    match = _CURIE_RE.match(code.strip().replace("_", ":"))
    if not match:
        return False
    local = match.group(2).strip()
    nested = _CURIE_RE.match(local)
    return bool(nested and nested.group(1).upper() in _known_prefixes())
