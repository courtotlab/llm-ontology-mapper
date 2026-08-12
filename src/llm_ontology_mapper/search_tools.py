"""
SearchTools — stateless ontology search adapters.

Carries only the configuration needed to call the four public APIs:
  • EBI OLS4  (HPO, MONDO, NCIT, UBERON, CHEBI, …)
  • LOINC Search API
  • RxNav (RxNorm)
  • NIH Clinical Tables (ICD-10-CM)

No loop state (ontologies list, SapBERT, cache, NER) lives here.
OntologyRetriever holds those concerns and delegates search calls to this class.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

import requests  # type: ignore[import-untyped]
from requests.auth import HTTPBasicAuth  # type: ignore[import-untyped]

from llm_ontology_mapper.ontology_identity import (
    normalize_code_for_ontology,
    resolve_candidate_identity,
)

logger = logging.getLogger(__name__)


class SearchTools:
    """
    Stateless search adapters for four public ontology APIs.

    Args:
        ols_base:       Base URL for EBI OLS4 (default: https://www.ebi.ac.uk/ols4/api)
        rxnav_base_url: Base URL for RxNav   (default: https://rxnav.nlm.nih.gov/REST)
        icd10_nih_url:  URL for NIH ICD-10-CM search endpoint
        loinc_search_url: LOINC Search API endpoint
        loinc_fhir_url: Base URL retained for backward compatibility
        loinc_username:  HTTP Basic Auth username for LOINC APIs. Falls back
                         to LOINC_USERNAME.
        loinc_password:  HTTP Basic Auth password for LOINC APIs. Falls back
                         to LOINC_PASSWORD.
        api_timeout:    Per-request timeout in seconds
        request_delay:  Sleep between consecutive requests (rate-limiting courtesy)
    """

    OLS_ONTOLOGY_MAP: dict[str, str] = {
        "MONDO": "mondo",
        "HP": "hp",
        "HPO": "hp",
        "NCIT": "ncit",
        "SNOMED": "snomed",
        "SNOMEDCT": "snomed",
        "UO": "uo",
        "UBERON": "uberon",
        "CHEBI": "chebi",
        "GO": "go",
        "DOID": "doid",
        "MESH": "mesh",
        "EFO": "efo",
    }

    def __init__(
        self,
        ols_base: str = "https://www.ebi.ac.uk/ols4/api",
        rxnav_base_url: str = "https://rxnav.nlm.nih.gov/REST",
        icd10_nih_url: str = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
        loinc_fhir_url: str = "https://fhir.loinc.org",
        loinc_fhir_user: str | None = None,
        loinc_fhir_pass: str | None = None,
        api_timeout: int = 10,
        request_delay: float = 0.3,
        loinc_username: str | None = None,
        loinc_password: str | None = None,
        loinc_search_url: str = "https://loinc.regenstrief.org/searchapi/loincs",
    ) -> None:
        self.ols_base = ols_base
        self.rxnav_base_url = rxnav_base_url
        self.icd10_nih_url = icd10_nih_url
        self.loinc_fhir_url = loinc_fhir_url
        self.loinc_search_url = loinc_search_url
        self.loinc_username = (
            loinc_username
            if loinc_username is not None
            else loinc_fhir_user
            if loinc_fhir_user is not None
            else os.environ.get("LOINC_USERNAME") or os.environ.get("LOINC_FHIR_USER")
        )
        self.loinc_password = (
            loinc_password
            if loinc_password is not None
            else loinc_fhir_pass
            if loinc_fhir_pass is not None
            else os.environ.get("LOINC_PASSWORD") or os.environ.get("LOINC_FHIR_PASS")
        )
        # Backward-compatible aliases for callers that inspect these attributes.
        self.loinc_fhir_user = self.loinc_username
        self.loinc_fhir_pass = self.loinc_password
        self.api_timeout = api_timeout
        self.request_delay = request_delay

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search EBI OLS4 for *query* within *ontology*."""
        ontology_id = self.OLS_ONTOLOGY_MAP.get(ontology.upper())
        if not ontology_id:
            logger.warning(f"Ontology {ontology} not supported by OLS")
            return []
        try:
            response = requests.get(
                f"{self.ols_base}/search",
                params={
                    "q": query,
                    "ontology": ontology_id,
                    "rows": str(top_k),
                    "exact": "false",
                    "groupField": "iri",
                    "start": "0",
                },
                timeout=self.api_timeout,
            )
            response.raise_for_status()
            docs = response.json().get("response", {}).get("docs", [])
            candidates = []
            for idx, doc in enumerate(docs[:top_k]):
                obo_id = doc.get("obo_id")
                iri = doc.get("iri", "")
                code = obo_id if obo_id else iri.split("/")[-1].replace("_", ":")
                identity = resolve_candidate_identity(
                    code=code,
                    iri=iri,
                    explicit_ontology=(
                        doc.get("ontology")
                        or doc.get("ontology_id")
                        or doc.get("ontology_name")
                        or doc.get("ontology_prefix")
                    ),
                    fallback_ontology=ontology,
                )
                if identity is None:
                    logger.debug(
                        "OLS result skipped because ontology/code identity could not "
                        "be resolved consistently. requested_ontology=%r iri=%r code=%r",
                        ontology,
                        iri,
                        code,
                    )
                    continue
                term = doc.get("label", "")
                _desc = doc.get("description", "")
                definition = (_desc[0] if isinstance(_desc, list) and _desc else _desc) or ""
                candidates.append(
                    {
                        "code": identity.code,
                        "term": term,
                        "ontology": identity.ontology,
                        "score": 1.0 - (idx * 0.05),
                        "definition": definition,
                        "source": "OLS",
                        "iri": iri,
                        "identity_resolution": identity.reason,
                    }
                )
            time.sleep(self.request_delay)
            return candidates
        except requests.exceptions.RequestException as e:
            logger.error(f"OLS API error for {ontology}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing OLS results: {e}")
            return []

    def search_loinc(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search LOINC codes using the official LOINC Search API."""
        if not self.loinc_username or not self.loinc_password:
            logger.warning(
                "LOINC credentials are required for live LOINC API search. "
                "Set LOINC_USERNAME and LOINC_PASSWORD."
            )
            return []

        try:
            response = requests.get(
                self.loinc_search_url,
                params={"query": query, "rows": str(top_k), "offset": "0"},
                auth=HTTPBasicAuth(self.loinc_username, self.loinc_password),
                timeout=self.api_timeout,
            )
            if response.status_code in (401, 403):
                logger.error(
                    "LOINC Search API authentication failed (HTTP %s). "
                    "Check the configured LOINC service credentials.",
                    response.status_code,
                )
                return []
            if response.status_code == 400:
                logger.error("LOINC Search API rejected the search request (HTTP 400).")
                return []
            response.raise_for_status()
            candidates: list[dict[str, Any]] = []
            for idx, concept in enumerate(self._loinc_results(response.json())[:top_k]):
                code = self._loinc_value(concept, "loincNum", "LOINC_NUM", "loincNumber", "code")
                if not code:
                    continue
                term = self._loinc_value(
                    concept,
                    "longCommonName",
                    "LONG_COMMON_NAME",
                    "displayName",
                    "display",
                    "name",
                    "component",
                )
                definition = self._loinc_value(
                    concept, "definition", "definitionDescription", "shortName"
                )
                candidates.append(
                    {
                        "code": f"LOINC:{code}" if not code.startswith("LOINC:") else code,
                        "term": term,
                        "ontology": "LOINC",
                        "score": 1.0 - (idx * 0.05),
                        "definition": definition,
                        "source": "LOINC-Search-API",
                    }
                )
            time.sleep(self.request_delay)
            return candidates
        except requests.exceptions.RequestException as exc:
            logger.error("LOINC search request failed: %s", exc)
            return []
        except Exception as exc:
            logger.error("Error parsing LOINC search results: %s", exc)
            return []

    @staticmethod
    def _loinc_results(payload: Any) -> list[dict[str, Any]]:
        """Extract result rows from common LOINC Search API response wrappers."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in (
            "results",
            "Results",
            "loincs",
            "docs",
            "items",
            "content",
            "data",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = SearchTools._loinc_results(value)
                if nested:
                    return nested
        for key in ("response", "Response"):
            response = payload.get(key)
            if isinstance(response, dict):
                nested = SearchTools._loinc_results(response)
                if nested:
                    return nested
        return []

    @staticmethod
    def _loinc_value(row: dict[str, Any], *field_names: str) -> str:
        """Read a LOINC field while tolerating case and separator variants."""
        normalized = {
            "".join(ch for ch in str(key).lower() if ch.isalnum()): value
            for key, value in row.items()
        }
        for field_name in field_names:
            value = normalized.get("".join(ch for ch in field_name.lower() if ch.isalnum()))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def search_rxnorm(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search RxNav for *query* using the approximateTerm endpoint.

        approximateTerm.json returns one record per source atom, not per concept.
        Multiple atoms can share the same rxcui (e.g. GS and NDDF atoms have no
        'name' field).  We group by rxcui, pick the best available name from within
        the same response (preferring the RXNORM-source atom), and emit one candidate
        per unique concept.  maxEntries is inflated so that dedup still yields up to
        top_k distinct concepts.
        """
        max_entries = max(top_k * 4, 20)
        try:
            response = requests.get(
                f"{self.rxnav_base_url}/approximateTerm.json",
                params={"term": query, "maxEntries": str(max_entries)},
                timeout=self.api_timeout,
            )
            response.raise_for_status()
            atom_list = response.json().get("approximateGroup", {}).get("candidate", [])
            if not isinstance(atom_list, list):
                atom_list = [atom_list] if atom_list else []

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for atom in atom_list:
                rxcui = (atom.get("rxcui") or "").strip()
                if rxcui:
                    grouped[rxcui].append(atom)

            candidates: list[dict[str, Any]] = []
            for rxcui, atoms in grouped.items():
                name = _rxnorm_best_name(atoms)
                if not name:
                    logger.debug(
                        "search_rxnorm: skipping rxcui=%s — no named atom in response",
                        rxcui,
                    )
                    continue
                rank = _rxnorm_best_rank(atoms)
                candidates.append(
                    {
                        "code": f"RXNORM:{rxcui}",
                        "term": name,
                        "score": 1.0 / (1.0 + rank),
                        "definition": "",
                        "source": "RxNav",
                    }
                )

            candidates.sort(key=lambda c: float(c["score"]), reverse=True)
            time.sleep(self.request_delay)
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"RxNorm search error: {e}")
            return []

    def search_icd10(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search NIH Clinical Tables ICD-10-CM endpoint."""
        try:
            response = requests.get(
                self.icd10_nih_url,
                params={"sf": "code,name", "terms": query, "maxList": str(top_k)},
                timeout=self.api_timeout,
            )
            response.raise_for_status()
            data = response.json()
            candidates = []
            if len(data) >= 4 and isinstance(data[3], list):
                for idx, item in enumerate(data[3][:top_k]):
                    if isinstance(item, list) and len(item) >= 2:
                        candidates.append(
                            {
                                "code": f"ICD10:{item[0]}",
                                "term": item[1],
                                "score": 1.0 - (idx * 0.05),
                                "definition": "",
                                "source": "NIH-ClinicalTables",
                            }
                        )
            time.sleep(self.request_delay)
            return candidates
        except Exception as e:
            logger.error(f"ICD-10 search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Code normalisation helper (needed by search_ols and agentic engine)
    # ------------------------------------------------------------------

    def _normalize_code(self, raw_code: str, ontology: str) -> str:
        return normalize_code_for_ontology(raw_code, ontology)


# ── Module-level helpers for search_rxnorm ───────────────────────────────────


def _rxnorm_best_name(atoms: list[dict[str, Any]]) -> str:
    """Return the best display name from a group of atoms for the same rxcui.

    Prefers the atom from the RXNORM source vocabulary; falls back to the first
    atom that carries any non-blank name.
    """
    rxnorm_atom_first = sorted(
        atoms,
        key=lambda a: 0 if a.get("source") == "RXNORM" else 1,
    )
    for atom in rxnorm_atom_first:
        name = (atom.get("name") or "").strip()
        if name:
            return name
    return ""


def _rxnorm_best_rank(atoms: list[dict[str, Any]]) -> float:
    """Return the minimum (best) rank across all atoms for the same rxcui."""
    ranks: list[float] = []
    for atom in atoms:
        try:
            ranks.append(float(atom.get("rank") or 0))
        except (TypeError, ValueError):
            continue
    return min(ranks) if ranks else 0.0
