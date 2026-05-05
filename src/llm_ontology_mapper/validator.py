"""
OntologyValidator — standalone code-existence checker against live ontology APIs.

Validates whether an ontology code (CURIE) actually exists in its source database.
Used by OntologyMappingEvaluator to distinguish between two failure modes:

  WRONG       — the code exists in the ontology but is not the correct mapping
  HALLUCINATED — the code does not exist at all (LLM invented it)

Supported ontologies and their backing APIs:

  HP, MONDO, NCIT, SNOMED, UO, UBERON, CHEBI, GO, DOID, MESH  → EBI OLS4
  LOINC                                                         → LOINC FHIR
  RXNORM / RXCUI                                                → RxNav
  ICD10 / ICD10CM                                               → NIH Clinical Tables

All results are cached in memory (dict) to minimise repeated HTTP calls.
The cache is per-instance and is not persisted across runs.

Requirements:
    pip install requests
    # Or install the 'eval' extra: pip install 'llm-ontology-mapper[eval]'
"""

from __future__ import annotations

import logging
import os
import urllib.parse

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore[import-untyped]
    from requests.auth import HTTPBasicAuth  # type: ignore[import-untyped]
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# OLS ontology routing table
# ─────────────────────────────────────────────────────────────────────────────

_OLS_ONTOLOGY_MAP: dict[str, str] = {
    "HP":         "hp",
    "HPO":        "hp",
    "MONDO":      "mondo",
    "NCIT":       "ncit",
    "SNOMED":     "snomed",
    "SNOMEDCT":   "snomed",
    "SNOMED-CT":  "snomed",
    "UO":         "uo",
    "UBERON":     "uberon",
    "CHEBI":      "chebi",
    "GO":         "go",
    "DOID":       "doid",
    "MESH":       "mesh",
}

_OBO_IRI_PREFIX = "http://purl.obolibrary.org/obo/"

_OLS_IRI_BASE: dict[str, str] = {
    "hp":     _OBO_IRI_PREFIX,
    "mondo":  _OBO_IRI_PREFIX,
    "ncit":   _OBO_IRI_PREFIX,
    "uberon": _OBO_IRI_PREFIX,
    "chebi":  _OBO_IRI_PREFIX,
    "go":     _OBO_IRI_PREFIX,
    "doid":   _OBO_IRI_PREFIX,
    "uo":     _OBO_IRI_PREFIX,
    "mesh":   "http://id.nlm.nih.gov/mesh/",
    "snomed": "http://snomed.info/id/",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class OntologyValidator:
    """
    Validate whether an ontology CURIE exists in its source database.

    Args:
        cache_enabled:  Cache lookup results in memory (default: True).
        api_timeout:    HTTP request timeout in seconds (default: 10).
        loinc_fhir_url: LOINC FHIR server base URL.
                        Falls back to env ``LOINC_FHIR_URL`` then
                        ``https://fhir.loinc.org``.
        loinc_fhir_user: LOINC FHIR username for authenticated requests.
                         Falls back to env ``LOINC_FHIR_USER``.
        loinc_fhir_pass: LOINC FHIR password.
                         Falls back to env ``LOINC_FHIR_PASS``.

    Example::

        from llm_ontology_mapper.validator import OntologyValidator

        v = OntologyValidator()
        print(v.validate_code("HP:0002110"))   # True
        print(v.validate_code("HP:9999999"))   # False
        print(v.validate_code("HP:FAKE"))      # False
    """

    # Public API endpoints (overridable for testing)
    OLS_BASE     = "https://www.ebi.ac.uk/ols4/api"
    RXNAV_BASE   = "https://rxnav.nlm.nih.gov/REST"
    ICD10_NIH    = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

    def __init__(
        self,
        cache_enabled:  bool = True,
        api_timeout:    int  = 10,
        loinc_fhir_url:  str | None = None,
        loinc_fhir_user: str | None = None,
        loinc_fhir_pass: str | None = None,
    ) -> None:
        self.cache_enabled = cache_enabled
        self.api_timeout   = api_timeout

        self.loinc_fhir_url  = loinc_fhir_url  or os.environ.get("LOINC_FHIR_URL",  "https://fhir.loinc.org")
        self.loinc_fhir_user = loinc_fhir_user or os.environ.get("LOINC_FHIR_USER")
        self.loinc_fhir_pass = loinc_fhir_pass or os.environ.get("LOINC_FHIR_PASS")

        self._cache: dict[str, bool | None] = {}

        if not _REQUESTS_AVAILABLE:
            logger.warning(
                "OntologyValidator: 'requests' library not installed — "
                "all validate_code() calls will return None. "
                "Install it with: pip install requests"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def validate_code(self, ontology_code: str) -> bool | None:
        """
        Check whether ``ontology_code`` exists in its source ontology database.

        Args:
            ontology_code: CURIE string, e.g. ``"HP:0002110"`` or ``"MONDO:0005180"``.

        Returns:
            ``True``  — code confirmed to exist.
            ``False`` — code confirmed not to exist (likely hallucinated).
            ``None``  — validation unavailable (no ``requests``, API timeout, etc.).
        """
        if not _REQUESTS_AVAILABLE:
            return None

        code = str(ontology_code).strip().upper()

        if self.cache_enabled and code in self._cache:
            return self._cache[code]

        result = self._dispatch(code)

        if self.cache_enabled:
            self._cache[code] = result

        return result

    def get_cache_stats(self) -> dict[str, int]:
        """Return summary counts from the in-memory cache."""
        return {
            "cached_codes":  len(self._cache),
            "valid_codes":   sum(1 for v in self._cache.values() if v is True),
            "invalid_codes": sum(1 for v in self._cache.values() if v is False),
            "unknown_codes": sum(1 for v in self._cache.values() if v is None),
        }

    def clear_cache(self) -> None:
        """Evict all cached results."""
        self._cache.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch(self, code: str) -> bool | None:
        """Route a normalised code to the appropriate backend API."""
        # Guard: invalid format or sentinel values
        if not code or ":" not in code or code in {"UNMAPPED", "ERROR", "NONE", "UNKNOWN"}:
            return False

        prefix, code_id = code.split(":", 1)

        if prefix == "LOINC":
            return self._check_loinc(code_id)
        if prefix in {"RXNORM", "RXCUI"}:
            return self._check_rxnorm(code_id)
        if prefix in {"ICD10", "ICD10CM"}:
            return self._check_icd10(code_id)
        if prefix in _OLS_ONTOLOGY_MAP:
            return self._check_ols(prefix, code)

        # Unknown prefix — cannot validate
        logger.debug("OntologyValidator: unknown prefix %r for code %r", prefix, code)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Backend checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_ols(self, prefix: str, full_code: str) -> bool | None:
        """
        Validate a code against EBI OLS4.

        Strategy 1: construct the OBO IRI and look it up directly (fast).
        Strategy 2: fall back to the OLS search endpoint with exact obo_id matching.
        """
        ontology_id = _OLS_ONTOLOGY_MAP.get(prefix)
        if not ontology_id:
            return None

        iri_base     = _OLS_IRI_BASE.get(ontology_id, _OBO_IRI_PREFIX)
        short_code   = full_code.replace(":", "_")          # HP:0002110 → HP_0002110
        full_iri     = f"{iri_base}{short_code}"
        encoded_iri  = urllib.parse.quote(urllib.parse.quote(full_iri, safe=""), safe="")
        direct_url   = f"{self.OLS_BASE}/ontologies/{ontology_id}/terms/{encoded_iri}"

        try:
            resp = requests.get(direct_url, timeout=self.api_timeout)
            if resp.status_code == 200:
                logger.debug("OLS4 ✓ %s", full_code)
                return True

            # Fallback: search by obo_id
            resp = requests.get(
                f"{self.OLS_BASE}/search",
                params={
                    "q":           full_code,
                    "ontology":    ontology_id,
                    "exact":       "true",
                    "queryFields": "obo_id",
                    "rows":        "1",
                },
                timeout=self.api_timeout,
            )
            if resp.status_code == 200:
                docs = resp.json().get("response", {}).get("docs", [])
                if docs:
                    logger.debug("OLS4 search ✓ %s", full_code)
                    return True

            logger.debug("OLS4 ✗ %s", full_code)
            return False

        except requests.exceptions.Timeout:
            logger.warning("OLS4 timeout for %s", full_code)
            return None
        except Exception as exc:
            logger.warning("OLS4 error for %s: %s", full_code, exc)
            return None

    def _check_loinc(self, code_id: str) -> bool | None:
        """Validate a LOINC code via the LOINC FHIR server."""
        fhir_base = self.loinc_fhir_url.rstrip("/")
        url = f"{fhir_base}/CodeSystem/$lookup?system=http://loinc.org&code={code_id}"

        try:
            resp = requests.get(url, timeout=self.api_timeout)

            # Retry with Basic Auth if the server returns 401
            if resp.status_code == 401 and self.loinc_fhir_user and self.loinc_fhir_pass:
                resp = requests.get(
                    url,
                    auth=HTTPBasicAuth(self.loinc_fhir_user, self.loinc_fhir_pass),
                    timeout=self.api_timeout,
                )

            if resp.status_code == 200:
                logger.debug("LOINC ✓ %s", code_id)
                return True

            logger.debug("LOINC ✗ %s (HTTP %s)", code_id, resp.status_code)
            return False

        except requests.exceptions.Timeout:
            logger.warning("LOINC timeout for %s", code_id)
            return None
        except Exception as exc:
            logger.warning("LOINC error for %s: %s", code_id, exc)
            return None

    def _check_rxnorm(self, code_id: str) -> bool | None:
        """Validate an RxCUI via RxNav."""
        url = f"{self.RXNAV_BASE}/rxcui/{code_id}/properties.json"
        try:
            resp = requests.get(url, timeout=self.api_timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "properties" in data:
                    logger.debug("RxNorm ✓ %s", code_id)
                    return True
            logger.debug("RxNorm ✗ %s", code_id)
            return False
        except requests.exceptions.Timeout:
            logger.warning("RxNorm timeout for %s", code_id)
            return None
        except Exception as exc:
            logger.warning("RxNorm error for %s: %s", code_id, exc)
            return None

    def _check_icd10(self, code_id: str) -> bool | None:
        """Validate an ICD-10-CM code via NIH Clinical Tables."""
        try:
            resp = requests.get(
                self.ICD10_NIH,
                params={"sf": "code", "terms": code_id, "maxList": "5"},
                timeout=self.api_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Response: [count, [codes], {synonyms}, [[code, description], …]]
                if len(data) >= 4 and isinstance(data[3], list):
                    normalised_search = code_id.replace(".", "").upper()
                    for item in data[3]:
                        if isinstance(item, list) and item and item[0].replace(".", "").upper() == normalised_search:
                            logger.debug("ICD-10 ✓ %s", code_id)
                            return True
            logger.debug("ICD-10 ✗ %s", code_id)
            return False
        except requests.exceptions.Timeout:
            logger.warning("ICD-10 timeout for %s", code_id)
            return None
        except Exception as exc:
            logger.warning("ICD-10 error for %s: %s", code_id, exc)
            return None
