"""
OntologyRetriever — RAG candidate retrieval from live ontology APIs.

Retrieve candidates from EBI OLS4, LOINC FHIR, RxNav, NIH Clinical Tables,
and SapBERT.  retrieve() returns Tuple[List[Dict], float] where the float is
the score of the top candidate (0.0 if no candidates found).
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from typing import Any

import requests  # type: ignore[import-untyped]

from llm_ontology_mapper.search_tools import SearchTools

logger = logging.getLogger(__name__)

_CLINICAL_ABBREVIATIONS: dict[str, str] = {
    'htn': 'hypertension', 'mi': 'myocardial infarction',
    'chf': 'congestive heart failure', 'cad': 'coronary artery disease',
    'af': 'atrial fibrillation', 'afib': 'atrial fibrillation',
    'pvd': 'peripheral vascular disease', 'pad': 'peripheral artery disease',
    'bp': 'blood pressure', 'hr': 'heart rate', 'lvh': 'left ventricular hypertrophy',
    'dm': 'diabetes mellitus', 'dm1': 'type 1 diabetes mellitus',
    'dm2': 'type 2 diabetes mellitus', 't1dm': 'type 1 diabetes mellitus',
    't2dm': 'type 2 diabetes mellitus', 'bmi': 'body mass index',
    'hba1c': 'glycated hemoglobin', 'ldl': 'low-density lipoprotein',
    'hdl': 'high-density lipoprotein', 'tg': 'triglycerides',
    'tsh': 'thyroid stimulating hormone', 'copd': 'chronic obstructive pulmonary disease',
    'sob': 'shortness of breath', 'ards': 'acute respiratory distress syndrome',
    'rr': 'respiratory rate', 'cva': 'cerebrovascular accident',
    'tia': 'transient ischemic attack', 'ms': 'multiple sclerosis',
    'ckd': 'chronic kidney disease', 'egfr': 'estimated glomerular filtration rate',
    'esrd': 'end stage renal disease', 'uti': 'urinary tract infection',
    'gerd': 'gastroesophageal reflux disease', 'ibd': 'inflammatory bowel disease',
    'ibs': 'irritable bowel syndrome', 'ra': 'rheumatoid arthritis',
    'sle': 'systemic lupus erythematosus', 'oa': 'osteoarthritis',
    'wbc': 'white blood cell count', 'rbc': 'red blood cell count',
    'hgb': 'hemoglobin', 'hb': 'hemoglobin', 'plt': 'platelet count',
    'ast': 'aspartate aminotransferase', 'alt': 'alanine aminotransferase',
    'crp': 'c-reactive protein', 'esr': 'erythrocyte sedimentation rate',
    'hiv': 'human immunodeficiency virus', 'hbv': 'hepatitis b',
    'hcv': 'hepatitis c', 'covid': 'covid-19', 'covid19': 'covid-19',
    'dx': 'diagnosis', 'hx': 'history', 'sx': 'symptoms', 'tx': 'treatment',
    'rx': 'medication', 'fx': 'fracture', 'diag': 'diagnosis',
    'med': 'medication', 'meds': 'medications', 'surg': 'surgery',
    'hosp': 'hospitalization', 'prev': 'previous', 'curr': 'current',
}


class OntologyRetriever:
    """
    Retrieve candidate ontology codes from public APIs for RAG-based mapping.

    Supported endpoints:
      • EBI OLS4  — HP, MONDO, NCIT, UO, SNOMED, …
      • LOINC Search API — LOINC (service credentials required)
      • RxNav      — RxNorm
      • NIH Clinical Tables — ICD-10-CM
      • SapBERT    — semantic search server (optional)

    Low-level search adapters live in SearchTools; this class owns retrieval
    strategy, SapBERT, query preprocessing, NER, and caching concerns.
    """

    def __init__(
        self,
        ontologies: list[str] | None = None,
        top_k: int = 5,
        cache_enabled: bool = True,
        api_timeout: int = 10,
        loinc_fhir_url: str | None = None,
        loinc_fhir_user: str | None = None,
        loinc_fhir_pass: str | None = None,
        # source-compat extras (ignored by new public API but used internally)
        use_cache: bool = True,
        request_delay: float = 0.3,
        ner_extractor: Any | None = None,
        sapbert_url: str | None = None,
        loinc_username: str | None = None,
        loinc_password: str | None = None,
        loinc_search_url: str = "https://loinc.regenstrief.org/searchapi/loincs",
    ) -> None:
        self.ontologies        = ontologies or ["HPO", "MONDO", "NCIT", "LOINC", "UO"]
        self.top_k             = top_k
        self.cache_enabled     = cache_enabled
        self.api_timeout       = api_timeout
        self._cache: dict[str, list[dict[str, Any]]] = {}

        self.use_cache     = cache_enabled or use_cache
        self.request_delay = request_delay
        self.ner_extractor = ner_extractor

        self.sapbert_url = (sapbert_url or os.environ.get("SAPBERT_SERVER_URL", "")).rstrip("/")
        if self.sapbert_url:
            logger.info(f"SapBERT server configured: {self.sapbert_url}")

        self.loinc_fhir_url: str  = loinc_fhir_url  or os.environ.get("LOINC_FHIR_URL", "https://fhir.loinc.org")
        self.loinc_username = (
            loinc_username
            if loinc_username is not None
            else loinc_fhir_user
            if loinc_fhir_user is not None
            else os.environ.get("LOINC_USERNAME")
            or os.environ.get("LOINC_FHIR_USER")
        )
        self.loinc_password = (
            loinc_password
            if loinc_password is not None
            else loinc_fhir_pass
            if loinc_fhir_pass is not None
            else os.environ.get("LOINC_PASSWORD")
            or os.environ.get("LOINC_FHIR_PASS")
        )
        self.loinc_fhir_user = self.loinc_username
        self.loinc_fhir_pass = self.loinc_password

        # Load ontology config (optional yaml next to assets/)
        from pathlib import Path

        import yaml  # type: ignore[import-untyped]
        _here = Path(__file__).parent
        config_path = _here / "assets" / "ontology_config.yaml"
        if not config_path.exists():
            config_path = _here / "assets" / "ontology_config.yaml"
        if config_path.exists():
            with open(config_path) as fh:
                self.ontology_config = yaml.safe_load(fh) or {}
        else:
            self.ontology_config = {}

        self._search_tools = SearchTools(
            ols_base="https://www.ebi.ac.uk/ols4/api",
            rxnav_base_url="https://rxnav.nlm.nih.gov/REST",
            icd10_nih_url="https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
            loinc_fhir_url=self.loinc_fhir_url,
            loinc_username=self.loinc_username,
            loinc_password=self.loinc_password,
            loinc_search_url=loinc_search_url,
            api_timeout=api_timeout,
            request_delay=request_delay,
        )

        # Convenience aliases kept for any callers that read these attrs directly
        self.ols_base       = self._search_tools.ols_base
        self.rxnav_base_url = self._search_tools.rxnav_base_url
        self.icd10_nih_url  = self._search_tools.icd10_nih_url

        self.ols_ontologies = list(SearchTools.OLS_ONTOLOGY_MAP.keys())
        self.ols_ontology_map = dict(SearchTools.OLS_ONTOLOGY_MAP)

        logger.info("OntologyRetriever initialized with public APIs")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        entity_type: str | None = None,
        ontologies: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Search ontology APIs and return ranked candidates.

        Returns:
            (candidates, top_score)
            candidates: List of dicts with keys {code, term, ontology, score}
            top_score:  Highest similarity score in the candidate list (0.0 if empty)
        """
        target_ontologies = ontologies or self.ontologies
        queries = self._preprocess_query(query)

        merged: dict[str, dict] = {}
        for onto in target_ontologies:
            for c in self._search_multi_query(queries, onto, top_k=self.top_k):
                code = c.get("code", "")
                if not code:
                    continue
                if c.get("score", 0) > merged.get(code, {}).get("score", -1):
                    merged[code] = c

        candidates = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[: self.top_k]
        top_score = candidates[0]["score"] if candidates else 0.0
        return candidates, top_score

    # ------------------------------------------------------------------
    # Query preprocessing
    # ------------------------------------------------------------------

    def _expand_abbreviations(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        expanded = []
        for word in words:
            key = word.lower().strip('.,;:()?!\'"/')
            expanded.append(_CLINICAL_ABBREVIATIONS.get(key, word))
        return ' '.join(expanded)

    def _preprocess_query(
        self,
        field_name: str,
        field_label: str | None = None,
    ) -> list[str]:
        queries: list[str] = []

        if self.ner_extractor is not None and self.ner_extractor.is_available():
            for span in self.ner_extractor.extract_entities(field_name, field_label):
                queries.append(self._expand_abbreviations(span))

        if field_label and isinstance(field_label, str):
            label = field_label.strip()

            choice_match = re.search(r'\(choice=([^)]+)\)', label, re.IGNORECASE)
            if choice_match:
                choice_val = choice_match.group(1).strip()
                stem = re.sub(r'\s*\(choice=[^)]*\)', '', label).strip().rstrip('?.,')
                queries.append(self._expand_abbreviations(f"{choice_val} {stem}"))
                queries.append(self._expand_abbreviations(choice_val))
            else:
                eg_match = re.search(r'\(e\.?g\.?,?\s*([^)]+)\)', label, re.IGNORECASE)
                if eg_match:
                    for ex in eg_match.group(1).split(',')[:2]:
                        ex = ex.strip().rstrip('.,')
                        if len(ex) > 3:
                            queries.append(self._expand_abbreviations(ex))

                stripped = re.sub(
                    r'^(?:have you(?: ever)?(?: been)?|do you(?: currently| have)?|'
                    r'has (?:a |your )?(?:doctor|physician)(?: ever)?(?:(?: told you)?(?:(?: that)? you have)?)?|'
                    r'are you(?: currently)?|did you|were you|'
                    r'(?:in the past \d+[- ]?years?[,]?\s*)?(?:have you|did you|were you)|'
                    r'what is (?:your )?|please (?:indicate|select|choose|rate)|'
                    r'how (?:many|much|often)|when did|at what age)\s+',
                    '', label, flags=re.IGNORECASE,
                ).strip().rstrip('?.,')
                stripped = re.sub(r'\s*\([A-Z0-9/-]{1,12}\)\s*$', '', stripped).strip()

                if '/' in stripped:
                    for part in re.split(r'\s*/\s*', stripped):
                        part = part.strip()
                        exp = self._expand_abbreviations(part)
                        if exp and len(exp) > 2:
                            queries.append(exp)

                label_expanded = self._expand_abbreviations(stripped)
                if label_expanded and len(label_expanded) > 2:
                    queries.append(label_expanded)

        elif field_name and isinstance(field_name, str):
            name = re.sub(r'([a-z])([A-Z])', r'\1 \2', field_name)
            name = re.sub(r'[_\-]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()
            name_expanded = self._expand_abbreviations(name)
            if name_expanded and len(name_expanded) > 2:
                queries.append(name_expanded)

        seen: set = set()
        unique: list[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)

        if not unique:
            label_str = field_label if isinstance(field_label, str) else None
            name_str = field_name if isinstance(field_name, str) else None
            fallback = (label_str or name_str or '').strip()
            if fallback:
                unique.append(fallback)

        return unique[:3]

    def _search_multi_query(
        self,
        queries: list[str],
        ontology: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        best_by_code: dict[str, dict] = {}
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                continue
            for candidate in self.search_ontology(query.strip(), ontology, top_k):
                code = candidate.get('code', '')
                if not code:
                    continue
                if candidate.get('score', 0) > best_by_code.get(code, {}).get('score', -1):
                    best_by_code[code] = candidate
        return sorted(best_by_code.values(), key=lambda x: x.get('score', 0), reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def search_ontology(
        self,
        query: str,
        ontology: str,
        top_k: int = 10,
    ) -> list[dict[str, str]]:
        ontology_upper = ontology.upper()

        if self.sapbert_url:
            results = self._search_sapbert(query, ontology_upper, top_k)
            if results:
                return results

        if ontology_upper in self.ols_ontologies:
            return self._search_ols(query, ontology, top_k)
        elif ontology_upper == 'LOINC':
            return self._search_loinc(query, top_k)
        elif ontology_upper in ['RXNORM', 'RXCUI']:
            return self._search_rxnorm(query, top_k)
        elif ontology_upper in ['ICD10', 'ICD10CM']:
            return self._search_icd10(query, top_k)

        return []

    def _search_sapbert(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
        sapbert_map = self.ontology_config.get('sapbert_index_map', {})
        index_key = sapbert_map.get(ontology, sapbert_map.get(ontology.upper()))
        if not index_key:
            _fb = {
                'HP': 'HPO', 'HPO': 'HPO', 'MONDO': 'MONDO', 'NCIT': 'NCIT',
                'LOINC': 'LOINC', 'SNOMED-CT': 'SNOMED', 'SNOMEDCT': 'SNOMED',
                'SNOMED': 'SNOMED', 'RXNORM': 'RXNORM', 'RXCUI': 'RXNORM',
                'ICD10': 'ICD10CM', 'ICD10CM': 'ICD10CM',
            }
            index_key = _fb.get(ontology.upper(), ontology.upper())
        try:
            resp = requests.post(
                f"{self.sapbert_url}/search",
                json={"query": query, "ontology": index_key, "top_k": top_k},
                timeout=self.api_timeout,
            )
            resp.raise_for_status()
            return [
                {"code": i["code"], "term": i["term"],
                 "score": float(i.get("score", 0.0)),
                 "definition": i.get("definition", ""), "source": "SapBERT"}
                for i in resp.json().get("results", [])
            ]
        except requests.exceptions.ConnectionError:
            logger.warning(f"SapBERT server unreachable ({self.sapbert_url}), falling back to OLS")
            return []
        except Exception as e:
            logger.warning(f"SapBERT search error: {e}")
            return []

    def _search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, str]]:
        return self._search_tools.search_ols(query, ontology, top_k)

    def _search_loinc(self, query: str, top_k: int = 10) -> list[dict[str, str]]:
        return self._search_tools.search_loinc(query, top_k)

    def _search_rxnorm(self, query: str, top_k: int = 10) -> list[dict[str, str]]:
        return self._search_tools.search_rxnorm(query, top_k)

    def _search_icd10(self, query: str, top_k: int = 10) -> list[dict[str, str]]:
        return self._search_tools.search_icd10(query, top_k)

    # ------------------------------------------------------------------
    # Validation methods
    #
    # NOTE: These methods return (exists, label) tuples because the
    # retrieval pipeline needs the term label for candidate ranking.
    # For pure existence-checking in the evaluation pipeline, use
    # OntologyValidator from validator.py instead — it is lighter
    # (no label extraction, BioPortal, or per-call rate-limiting)
    # and adds in-memory caching.
    # ------------------------------------------------------------------

    def validate_code(
        self,
        code: str,
        ontology: str,
        iri: str | None = None,
    ) -> tuple[bool | None, str | None]:
        """
        Check whether *code* exists in *ontology* and return its label.

        Returns:
            (True, label)   — code confirmed to exist; label may be empty string.
            (False, None)   — code confirmed not to exist.
            (None, None)    — result unknown (API error / timeout).
        """
        full_code = self._normalize_code(code, ontology)
        ontology_upper = ontology.upper()
        if ontology_upper in self.ols_ontologies:
            return self._validate_ols(full_code, ontology, iri=iri)
        elif ontology_upper == 'LOINC':
            return self._validate_loinc(code.split(':')[-1])
        elif ontology_upper in ['RXNORM', 'RXCUI']:
            return self._validate_rxnorm(code.split(':')[-1])
        elif ontology_upper in ['ICD10', 'ICD10CM']:
            return self._validate_icd10(code.split(':')[-1])
        return None, None

    def _validate_ols(self, full_code: str, ontology: str, iri: str | None = None) -> tuple[bool | None, str | None]:
        ontology_id = self.ols_ontology_map.get(ontology.upper())
        if not ontology_id:
            return None, None
        _iri_prefix = {
            'hp': 'http://purl.obolibrary.org/obo/', 'mondo': 'http://purl.obolibrary.org/obo/',
            'ncit': 'http://purl.obolibrary.org/obo/', 'uberon': 'http://purl.obolibrary.org/obo/',
            'chebi': 'http://purl.obolibrary.org/obo/', 'go': 'http://purl.obolibrary.org/obo/',
            'doid': 'http://purl.obolibrary.org/obo/', 'uo': 'http://purl.obolibrary.org/obo/',
            'mesh': 'http://id.nlm.nih.gov/mesh/', 'snomed': 'http://snomed.info/id/',
        }

        def _ols_lookup(candidate_iri: str) -> tuple[bool | None, str | None]:
            encoded = urllib.parse.quote(urllib.parse.quote(candidate_iri, safe=''), safe='')
            resp = requests.get(
                f"{self.ols_base}/ontologies/{ontology_id}/terms/{encoded}",
                timeout=self.api_timeout,
            )
            if resp.status_code == 200:
                return True, resp.json().get('label', '')
            return None, None

        try:
            if iri:
                ok, term = _ols_lookup(iri)
                if ok:
                    time.sleep(self.request_delay)
                    return True, term
            prefix = _iri_prefix.get(ontology_id, 'http://purl.obolibrary.org/obo/')
            constructed_iri = f"{prefix}{full_code.replace(':', '_')}"
            ok, term = _ols_lookup(constructed_iri)
            if ok:
                time.sleep(self.request_delay)
                return True, term
            resp = requests.get(
                f"{self.ols_base}/search",
                params={'q': full_code, 'ontology': ontology_id, 'exact': 'true',
                        'queryFields': 'obo_id', 'rows': '1'},
                timeout=self.api_timeout,
            )
            if resp.status_code == 200:
                docs = resp.json().get('response', {}).get('docs', [])
                if docs:
                    time.sleep(self.request_delay)
                    return True, docs[0].get('label', '')
            time.sleep(self.request_delay)
            return False, None
        except Exception as e:
            logger.error(f"Error validating {full_code}: {e}")
            return None, None

    def _validate_loinc(self, code: str) -> tuple[bool | None, str | None]:
        try:
            auth = None
            if self.loinc_fhir_user and self.loinc_fhir_pass:
                from requests.auth import HTTPBasicAuth  # type: ignore[import-untyped]
                auth = HTTPBasicAuth(self.loinc_fhir_user, self.loinc_fhir_pass)
            response = requests.get(
                f"{self.loinc_fhir_url.rstrip('/')}/CodeSystem/$lookup?system=http://loinc.org&code={code}",
                auth=auth, timeout=self.api_timeout,
            )
            if response.status_code == 200:
                term = response.json().get('parameter', [{}])[0].get('valueString', '')
                time.sleep(self.request_delay)
                return True, term
            time.sleep(self.request_delay)
            return False, None
        except Exception as e:
            logger.error(f"LOINC validation error for {code}: {e}")
            return None, None

    def _validate_rxnorm(self, code: str) -> tuple[bool | None, str | None]:
        try:
            response = requests.get(f"{self.rxnav_base_url}/rxcui/{code}/properties.json", timeout=self.api_timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'properties' in data:
                    time.sleep(self.request_delay)
                    return True, data['properties'].get('name', '')
            time.sleep(self.request_delay)
            return False, None
        except Exception as e:
            logger.error(f"RxNorm validation error for {code}: {e}")
            return None, None

    def _validate_icd10(self, code: str) -> tuple[bool | None, str | None]:
        try:
            response = requests.get(
                self.icd10_nih_url,
                params={'sf': 'code', 'terms': code, 'maxList': '5'},
                timeout=self.api_timeout,
            )
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 4 and isinstance(data[3], list):
                    for item in data[3]:
                        if isinstance(item, list) and len(item) >= 2 and item[0].replace('.', '') == code.replace('.', ''):
                            time.sleep(self.request_delay)
                            return True, item[1]
            time.sleep(self.request_delay)
            return False, None
        except Exception as e:
            logger.error(f"ICD-10 validation error for {code}: {e}")
            return None, None

    def _normalize_code(self, raw_code: str, ontology: str) -> str:
        return self._search_tools._normalize_code(raw_code, ontology)

    def batch_validate_codes(self, codes: list[str], ontology: str) -> dict[str, tuple[bool | None, str | None]]:
        return {code: self.validate_code(code, ontology) for code in codes}
