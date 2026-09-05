"""
Scenario 2 (retrieval-mode ablation) hallucination validation (Part 17/18).

Hallucination != semantic incorrectness: it means the emitted ontology code
itself cannot be confirmed to exist in its claimed ontology, using the
existing llm_ontology_mapper.validator.OntologyValidator (live HTTP calls to
EBI OLS4 / LOINC FHIR / RxNav / NIH Clinical Tables). Never validates
UNKNOWN:UNMAPPED and never validates unmapped/error rows.

Results are cached by normalized CURIE (across rows AND across modes when a
caller shares one cache) so repeated codes never repeatedly hit external
services. The cache is persisted to validation_cache.csv so hallucination
analysis is re-runnable later via --evaluate-existing with ZERO mapper/LLM
calls, and can be resumed if the validator service was temporarily down
(Part 18) -- an UNRESOLVED code is simply retried on the next call rather
than cached as a permanent failure.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

VALID = "VALID"
INVALID = "INVALID"
UNRESOLVED = "UNRESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"  # unmapped / execution-error rows -- never validated

UNMAPPED_SENTINEL = "UNKNOWN:UNMAPPED"
STATUS_MAPPED = "mapped"

# Deterministic routing used only to LABEL which backend a code's validation
# used -- OntologyValidator itself owns the actual dispatch/HTTP logic; this
# mirrors its routing table (validator._OLS_ONTOLOGY_MAP union the LOINC/
# RxNorm/ICD10 special cases) for the seven ontology families in this
# workbook (HPO, MONDO, LOINC, ICD10, SNOMED, NCIT, RxNorm).
_OLS_PREFIXES = {"HP", "MONDO", "NCIT", "SNOMED", "SNOMEDCT", "SNOMED-CT", "UO", "UBERON", "CHEBI", "GO", "DOID", "MESH"}


def validation_source_for_code(code: str) -> str:
    prefix = code.split(":", 1)[0].strip().upper() if ":" in code else code.strip().upper()
    if prefix == "LOINC":
        return "LOINC-FHIR"
    if prefix in {"RXNORM", "RXCUI"}:
        return "RxNav"
    if prefix in {"ICD10", "ICD10CM"}:
        return "NIH-ClinicalTables"
    if prefix in _OLS_PREFIXES:
        return "OLS4"
    return "UNKNOWN"


class SupportsValidateCode(Protocol):
    def validate_code(self, ontology_code: str) -> bool | None: ...


@dataclass(frozen=True)
class ValidationOutcome:
    row_id: int
    validation_status: str  # VALID | INVALID | UNRESOLVED | NOT_APPLICABLE
    validation_source: str | None


ValidationCache = dict[str, tuple[str, str]]  # normalized CURIE -> (status, source)


def read_validation_cache(path: Path) -> ValidationCache:
    cache: ValidationCache = {}
    if not path.exists():
        return cache
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            curie = row.get("curie")
            status = row.get("status")
            source = row.get("source") or ""
            if curie and status:
                cache[curie] = (status, source)
    return cache


def write_validation_cache(cache: ValidationCache, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("curie", "status", "source"))
        writer.writeheader()
        for curie, (status, source) in sorted(cache.items()):
            writer.writerow({"curie": curie, "status": status, "source": source})


def validate_one(
    *,
    status: str,
    mapped_code: str | None,
    validator: SupportsValidateCode,
    cache: ValidationCache,
) -> tuple[str, str | None]:
    """Validate a single row's mapped code, consulting/populating `cache` by
    normalized CURIE. Returns (validation_status, validation_source).

    UNRESOLVED results are cached the same as VALID/INVALID (Part 18: a
    temporarily-down service should be retried on a LATER --evaluate-existing
    call, not treated as permanently unresolved) -- callers who want to force
    a retry of previously-UNRESOLVED codes should drop those cache entries
    before calling this function, e.g. via a filtered validation_cache.csv.
    """
    if status != STATUS_MAPPED:
        return NOT_APPLICABLE, None
    if not mapped_code:
        return NOT_APPLICABLE, None
    normalized = mapped_code.strip().upper()
    if normalized == UNMAPPED_SENTINEL:
        return NOT_APPLICABLE, None

    if normalized in cache:
        return cache[normalized]

    raw_result = validator.validate_code(normalized)
    if raw_result is True:
        validation_status = VALID
    elif raw_result is False:
        validation_status = INVALID
    else:
        validation_status = UNRESOLVED
    source = validation_source_for_code(normalized)
    cache[normalized] = (validation_status, source)
    return validation_status, source


@dataclass(frozen=True)
class HallucinationSummary:
    mapped_count: int
    valid_count: int
    invalid_count: int
    unresolved_count: int
    not_applicable_count: int

    @property
    def hallucination_rate(self) -> float | None:
        denom = self.valid_count + self.invalid_count
        return self.invalid_count / denom if denom else None

    @property
    def validation_coverage(self) -> float | None:
        return (self.valid_count + self.invalid_count) / self.mapped_count if self.mapped_count else None

    @property
    def unresolved_validation_count(self) -> int:
        return self.unresolved_count

    @property
    def unresolved_validation_rate(self) -> float | None:
        return self.unresolved_count / self.mapped_count if self.mapped_count else None


def summarize_hallucination(
    *, mapped_count: int, validation_statuses: list[str]
) -> HallucinationSummary:
    return HallucinationSummary(
        mapped_count=mapped_count,
        valid_count=sum(1 for s in validation_statuses if s == VALID),
        invalid_count=sum(1 for s in validation_statuses if s == INVALID),
        unresolved_count=sum(1 for s in validation_statuses if s == UNRESOLVED),
        not_applicable_count=sum(1 for s in validation_statuses if s == NOT_APPLICABLE),
    )
