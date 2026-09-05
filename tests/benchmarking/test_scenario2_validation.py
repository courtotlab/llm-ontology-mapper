"""
Scenario 2 hallucination-validation tests (Part 30, items 36-41).

Uses a fake validator (no real HTTP/OntologyValidator calls) implementing the
SupportsValidateCode protocol, so these tests never hit EBI OLS4 / LOINC FHIR
/ RxNav / NIH Clinical Tables.
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scenario2_validation import (
    INVALID,
    NOT_APPLICABLE,
    UNRESOLVED,
    VALID,
    ValidationCache,
    summarize_hallucination,
    validate_one,
)

pytestmark = pytest.mark.unit


class _FakeValidator:
    """Records every validate_code() call so cache-reuse can be asserted."""

    def __init__(self, responses: dict[str, bool | None]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def validate_code(self, ontology_code: str) -> bool | None:
        self.calls.append(ontology_code)
        return self._responses.get(ontology_code)


# ─────────────────────────────────────────────────────────────────────────────
# 36. validator True -> VALID
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_true_yields_valid() -> None:
    validator = _FakeValidator({"HP:0002110": True})
    cache: ValidationCache = {}
    status, source = validate_one(status="mapped", mapped_code="HP:0002110", validator=validator, cache=cache)
    assert status == VALID
    assert source == "OLS4"


# ─────────────────────────────────────────────────────────────────────────────
# 37. validator False -> INVALID (hallucinated)
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_false_yields_invalid() -> None:
    validator = _FakeValidator({"HP:9999999": False})
    cache: ValidationCache = {}
    status, _ = validate_one(status="mapped", mapped_code="HP:9999999", validator=validator, cache=cache)
    assert status == INVALID


# ─────────────────────────────────────────────────────────────────────────────
# 38. validator unresolved (None) -> UNRESOLVED, not counted as hallucinated
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_none_yields_unresolved_not_hallucinated() -> None:
    validator = _FakeValidator({"LOINC:1234-5": None})
    cache: ValidationCache = {}
    status, _ = validate_one(status="mapped", mapped_code="LOINC:1234-5", validator=validator, cache=cache)
    assert status == UNRESOLVED
    assert status != INVALID


# ─────────────────────────────────────────────────────────────────────────────
# 39. unresolved affects coverage but not hallucination_rate denominator
# ─────────────────────────────────────────────────────────────────────────────


def test_unresolved_reduces_coverage_but_excluded_from_hallucination_rate() -> None:
    summary = summarize_hallucination(
        mapped_count=4,
        validation_statuses=[VALID, INVALID, UNRESOLVED, UNRESOLVED],
    )
    assert summary.valid_count == 1
    assert summary.invalid_count == 1
    assert summary.unresolved_count == 2
    # hallucination_rate = INVALID / (VALID + INVALID) = 1/2, never divided by
    # mapped_count directly (UNRESOLVED rows are excluded from the denominator).
    assert summary.hallucination_rate == pytest.approx(0.5)
    # validation_coverage = (VALID+INVALID)/mapped_count = 2/4
    assert summary.validation_coverage == pytest.approx(0.5)
    assert summary.unresolved_validation_count == 2
    assert summary.unresolved_validation_rate == pytest.approx(0.5)


def test_hallucination_rate_none_when_no_resolved_codes() -> None:
    summary = summarize_hallucination(mapped_count=2, validation_statuses=[UNRESOLVED, UNRESOLVED])
    assert summary.hallucination_rate is None
    assert summary.validation_coverage == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 40. validation cache reused -- repeated/shared codes hit the validator once
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_cache_reused_across_rows() -> None:
    validator = _FakeValidator({"HP:0002110": True})
    cache: ValidationCache = {}
    for _ in range(3):
        status, _ = validate_one(status="mapped", mapped_code="HP:0002110", validator=validator, cache=cache)
        assert status == VALID
    assert validator.calls == ["HP:0002110"]  # only the first call actually hit the validator


def test_validation_cache_normalizes_case() -> None:
    validator = _FakeValidator({"HP:0002110": True})
    cache: ValidationCache = {}
    validate_one(status="mapped", mapped_code="hp:0002110", validator=validator, cache=cache)
    status, _ = validate_one(status="mapped", mapped_code="HP:0002110", validator=validator, cache=cache)
    assert status == VALID
    assert len(validator.calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 41. unmapped/error rows are never validated
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_rows_not_validated() -> None:
    validator = _FakeValidator({})
    cache: ValidationCache = {}
    status, source = validate_one(status="unmapped", mapped_code=None, validator=validator, cache=cache)
    assert status == NOT_APPLICABLE
    assert source is None
    assert validator.calls == []


def test_error_rows_not_validated() -> None:
    validator = _FakeValidator({})
    cache: ValidationCache = {}
    status, _ = validate_one(status="error", mapped_code=None, validator=validator, cache=cache)
    assert status == NOT_APPLICABLE
    assert validator.calls == []


def test_unmapped_sentinel_code_never_validated() -> None:
    validator = _FakeValidator({})
    cache: ValidationCache = {}
    status, _ = validate_one(status="mapped", mapped_code="UNKNOWN:UNMAPPED", validator=validator, cache=cache)
    assert status == NOT_APPLICABLE
    assert validator.calls == []
