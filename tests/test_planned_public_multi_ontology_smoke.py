from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_smoke_module() -> Any:
    live_dir = Path(__file__).parent / "live"
    sys.path.insert(0, str(live_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "planned_public_multi_ontology_smoke",
            live_dir / "planned_public_multi_ontology_smoke.py",
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(live_dir))


smoke = _load_smoke_module()


def _result(
    ontology: str,
    *,
    alternatives: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        target_code=f"{ontology}:123" if ontology != "UNKNOWN" else "UNKNOWN:UNMAPPED",
        ontology=ontology,
        confidence=0.7,
        alternatives=alternatives or [],
    )


def _alt(ontology: str) -> SimpleNamespace:
    return SimpleNamespace(
        code=f"{ontology}:456",
        term=f"{ontology} alternative",
        ontology=ontology,
        confidence=0.5,
        explanation="Alternative explanation.",
    )


def test_parse_allowed_ontologies_defaults() -> None:
    assert smoke.parse_allowed_ontologies(None) == ["LOINC", "HPO", "MONDO"]


def test_parse_allowed_ontologies_trims_uppercases_and_dedupes() -> None:
    assert smoke.parse_allowed_ontologies(" loinc, HPO, hpO,,MONDO ") == [
        "LOINC",
        "HPO",
        "MONDO",
    ]


def test_resolve_allowed_ontologies_unrestricted_mode() -> None:
    assert (
        smoke.resolve_allowed_ontologies(
            no_filter=True,
            raw_allowed="LOINC,HPO,MONDO",
        )
        is None
    )
    assert smoke.hard_filter_active(None) is False


def test_validate_scope_accepts_allowed_primary_unknown_and_alternatives() -> None:
    validation = smoke.validate_scope(
        result=_result("UNKNOWN", alternatives=[_alt("HPO"), _alt("MONDO")]),
        allowed_target_ontologies=["LOINC", "HPO", "MONDO"],
        pipeline_metadata={
            "retrieval_trace": {
                "route_calls": [
                    {"candidate_ontologies": ["LOINC", "HPO", "MONDO"]},
                ],
            },
        },
    )

    assert validation["primary_valid"] is True
    assert validation["alternatives_valid"] is True
    assert validation["searched_ontologies_valid"] is True
    assert validation["violations"] == []


def test_validate_scope_flags_unselected_primary() -> None:
    validation = smoke.validate_scope(
        result=_result("NCIT"),
        allowed_target_ontologies=["LOINC", "HPO", "MONDO"],
        pipeline_metadata={},
    )

    assert validation["primary_valid"] is False
    assert validation["violations"]


def test_validate_scope_flags_unselected_alternative() -> None:
    validation = smoke.validate_scope(
        result=_result("LOINC", alternatives=[_alt("NCIT")]),
        allowed_target_ontologies=["LOINC", "HPO", "MONDO"],
        pipeline_metadata={},
    )

    assert validation["alternatives_valid"] is False
    assert any("alternative ontology" in item for item in validation["violations"])


def test_validate_scope_flags_unselected_searched_ontology() -> None:
    validation = smoke.validate_scope(
        result=_result("LOINC"),
        allowed_target_ontologies=["LOINC", "HPO", "MONDO"],
        pipeline_metadata={
            "retrieval_trace": {
                "route_calls": [
                    {"candidate_ontologies": ["LOINC", "NCIT"]},
                ],
            },
        },
    )

    assert validation["searched_ontologies_valid"] is False
    assert any("searched ontologies" in item for item in validation["violations"])


def test_validate_scope_unrestricted_skips_membership_assertions() -> None:
    validation = smoke.validate_scope(
        result=_result("NCIT", alternatives=[_alt("CHEBI")]),
        allowed_target_ontologies=None,
        pipeline_metadata={
            "retrieval_trace": {
                "route_calls": [
                    {"candidate_ontologies": ["NCIT", "CHEBI"]},
                ],
            },
        },
    )

    assert validation["hard_filter_active"] is False
    assert validation["primary_valid"] is True
    assert validation["alternatives_valid"] is True
    assert validation["searched_ontologies_valid"] is True
    assert validation["violations"] == []
