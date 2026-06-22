from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.search_tools import SearchTools


def _loinc_response(status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "results": [
            {
                "LOINC_NUM": "8480-6",
                "LONG_COMMON_NAME": "Systolic blood pressure",
                "COMPONENT": "Systolic blood pressure",
            },
        ],
    }
    return response


@pytest.mark.unit
def test_search_loinc_passes_basic_auth() -> None:
    tools = SearchTools(
        loinc_username="service-user",
        loinc_password="service-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        results = tools.search_loinc("systolic blood pressure")

    auth = mock_get.call_args.kwargs["auth"]
    assert auth.username == "service-user"
    assert auth.password == "service-password"
    assert mock_get.call_args.args[0] == (
        "https://loinc.regenstrief.org/searchapi/loincs"
    )
    assert mock_get.call_args.kwargs["params"] == {
        "query": "systolic blood pressure",
        "rows": 10,
        "offset": 0,
    }
    assert results[0]["code"] == "LOINC:8480-6"
    assert results[0]["ontology"] == "LOINC"
    assert results[0]["term"]


@pytest.mark.unit
def test_search_loinc_uses_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOINC_USERNAME", "environment-user")
    monkeypatch.setenv("LOINC_PASSWORD", "environment-password")
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        tools.search_loinc("systolic blood pressure")

    auth = mock_get.call_args.kwargs["auth"]
    assert auth.username == "environment-user"
    assert auth.password == "environment-password"


@pytest.mark.unit
def test_search_loinc_constructor_credentials_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOINC_USERNAME", "environment-user")
    monkeypatch.setenv("LOINC_PASSWORD", "environment-password")
    tools = SearchTools(
        loinc_username="constructor-user",
        loinc_password="constructor-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        tools.search_loinc("systolic blood pressure")

    auth = mock_get.call_args.kwargs["auth"]
    assert auth.username == "constructor-user"
    assert auth.password == "constructor-password"


@pytest.mark.unit
def test_search_loinc_without_credentials_does_not_call_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in (
        "LOINC_USERNAME",
        "LOINC_PASSWORD",
        "LOINC_FHIR_USER",
        "LOINC_FHIR_PASS",
    ):
        monkeypatch.delenv(name, raising=False)
    tools = SearchTools(request_delay=0)

    with patch("requests.get") as mock_get, caplog.at_level(logging.WARNING):
        results = tools.search_loinc("systolic blood pressure")

    assert results == []
    mock_get.assert_not_called()
    assert (
        "LOINC credentials are required for live LOINC API search. "
        "Set LOINC_USERNAME and LOINC_PASSWORD."
    ) in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_search_loinc_http_error_returns_empty_without_logging_credentials(
    status_code: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    username = "secret-service-user"
    password = "secret-service-password"
    tools = SearchTools(
        loinc_username=username,
        loinc_password=password,
        request_delay=0,
    )

    with patch(
        "requests.get",
        return_value=_loinc_response(status_code),
    ), caplog.at_level(logging.ERROR):
        results = tools.search_loinc("systolic blood pressure")

    assert results == []
    assert f"HTTP {status_code}" in caplog.text
    assert username not in caplog.text
    assert password not in caplog.text


@pytest.mark.unit
def test_search_loinc_parser_accepts_field_name_variants() -> None:
    response = _loinc_response()
    response.json.return_value = {
        "response": {
            "docs": [
                {
                    "loincNum": "8480-6",
                    "displayName": "Systolic blood pressure",
                    "definitionDescription": "Arterial systolic pressure",
                },
            ],
        },
    }
    tools = SearchTools(
        loinc_username="service-user",
        loinc_password="service-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=response):
        results = tools.search_loinc("systolic blood pressure", top_k=5)

    assert results == [
        {
            "code": "LOINC:8480-6",
            "term": "Systolic blood pressure",
            "ontology": "LOINC",
            "score": 1.0,
            "definition": "Arterial systolic pressure",
            "source": "LOINC-Search-API",
        },
    ]


# ── _normalize_code tests ─────────────────────────────────────────────────────

@pytest.mark.unit
def test_normalize_code_snomed_obo_id_strips_alias_prefix() -> None:
    """OLS4 obo_id "SNOMED:768500006" must normalize to "SNOMEDCT:768500006", not
    "SNOMEDCT:SNOMED:768500006"."""
    tools = SearchTools(request_delay=0)
    assert tools._normalize_code("SNOMED:768500006", "SNOMEDCT") == "SNOMEDCT:768500006"
    assert tools._normalize_code("SNOMED:109081006", "SNOMEDCT") == "SNOMEDCT:109081006"


@pytest.mark.unit
def test_normalize_code_snomed_idempotent() -> None:
    """Applying _normalize_code twice must equal applying it once for SNOMED inputs."""
    tools = SearchTools(request_delay=0)
    once = tools._normalize_code("SNOMED:768500006", "SNOMEDCT")
    twice = tools._normalize_code(once, "SNOMEDCT")
    assert once == twice == "SNOMEDCT:768500006"

    # Already-correct canonical form is also idempotent.
    canonical = tools._normalize_code("SNOMEDCT:768500006", "SNOMEDCT")
    assert canonical == tools._normalize_code(canonical, "SNOMEDCT") == "SNOMEDCT:768500006"


@pytest.mark.unit
def test_normalize_code_bare_snomed_id() -> None:
    """A bare numeric SNOMED id (no namespace) normalizes to "SNOMEDCT:<id>"."""
    tools = SearchTools(request_delay=0)
    assert tools._normalize_code("768500006", "SNOMEDCT") == "SNOMEDCT:768500006"


@pytest.mark.unit
def test_normalize_code_clean_codes_unchanged() -> None:
    """Already-correct codes for other ontologies must pass through unaltered
    (verifies the strip does not corrupt non-SNOMED codes)."""
    tools = SearchTools(request_delay=0)
    cases = [
        ("CHEBI:6801", "CHEBI", "CHEBI:6801"),
        ("RXNORM:6809", "RXNORM", "RXNORM:6809"),
        ("LOINC:8480-6", "LOINC", "LOINC:8480-6"),
        ("HP:0001234", "HPO", "HP:0001234"),
        ("MONDO:0007037", "MONDO", "MONDO:0007037"),
        ("ICD10:A00", "ICD10", "ICD10:A00"),
        ("ICD10:A00", "ICD10CM", "ICD10:A00"),  # ICD10CM alias → canonical ICD10
    ]
    for raw, ontology, expected in cases:
        result = tools._normalize_code(raw, ontology)
        assert result == expected, (
            f"_normalize_code({raw!r}, {ontology!r}) = {result!r}, expected {expected!r}"
        )


@pytest.mark.unit
def test_normalize_code_idempotent_all_prefix_map_ontologies() -> None:
    """For every ontology in the prefix map, normalizing twice equals normalizing once."""
    tools = SearchTools(request_delay=0)
    cases = [
        ("HP:0001234", "HP"),
        ("HP:0001234", "HPO"),
        ("MONDO:0007037", "MONDO"),
        ("NCIT:C12345", "NCIT"),
        ("LOINC:8480-6", "LOINC"),
        ("ICD10:A00", "ICD10"),
        ("ICD10CM:A00", "ICD10CM"),
        ("SNOMEDCT:768500006", "SNOMEDCT"),
        ("SNOMED:768500006", "SNOMED"),
        ("RXNORM:6809", "RXNORM"),
        ("RXCUI:6809", "RXCUI"),
        ("UO:0000001", "UO"),
    ]
    for raw, ontology in cases:
        once = tools._normalize_code(raw, ontology)
        twice = tools._normalize_code(once, ontology)
        assert once == twice, (
            f"_normalize_code not idempotent for ({raw!r}, {ontology!r}): "
            f"once={once!r}, twice={twice!r}"
        )


# ── search_rxnorm tests ───────────────────────────────────────────────────────

def _rxnorm_response(candidates: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "approximateGroup": {
            "inputTerm": None,
            "candidate": candidates,
        }
    }
    return response


@pytest.mark.unit
def test_search_rxnorm_deduplicates_atoms() -> None:
    """Two atoms for the same rxcui — one named (RXNORM source), one not (GS source).
    Must emit exactly one candidate carrying the RXNORM atom's name."""
    atoms = [
        {"rxcui": "6809", "rxaui": "10328664", "rank": "1", "source": "GS"},
        {"rxcui": "6809", "rxaui": "12251601", "rank": "1", "name": "metformin", "source": "RXNORM"},
    ]
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_rxnorm_response(atoms)):
        results = tools.search_rxnorm("metformin", top_k=6)

    assert len(results) == 1
    assert results[0]["code"] == "RXNORM:6809"
    assert results[0]["term"] == "metformin"
    assert results[0]["source"] == "RxNav"


@pytest.mark.unit
def test_search_rxnorm_skips_all_unnamed_rxcui(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An rxcui whose every atom lacks 'name' must be excluded; a debug log is emitted."""
    atoms = [
        {"rxcui": "9999", "rxaui": "AAA", "rank": "1", "source": "GS"},
        {"rxcui": "9999", "rxaui": "BBB", "rank": "1", "source": "NDDF"},
    ]
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_rxnorm_response(atoms)), \
            caplog.at_level(logging.DEBUG):
        results = tools.search_rxnorm("unknown", top_k=6)

    assert results == []
    assert "9999" in caplog.text


@pytest.mark.unit
def test_search_rxnorm_multiple_concepts() -> None:
    """Three distinct rxcuis → three distinct candidates, no duplicate codes,
    ordered by score descending (best rank first)."""
    atoms = [
        {"rxcui": "6809", "rxaui": "A1", "rank": "1", "name": "metformin", "source": "RXNORM"},
        {"rxcui": "1161611", "rxaui": "B1", "rank": "2", "name": "metformin Pill", "source": "RXNORM"},
        {"rxcui": "583194", "rxaui": "C1", "rank": "3", "name": "metformin Oral Tablet", "source": "RXNORM"},
    ]
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_rxnorm_response(atoms)):
        results = tools.search_rxnorm("metformin", top_k=6)

    codes = [r["code"] for r in results]
    assert len(codes) == 3
    assert len(set(codes)) == 3, "duplicate codes present"
    assert codes == sorted(codes, key=lambda c: -next(r["score"] for r in results if r["code"] == c))
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]


@pytest.mark.unit
def test_search_rxnorm_caps_concepts_not_atoms() -> None:
    """top_k=2 with 3 distinct concepts spread across 5 atoms.

    The concept whose atoms sit past index top_k in the raw list must NOT be
    dropped — top_k must cap concepts after dedup, not atoms before dedup.
    """
    atoms = [
        # Concept A — 3 atoms (indices 0-2)
        {"rxcui": "AAA", "rxaui": "a1", "rank": "1", "source": "GS"},
        {"rxcui": "AAA", "rxaui": "a2", "rank": "1", "name": "Concept A", "source": "RXNORM"},
        {"rxcui": "AAA", "rxaui": "a3", "rank": "1", "source": "NDDF"},
        # Concept B — 1 atom (index 3)
        {"rxcui": "BBB", "rxaui": "b1", "rank": "2", "name": "Concept B", "source": "RXNORM"},
        # Concept C — 1 atom (index 4, past top_k=2 if sliced early)
        {"rxcui": "CCC", "rxaui": "c1", "rank": "3", "name": "Concept C", "source": "RXNORM"},
    ]
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_rxnorm_response(atoms)):
        results = tools.search_rxnorm("test", top_k=2)

    codes = {r["code"] for r in results}
    assert len(results) == 2
    # The two highest-scoring concepts (lowest rank) must be returned
    assert "RXNORM:AAA" in codes
    assert "RXNORM:BBB" in codes
    # Concept C (rank 3) is cut by the top_k cap on concepts, not by pre-slicing atoms
    assert "RXNORM:CCC" not in codes
    # Confirm none of the returned candidates have a blank term
    for r in results:
        assert r["term"], f"blank term in result {r}"
