from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

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


def _ols_response(docs: list[dict]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": {"docs": docs}}
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
    assert mock_get.call_args.args[0] == ("https://loinc.regenstrief.org/searchapi/loincs")
    assert mock_get.call_args.kwargs["params"] == {
        "query": "systolic blood pressure =status:ACTIVE",
        "rows": "10",
        "offset": "0",
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

    with (
        patch(
            "requests.get",
            return_value=_loinc_response(status_code),
        ),
        caplog.at_level(logging.ERROR),
    ):
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


# ── LOINC =status:ACTIVE eligibility filter ───────────────────────────────────


@pytest.mark.unit
def test_search_loinc_appends_active_status_filter() -> None:
    """Every public LOINC search must be restricted to ACTIVE concepts, and the
    original semantic query text must remain intact ahead of the filter."""
    tools = SearchTools(
        loinc_username="service-user",
        loinc_password="service-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        tools.search_loinc("glucose")

    assert mock_get.call_args.kwargs["params"]["query"] == "glucose =status:ACTIVE"


@pytest.mark.unit
def test_search_loinc_active_status_filter_added_exactly_once() -> None:
    """A query that already carries a `=status:` filter (e.g. a retried or
    pre-filtered query) must not have the filter appended a second time."""
    tools = SearchTools(
        loinc_username="service-user",
        loinc_password="service-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        tools.search_loinc("glucose =status:ACTIVE")

    assert mock_get.call_args.kwargs["params"]["query"] == "glucose =status:ACTIVE"
    assert mock_get.call_args.kwargs["params"]["query"].count("=status:") == 1


@pytest.mark.unit
def test_search_loinc_active_status_filter_applied_to_every_planned_query() -> None:
    """A query planner may issue several LOINC searches (variants/retries) for a
    single mapping request; each independent call must receive the filter."""
    tools = SearchTools(
        loinc_username="service-user",
        loinc_password="service-password",
        request_delay=0,
    )

    with patch("requests.get", return_value=_loinc_response()) as mock_get:
        tools.search_loinc("glucose")
        tools.search_loinc("blood glucose level")
        tools.search_loinc("fasting glucose")

    queries = [call.kwargs["params"]["query"] for call in mock_get.call_args_list]
    assert queries == [
        "glucose =status:ACTIVE",
        "blood glucose level =status:ACTIVE",
        "fasting glucose =status:ACTIVE",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_query", "expected"),
    [
        # 1. no existing status filter
        ("glucose", "glucose =status:ACTIVE"),
        ("glucose  ", "glucose =status:ACTIVE"),
        ("", "=status:ACTIVE"),
        # 2. ACTIVE already present
        ("glucose =status:ACTIVE", "glucose =status:ACTIVE"),
        # 3. differently-cased ACTIVE
        ("glucose =STATUS:active", "glucose =status:ACTIVE"),
        ("glucose =Status:Active", "glucose =status:ACTIVE"),
        # 4-6. a non-ACTIVE filter must be replaced, not preserved/stacked
        ("glucose =status:DEPRECATED", "glucose =status:ACTIVE"),
        ("glucose =status:DISCOURAGED", "glucose =status:ACTIVE"),
        ("glucose =status:TRIAL", "glucose =status:ACTIVE"),
        ("glucose =status:deprecated", "glucose =status:ACTIVE"),
    ],
)
def test_with_active_status_filter_enforces_active_only(
    raw_query: str, expected: str
) -> None:
    result = SearchTools._with_active_status_filter(raw_query)
    assert result == expected
    # 7. exactly one status filter remains in every case
    assert result.count("=status:") == 1


@pytest.mark.unit
def test_search_ols_query_unaffected_by_loinc_status_filter() -> None:
    """The =status:ACTIVE eligibility filter is LOINC-only; OLS-backed ontology
    searches (HPO, MONDO, NCIT, EFO, SNOMED, ...) must be untouched."""
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_ols_response([])) as mock_get:
        tools.search_ols("glucose", "EFO")

    assert mock_get.call_args.kwargs["params"]["q"] == "glucose"
    assert "status" not in mock_get.call_args.kwargs["params"]["q"]


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
def test_normalize_code_keeps_authoritative_hp_namespace_for_mondo_request() -> None:
    tools = SearchTools(request_delay=0)

    result = tools._normalize_code("HP:0002099", "MONDO")

    assert result == "HP:0002099"
    assert result != "MONDO:HP:0002099"


@pytest.mark.unit
def test_search_ols_classifies_hp_result_from_mondo_scoped_request_as_hpo() -> None:
    tools = SearchTools(request_delay=0)
    doc = {
        "obo_id": "HP:0002099",
        "iri": "http://purl.obolibrary.org/obo/HP_0002099",
        "label": "Asthma",
        "description": ["A respiratory phenotype."],
    }

    with patch("requests.get", return_value=_ols_response([doc])):
        results = tools.search_ols("asthma", ontology="MONDO", top_k=1)

    assert results[0]["code"] == "HP:0002099"
    assert results[0]["ontology"] == "HPO"
    assert results[0]["code"] != "MONDO:HP:0002099"


@pytest.mark.unit
def test_search_ols_scopes_efo_request_to_efo_ontology_id() -> None:
    tools = SearchTools(request_delay=0)
    doc = {
        "obo_id": "EFO:0000408",
        "iri": "http://www.ebi.ac.uk/efo/EFO_0000408",
        "label": "disease",
        "description": ["A disease experimental factor."],
    }

    with patch("requests.get", return_value=_ols_response([doc])) as mock_get:
        results = tools.search_ols("disease", ontology="EFO", top_k=3)

    assert mock_get.call_args.kwargs["params"]["ontology"] == "efo"
    assert mock_get.call_args.kwargs["params"]["q"] == "disease"
    assert mock_get.call_args.kwargs["params"]["rows"] == "3"
    assert results[0]["code"] == "EFO:0000408"
    assert results[0]["ontology"] == "EFO"


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
        ("EFO:0000408", "EFO"),
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
        {
            "rxcui": "6809",
            "rxaui": "12251601",
            "rank": "1",
            "name": "metformin",
            "source": "RXNORM",
        },
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

    with (
        patch("requests.get", return_value=_rxnorm_response(atoms)),
        caplog.at_level(logging.DEBUG),
    ):
        results = tools.search_rxnorm("unknown", top_k=6)

    assert results == []
    assert "9999" in caplog.text


@pytest.mark.unit
def test_search_rxnorm_multiple_concepts() -> None:
    """Three distinct rxcuis → three distinct candidates, no duplicate codes,
    ordered by score descending (best rank first)."""
    atoms = [
        {"rxcui": "6809", "rxaui": "A1", "rank": "1", "name": "metformin", "source": "RXNORM"},
        {
            "rxcui": "1161611",
            "rxaui": "B1",
            "rank": "2",
            "name": "metformin Pill",
            "source": "RXNORM",
        },
        {
            "rxcui": "583194",
            "rxaui": "C1",
            "rank": "3",
            "name": "metformin Oral Tablet",
            "source": "RXNORM",
        },
    ]
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_rxnorm_response(atoms)):
        results = tools.search_rxnorm("metformin", top_k=6)

    codes = [r["code"] for r in results]
    assert len(codes) == 3
    assert len(set(codes)) == 3, "duplicate codes present"
    assert codes == sorted(
        codes, key=lambda c: -next(r["score"] for r in results if r["code"] == c)
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# Bounded retry for transient public-API failures (search_tools._get_with_retry)
# ─────────────────────────────────────────────────────────────────────────────


def _cough_doc() -> dict:
    return {
        "obo_id": "HP:0012735",
        "iri": "http://purl.obolibrary.org/obo/HP_0012735",
        "label": "Cough",
    }


def _status_response(status_code: int, body: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error", response=response
        )
    else:
        response.raise_for_status.return_value = None
    response.json.return_value = {"response": {"docs": []}}
    return response


@pytest.mark.unit
def test_ols_successful_request_makes_one_attempt() -> None:
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_ols_response([_cough_doc()])) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 1
    assert len(results) == 1
    assert diagnostics == {"attempts": 1}


@pytest.mark.unit
def test_ols_timeout_then_success_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    tools = SearchTools(request_delay=0)

    with patch(
        "requests.get",
        side_effect=[
            requests.exceptions.ReadTimeout("read timeout"),
            _ols_response([_cough_doc()]),
        ],
    ) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 2
    assert len(results) == 1
    assert diagnostics["attempts"] == 2
    assert "final_error_type" not in diagnostics


@pytest.mark.unit
def test_ols_connection_reset_then_success_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    tools = SearchTools(request_delay=0)
    reset_exc = requests.exceptions.ConnectionError("Connection reset by peer")

    with patch("requests.get", side_effect=[reset_exc, _ols_response([_cough_doc()])]) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 2
    assert len(results) == 1
    assert diagnostics["attempts"] == 2


@pytest.mark.unit
def test_ols_http_500_then_success_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    tools = SearchTools(request_delay=0)

    with patch(
        "requests.get", side_effect=[_status_response(500), _ols_response([_cough_doc()])]
    ) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 2
    assert len(results) == 1
    assert diagnostics["attempts"] == 2
    assert "final_error_type" not in diagnostics


@pytest.mark.unit
def test_ols_http_429_then_success_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    tools = SearchTools(request_delay=0)

    with patch(
        "requests.get", side_effect=[_status_response(429), _ols_response([_cough_doc()])]
    ) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 2
    assert len(results) == 1


@pytest.mark.unit
def test_ols_timeout_on_all_attempts_gives_up_after_exactly_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda s: sleeps.append(s))
    tools = SearchTools(request_delay=0)

    with patch(
        "requests.get",
        side_effect=[
            requests.exceptions.ReadTimeout("timeout 1"),
            requests.exceptions.ReadTimeout("timeout 2"),
            requests.exceptions.ReadTimeout("timeout 3"),
        ],
    ) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 3
    assert results == []  # graceful final retrieval failure, no exception escapes
    assert diagnostics["attempts"] == 3
    assert diagnostics["final_error_type"] == "timeout"
    assert len(sleeps) == 2  # exactly two waits between three attempts


@pytest.mark.unit
def test_ols_http_400_does_not_retry() -> None:
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_status_response(400)) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 1
    assert results == []
    assert diagnostics["attempts"] == 1
    assert diagnostics["final_error_type"] == "http_400"


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403])
def test_ols_http_401_403_does_not_retry(status_code: int) -> None:
    tools = SearchTools(request_delay=0)

    with patch("requests.get", return_value=_status_response(status_code)) as mock_get:
        diagnostics: dict = {}
        results = tools.search_ols("cough", ontology="HPO", route_diagnostics=diagnostics)

    assert mock_get.call_count == 1
    assert results == []
    assert diagnostics["final_error_type"] == f"http_{status_code}"


@pytest.mark.unit
def test_retry_backoff_does_not_duplicate_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    tools = SearchTools(request_delay=0)

    with patch(
        "requests.get",
        side_effect=[requests.exceptions.ReadTimeout("timeout"), _ols_response([_cough_doc()])],
    ):
        results = tools.search_ols("cough", ontology="HPO")

    assert len(results) == 1  # the failed first attempt contributed nothing


@pytest.mark.unit
def test_loinc_credentials_never_appear_in_logs_on_retry_or_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("llm_ontology_mapper.search_tools.time.sleep", lambda *_: None)
    username, password = "secret-user", "secret-pass"
    tools = SearchTools(loinc_username=username, loinc_password=password, request_delay=0)

    with (
        patch(
            "requests.get",
            side_effect=[
                requests.exceptions.ReadTimeout("timeout"),
                _status_response(400, body="Bad request: malformed query"),
            ],
        ),
        caplog.at_level(logging.WARNING),
    ):
        results = tools.search_loinc("systolic blood pressure")

    assert results == []
    assert username not in caplog.text
    assert password not in caplog.text
