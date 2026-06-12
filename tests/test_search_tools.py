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
