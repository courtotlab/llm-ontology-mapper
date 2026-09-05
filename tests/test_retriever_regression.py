"""
Characterization / regression tests for OntologyRetriever.retrieve().

These tests pin the CURRENT observable behaviour of retrieve() for both
(a) the lexical adapter path (OLS HTTP) and
(b) the SapBERT path (mocked endpoint).

They must stay green through ALL of Stage 1 (A1→A4) — green here means
"no silent behaviour change in retrieve()".  All HTTP is mocked; no live
services required.

Run with:  pytest tests/test_retriever_regression.py -v -m unit
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.retriever import OntologyRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ols_response(docs: list[dict[str, Any]]) -> MagicMock:
    """Fake requests.Response for an OLS search hit."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": {"docs": docs}}
    return mock_resp


def _make_sapbert_response(results: list[dict[str, Any]]) -> MagicMock:
    """Fake requests.Response for a SapBERT /search hit."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"results": results}
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# (a) Lexical adapter path — OLS HTTP
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_retrieve_lexical_ols_returns_candidates() -> None:
    """
    retrieve() via OLS must return a non-empty candidate list with the
    expected schema: each item has code, term, score; top_score matches
    the highest score in the list.
    """
    ols_doc = {
        "obo_id": "HP:0002110",
        "iri": "http://purl.obolibrary.org/obo/HP_0002110",
        "label": "Bronchiectasis",
        "description": ["Widening of the bronchi"],
    }
    mock_get_resp = _make_ols_response([ols_doc])

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_get_resp) as mock_get:
        candidates, top_score = retriever.retrieve(
            query="bronchiectasis",
            ontologies=["HPO"],
        )

    # Shape checks
    assert isinstance(candidates, list)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["code"] == "HP:0002110"
    assert c["term"] == "Bronchiectasis"
    assert isinstance(c["score"], float)
    assert top_score == c["score"]


@pytest.mark.unit
def test_retrieve_lexical_ols_empty_docs_returns_empty() -> None:
    """OLS returning zero docs → retrieve() returns ([], 0.0)."""
    mock_get_resp = _make_ols_response([])

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_get_resp):
        candidates, top_score = retriever.retrieve(
            query="zzzyyyxxx_no_match",
            ontologies=["HPO"],
        )

    assert candidates == []
    assert top_score == 0.0


@pytest.mark.unit
def test_retrieve_lexical_scores_descending() -> None:
    """retrieve() must return candidates sorted by descending score."""
    docs = [
        {"obo_id": "HP:0002110", "iri": "", "label": "Bronchiectasis", "description": ""},
        {"obo_id": "HP:0001234", "iri": "", "label": "Other", "description": ""},
        {"obo_id": "HP:0005678", "iri": "", "label": "Another", "description": ""},
    ]
    mock_get_resp = _make_ols_response(docs)

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_get_resp):
        candidates, top_score = retriever.retrieve(
            query="bronchiectasis",
            ontologies=["HPO"],
        )

    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert top_score == scores[0]


@pytest.mark.unit
def test_retrieve_lexical_http_error_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """An HTTP error on OLS must not raise — retrieve() returns ([], 0.0)."""
    import requests as _requests

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", side_effect=_requests.exceptions.RequestException("timeout")):
        candidates, top_score = retriever.retrieve(
            query="bronchiectasis",
            ontologies=["HPO"],
        )

    assert candidates == []
    assert top_score == 0.0


@pytest.mark.unit
def test_retrieve_loinc_path() -> None:
    """retrieve() with LOINC ontology calls the shared LOINC Search API."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "results": [
            {"loincNum": "8480-6", "longCommonName": "Systolic blood pressure"},
            {"loincNum": "8462-4", "longCommonName": "Diastolic blood pressure"},
        ],
    }

    retriever = OntologyRetriever(
        ontologies=["LOINC"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
        loinc_username="service-user",
        loinc_password="service-password",
    )

    with patch("requests.get", return_value=mock_resp) as mock_get:
        candidates, top_score = retriever.retrieve(
            query="blood pressure",
            ontologies=["LOINC"],
        )

    assert len(candidates) == 2
    codes = {c["code"] for c in candidates}
    assert "LOINC:8480-6" in codes
    assert top_score > 0.0
    assert mock_get.call_args.args[0] == (
        "https://loinc.regenstrief.org/searchapi/loincs"
    )
    assert mock_get.call_args.kwargs["params"] == {
        "query": "blood pressure =status:ACTIVE",
        "rows": "5",
        "offset": "0",
    }
    assert mock_get.call_args.kwargs["auth"].username == "service-user"


@pytest.mark.unit
def test_retrieve_loinc_uses_service_credentials_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOINC_USERNAME", "environment-user")
    monkeypatch.setenv("LOINC_PASSWORD", "environment-password")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "results": [
            {"loincNum": "8480-6", "longCommonName": "Systolic blood pressure"},
        ],
    }
    retriever = OntologyRetriever(
        ontologies=["LOINC"],
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_resp) as mock_get:
        candidates, _ = retriever.retrieve(
            query="systolic blood pressure",
            ontologies=["LOINC"],
        )

    auth = mock_get.call_args.kwargs["auth"]
    assert auth.username == "environment-user"
    assert auth.password == "environment-password"
    assert candidates[0]["code"] == "LOINC:8480-6"


@pytest.mark.unit
def test_retrieve_rxnorm_path() -> None:
    """retrieve() with RXNORM ontology calls RxNav."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "approximateGroup": {
            "candidate": [
                {"rxcui": "1049502", "name": "Ibuprofen 200 MG", "rank": 1},
            ]
        }
    }

    retriever = OntologyRetriever(
        ontologies=["RXNORM"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_resp):
        candidates, top_score = retriever.retrieve(
            query="ibuprofen",
            ontologies=["RXNORM"],
        )

    assert len(candidates) == 1
    assert candidates[0]["code"] == "RXNORM:1049502"
    assert top_score > 0.0


@pytest.mark.unit
def test_retrieve_icd10_path() -> None:
    """retrieve() with ICD10 ontology calls NIH Clinical Tables."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        1, ["J45.909"], None,
        [["J45.909", "Unspecified asthma, uncomplicated"]]
    ]

    retriever = OntologyRetriever(
        ontologies=["ICD10"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_resp):
        candidates, top_score = retriever.retrieve(
            query="asthma",
            ontologies=["ICD10"],
        )

    assert len(candidates) == 1
    assert candidates[0]["code"] == "ICD10:J45.909"
    assert top_score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# (b) SapBERT path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_retrieve_sapbert_path_returns_results() -> None:
    """
    When sapbert_url is configured and the server responds, retrieve()
    must use the SapBERT results and skip OLS.
    """
    sapbert_results = [
        {"code": "HP:0002110", "term": "Bronchiectasis", "score": 0.95},
        {"code": "HP:0005381", "term": "Chronic bronchitis", "score": 0.82},
    ]
    mock_post_resp = _make_sapbert_response(sapbert_results)

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
        sapbert_url="http://sapbert.local",
    )

    with patch("requests.post", return_value=mock_post_resp) as mock_post, \
         patch("requests.get") as mock_get:
        candidates, top_score = retriever.retrieve(
            query="bronchiectasis",
            ontologies=["HPO"],
        )

    # SapBERT was called; OLS was NOT called
    assert mock_post.called
    assert not mock_get.called

    assert len(candidates) == 2
    assert candidates[0]["code"] == "HP:0002110"
    assert candidates[0]["score"] == pytest.approx(0.95)
    assert top_score == pytest.approx(0.95)


@pytest.mark.unit
def test_retrieve_sapbert_fallback_to_ols_on_connection_error() -> None:
    """
    If the SapBERT server is unreachable, retrieve() must fall back to
    OLS transparently (no exception raised; results from OLS are returned).
    """
    import requests as _requests

    ols_doc = {
        "obo_id": "HP:0002110",
        "iri": "http://purl.obolibrary.org/obo/HP_0002110",
        "label": "Bronchiectasis",
        "description": [],
    }
    mock_ols_resp = _make_ols_response([ols_doc])

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
        sapbert_url="http://sapbert.local",
    )

    with patch(
        "requests.post",
        side_effect=_requests.exceptions.ConnectionError("refused"),
    ), patch("requests.get", return_value=mock_ols_resp):
        candidates, top_score = retriever.retrieve(
            query="bronchiectasis",
            ontologies=["HPO"],
        )

    # Should fall back to OLS and still get a result
    assert len(candidates) == 1
    assert candidates[0]["code"] == "HP:0002110"


@pytest.mark.unit
def test_retrieve_sapbert_source_field() -> None:
    """Candidates from SapBERT carry source='SapBERT'."""
    mock_post_resp = _make_sapbert_response([
        {"code": "HP:0002110", "term": "Bronchiectasis", "score": 0.95,
         "definition": "An example"},
    ])

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
        sapbert_url="http://sapbert.local",
    )

    with patch("requests.post", return_value=mock_post_resp):
        candidates, _ = retriever.retrieve("bronchiectasis", ontologies=["HPO"])

    assert candidates[0]["source"] == "SapBERT"


@pytest.mark.unit
def test_retrieve_top_k_limits_results() -> None:
    """retrieve() must return at most top_k results."""
    docs = [
        {"obo_id": f"HP:{i:07d}", "iri": "", "label": f"Term{i}", "description": ""}
        for i in range(10)
    ]
    mock_resp = _make_ols_response(docs)

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=3,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_resp):
        candidates, _ = retriever.retrieve("something", ontologies=["HPO"])

    assert len(candidates) <= 3


@pytest.mark.unit
def test_retrieve_return_types() -> None:
    """retrieve() always returns (list, float) — even on empty results."""
    mock_resp = _make_ols_response([])

    retriever = OntologyRetriever(
        ontologies=["HPO"],
        top_k=5,
        api_timeout=5,
        request_delay=0,
    )

    with patch("requests.get", return_value=mock_resp):
        result = retriever.retrieve("anything", ontologies=["HPO"])

    assert isinstance(result, tuple) and len(result) == 2
    candidates, top_score = result
    assert isinstance(candidates, list)
    assert isinstance(top_score, float)
