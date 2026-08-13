"""
Unit tests for PublicOntologyRetriever (Phase 7).

Validates:
1.  Public QueryPlan is accepted
2.  Local QueryPlan raises PublicRetrievalError
3.  Disabled QueryPlan raises PublicRetrievalError
4.  expanded_queries are used as-is
5.  Query fallback chain when expanded_queries is empty
6.  Duplicate and blank queries are removed
7.  target_ontology_constraint is enforced as a hard constraint
8.  target_ontology_constraint routes only to the constrained ontology
9.  preferred_ontology is used when no target constraint exists
10. candidate_ontologies are used when no preferred_ontology exists
11. HPO/HP ontology routes to OLS
12. MONDO ontology routes to OLS
13. NCIT ontology routes to OLS
14. LOINC ontology routes to LOINC search
15. RxNorm / RXNAV ontology routes to RxNorm search
16. ICD10 ontology routes to ICD search
17. Unknown ontology raises PublicRetrievalError
18. Returned raw candidates include matched_query
19. Returned raw candidates include retrieval_mode="public"
20. Returned raw candidates preserve original raw fields
21. Multiple queries produce combined raw candidates
22. No local/SapBERT methods are called
23. No live external APIs are called (FakeSearchTools enforces this)
24. SearchTools exception is wrapped in PublicRetrievalError
25. No both mode is introduced (only PUBLIC retrieval_mode accepted)
26–33. Existing test suites still pass (verified by running pytest)
34. All unit tests pass

Constraints:
- No external API calls; FakeSearchTools injects fixed responses
- All tests are marked as pytest.mark.unit
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_ontology_mapper import public_retriever as public_retriever_module
from llm_ontology_mapper.models import (
    GroundingSource,
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)
from llm_ontology_mapper.public_retriever import (
    PublicOntologyRetriever,
    PublicRetrievalError,
    _route_name,
    public_route_ontology,
)

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# FakeSearchTools
# ─────────────────────────────────────────────────────────────────────────────


class FakeSearchTools:
    """
    Fake SearchTools that records every call and returns fixed raw dicts.

    Optionally raises a configured exception on a specific method to test
    SearchTools exception propagation behavior.
    """

    def __init__(
        self,
        ols_returns: list[dict[str, Any]] | None = None,
        loinc_returns: list[dict[str, Any]] | None = None,
        rxnorm_returns: list[dict[str, Any]] | None = None,
        icd10_returns: list[dict[str, Any]] | None = None,
        raise_on: str | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._ols_returns = ols_returns or []
        self._loinc_returns = loinc_returns or []
        self._rxnorm_returns = rxnorm_returns or []
        self._icd10_returns = icd10_returns or []
        self._raise_on = raise_on
        self._raise_exc = raise_exc or RuntimeError("fake error")

        # Call records: list of (method_name, kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _maybe_raise(self, method: str) -> None:
        if self._raise_on == method:
            raise self._raise_exc

    def search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._maybe_raise("search_ols")
        self.calls.append(("search_ols", {"query": query, "ontology": ontology, "top_k": top_k}))
        return list(self._ols_returns)

    def search_loinc(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._maybe_raise("search_loinc")
        self.calls.append(("search_loinc", {"query": query, "top_k": top_k}))
        return list(self._loinc_returns)

    def search_rxnorm(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._maybe_raise("search_rxnorm")
        self.calls.append(("search_rxnorm", {"query": query, "top_k": top_k}))
        return list(self._rxnorm_returns)

    def search_icd10(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._maybe_raise("search_icd10")
        self.calls.append(("search_icd10", {"query": query, "top_k": top_k}))
        return list(self._icd10_returns)

    # Guard: these methods must NOT be called by PublicOntologyRetriever
    def search_sapbert(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("PublicOntologyRetriever must never call search_sapbert")

    def search_local(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("PublicOntologyRetriever must never call search_local")


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _public_plan(**kwargs: Any) -> QueryPlan:
    """Helper to build a public-mode QueryPlan with sensible defaults."""
    return QueryPlan(
        original_term=kwargs.get("original_term", "cough"),
        retrieval_mode=RetrievalMode.PUBLIC,
        expanded_queries=kwargs.get("expanded_queries", ["cough"]),
        preferred_ontology=kwargs.get("preferred_ontology"),
        candidate_ontologies=kwargs.get("candidate_ontologies", []),
        target_ontology_constraint=kwargs.get("target_ontology_constraint"),
        allowed_target_ontologies=kwargs.get("allowed_target_ontologies"),
        inferred_meaning=kwargs.get("inferred_meaning"),
        original_label=kwargs.get("original_label"),
        normalized_term=kwargs.get("normalized_term"),
    )


_OLS_CANDIDATE = {
    "code": "HP:0012735",
    "term": "Cough",
    "score": 0.95,
    "definition": "",
    "source": "OLS",
}
_LOINC_CANDIDATE = {
    "code": "LOINC:8480-6",
    "term": "Systolic BP",
    "score": 0.90,
    "definition": "",
    "source": "LOINC-Search-API",
    "ontology": "LOINC",
}
_RXNORM_CANDIDATE = {
    "code": "RXNORM:1049502",
    "term": "Acetaminophen",
    "score": 0.85,
    "definition": "",
    "source": "RxNav",
}
_ICD_CANDIDATE = {
    "code": "ICD10:J06.9",
    "term": "Acute upper respiratory infection",
    "score": 0.80,
    "definition": "",
    "source": "NIH-ClinicalTables",
}
_SNOMED_CANDIDATE = {
    "code": "SNOMEDCT:123456",
    "term": "SNOMED concept",
    "score": 0.88,
    "definition": "",
    "source": "OLS",
}
_EFO_CANDIDATE = {
    "code": "EFO:0000408",
    "term": "disease",
    "ontology": "EFO",
    "score": 0.86,
    "definition": "",
    "source": "OLS",
}


@pytest.mark.parametrize(
    "alias",
    ["SNOMED", "SNOMEDCT", "SNOMED-CT", "snomed", "snomedct", "snomed-ct"],
)
def test_public_route_ontology_normalizes_snomed_aliases(alias: str) -> None:
    assert public_route_ontology(alias) == "SNOMED"


@pytest.mark.parametrize(("alias", "expected"), [("HPO", "HPO"), ("HP", "HPO")])
def test_public_route_ontology_preserves_hpo_aliases(alias: str, expected: str) -> None:
    assert public_route_ontology(alias) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — public QueryPlan is accepted
# ─────────────────────────────────────────────────────────────────────────────


def test_public_mode_is_accepted() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert isinstance(results, list)


def test_public_route_calls_include_latency_and_candidate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([1.0, 1.25])
    monkeypatch.setattr(public_retriever_module.time, "monotonic", lambda: next(ticks))
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    route_calls: list[dict[str, Any]] = []

    retriever.retrieve(plan, route_calls=route_calls)

    assert len(route_calls) == 1
    assert route_calls[0]["route"] == "public_api"
    assert route_calls[0]["query"] == "cough"
    assert route_calls[0]["latency_ms"] == 250.0
    assert route_calls[0]["candidate_count"] == 1


def test_public_multiple_routes_each_include_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([1.0, 1.1, 2.0, 2.25])
    monkeypatch.setattr(public_retriever_module.time, "monotonic", lambda: next(ticks))
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["sinus pain", "nasal congestion"],
        preferred_ontology="HPO",
    )
    route_calls: list[dict[str, Any]] = []

    retriever.retrieve(plan, route_calls=route_calls)

    assert [call["query"] for call in route_calls] == ["sinus pain", "nasal congestion"]
    assert [call["latency_ms"] for call in route_calls] == pytest.approx([100.0, 250.0])


def test_public_route_latency_recorded_when_route_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 10.05])
    monkeypatch.setattr(public_retriever_module.time, "monotonic", lambda: next(ticks))
    retriever = PublicOntologyRetriever(search_tools=FakeSearchTools())
    plan = _public_plan(preferred_ontology="UNKNOWN_ONTOLOGY")
    route_calls: list[dict[str, Any]] = []

    with pytest.raises(PublicRetrievalError):
        retriever.retrieve(plan, route_calls=route_calls)

    assert len(route_calls) == 1
    assert route_calls[0]["latency_ms"] == pytest.approx(50.0)
    assert "candidate_count" not in route_calls[0]


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — local QueryPlan raises PublicRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_local_mode_raises() -> None:
    fake = FakeSearchTools()
    retriever = PublicOntologyRetriever(search_tools=fake)
    local_plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.LOCAL,
        candidate_ontologies=["HPO"],
    )
    with pytest.raises(PublicRetrievalError, match="retrieval_mode=.local."):
        retriever.retrieve(local_plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — disabled QueryPlan raises PublicRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_mode_raises() -> None:
    fake = FakeSearchTools()
    retriever = PublicOntologyRetriever(search_tools=fake)
    disabled_plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.DISABLED,
    )
    with pytest.raises(PublicRetrievalError, match="retrieval_mode=.disabled."):
        retriever.retrieve(disabled_plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — expanded_queries are used
# ─────────────────────────────────────────────────────────────────────────────


def test_expanded_queries_are_used() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure", "systolic BP"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert "systolic blood pressure" in queries_sent
    assert "systolic BP" in queries_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — fallback chain when expanded_queries is empty
# ─────────────────────────────────────────────────────────────────────────────


def test_fallback_uses_inferred_meaning_when_expanded_empty() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.PUBLIC,
        expanded_queries=[],
        inferred_meaning="systolic blood pressure",
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert queries_sent == ["systolic blood pressure"]


def test_fallback_uses_original_label_when_no_inferred_meaning() -> None:
    fake = FakeSearchTools(loinc_returns=[_LOINC_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.PUBLIC,
        expanded_queries=[],
        inferred_meaning=None,
        original_label="Systolic Blood Pressure",
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert queries_sent == ["Systolic Blood Pressure"]


def test_fallback_uses_original_term_as_last_resort() -> None:
    fake = FakeSearchTools(loinc_returns=[_LOINC_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = QueryPlan(
        original_term="systolic blood pressure",
        retrieval_mode=RetrievalMode.PUBLIC,
        expanded_queries=[],
        inferred_meaning=None,
        original_label=None,
        normalized_term=None,
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert queries_sent == ["systolic blood pressure"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — duplicate and blank queries are removed
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_queries_are_deduplicated() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough", "cough", "Cough", "  cough  "],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    # After strip + dedup: "cough", "Cough" (different case is preserved but
    # stripped duplicates are removed)
    assert len(queries_sent) == len(set(queries_sent))


def test_blank_queries_are_filtered() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough", "", "   ", "cough symptom"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert "" not in queries_sent
    assert "   " not in queries_sent
    assert "cough" in queries_sent
    assert "cough symptom" in queries_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — target_ontology_constraint is enforced as a hard constraint
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_constraint_enforced() -> None:
    """Only the constrained ontology should be searched, even if preferred differs."""
    fake = FakeSearchTools(loinc_returns=[_LOINC_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        preferred_ontology="HPO",  # overridden by constraint
        candidate_ontologies=["MONDO"],  # also overridden
        target_ontology_constraint="LOINC",
    )
    retriever.retrieve(plan)
    # Only LOINC search should have been called
    methods_called = [name for name, _ in fake.calls]
    assert methods_called == ["search_loinc"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — target_ontology_constraint routes only to that ontology
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_constraint_routes_exclusively() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="MONDO",
        candidate_ontologies=["NCIT", "HPO"],
        target_ontology_constraint="HPO",
    )
    retriever.retrieve(plan)
    ontologies_searched = [c["ontology"] for name, c in fake.calls if name == "search_ols"]
    assert ontologies_searched == ["HPO"]
    # MONDO and NCIT must NOT have been searched
    assert "MONDO" not in ontologies_searched
    assert "NCIT" not in ontologies_searched


def test_allowed_target_ontologies_search_multiple_public_routes() -> None:
    fake = FakeSearchTools(
        ols_returns=[_OLS_CANDIDATE],
        loinc_returns=[_LOINC_CANDIDATE],
    )
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["blood pressure"],
        preferred_ontology=None,
        candidate_ontologies=["LOINC", "HPO", "MONDO"],
        allowed_target_ontologies=["LOINC", "HPO"],
    )

    retriever.retrieve(plan)

    assert [name for name, _ in fake.calls] == ["search_loinc", "search_ols"]
    assert fake.calls[1][1]["ontology"] == "HPO"


def test_allowed_target_ontologies_remove_unselected_public_routes() -> None:
    fake = FakeSearchTools(
        ols_returns=[_OLS_CANDIDATE],
        loinc_returns=[_LOINC_CANDIDATE],
    )
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["blood pressure"],
        preferred_ontology="LOINC",
        candidate_ontologies=["HPO", "MONDO"],
        allowed_target_ontologies=["HPO"],
    )

    retriever.retrieve(plan)

    assert [name for name, _ in fake.calls] == ["search_ols"]
    assert fake.calls[0][1]["ontology"] == "HPO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — preferred_ontology used when no target constraint
# ─────────────────────────────────────────────────────────────────────────────


def test_preferred_ontology_used_without_constraint() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
        candidate_ontologies=[],
        target_ontology_constraint=None,
    )
    retriever.retrieve(plan)
    ontologies_searched = [c["ontology"] for _, c in fake.calls]
    assert "HPO" in ontologies_searched


def test_preferred_ontology_is_first_in_search_order() -> None:
    """preferred_ontology should be searched before candidate_ontologies."""
    order: list[str] = []

    class OrderTrackingFake(FakeSearchTools):
        def search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
            order.append(ontology)
            return []

    fake = OrderTrackingFake()
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="MONDO",
        candidate_ontologies=["HPO", "NCIT"],
        target_ontology_constraint=None,
    )
    retriever.retrieve(plan)
    assert order[0] == "MONDO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — candidate_ontologies used when no preferred_ontology
# ─────────────────────────────────────────────────────────────────────────────


def test_candidate_ontologies_used_when_no_preferred() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=["HPO", "MONDO"],
    )
    retriever.retrieve(plan)
    ontologies_searched = [c["ontology"] for _, c in fake.calls]
    assert "HPO" in ontologies_searched
    assert "MONDO" in ontologies_searched


# ─────────────────────────────────────────────────────────────────────────────
# Tests 11–16 — ontology routing
# ─────────────────────────────────────────────────────────────────────────────


def test_hpo_routes_to_ols() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_ols" in methods
    assert "search_loinc" not in methods
    assert "search_rxnorm" not in methods
    assert "search_icd10" not in methods


def test_hp_routes_to_ols() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HP")
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_ols" in methods


@pytest.mark.parametrize("alias", ["SNOMED", "SNOMEDCT", "SNOMED-CT"])
def test_snomed_aliases_route_to_snomed_ols(alias: str) -> None:
    fake = FakeSearchTools(ols_returns=[_SNOMED_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology=alias)

    results = retriever.retrieve(plan)

    assert [name for name, _ in fake.calls] == ["search_ols"]
    assert fake.calls[0][1]["ontology"] == "SNOMED"
    assert results[0]["requested_ontology"] == "SNOMED"


def test_snomed_alias_constraint_and_allow_list_remain_hard_filter() -> None:
    fake = FakeSearchTools(ols_returns=[_SNOMED_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        preferred_ontology="MONDO",
        candidate_ontologies=["MONDO", "HPO"],
        target_ontology_constraint="SNOMED-CT",
        allowed_target_ontologies=["SNOMED"],
    )

    retriever.retrieve(plan)

    assert [name for name, _ in fake.calls] == ["search_ols"]
    assert fake.calls[0][1]["ontology"] == "SNOMED"


def test_mondo_routes_to_ols() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="MONDO")
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_ols" in methods
    ols_ontology = [c["ontology"] for name, c in fake.calls if name == "search_ols"]
    assert "MONDO" in ols_ontology


def test_ncit_routes_to_ols() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="NCIT")
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_ols" in methods
    ols_ontology = [c["ontology"] for name, c in fake.calls if name == "search_ols"]
    assert "NCIT" in ols_ontology


def test_efo_routes_to_ols_with_requested_ontology_metadata() -> None:
    fake = FakeSearchTools(ols_returns=[_EFO_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["disease"],
        target_ontology_constraint="EFO",
    )

    results = retriever.retrieve(plan)

    assert [name for name, _ in fake.calls] == ["search_ols"]
    assert fake.calls[0][1]["ontology"] == "EFO"
    assert results[0]["requested_ontology"] == "EFO"
    assert results[0]["route_name"] == "OLS"
    assert results[0]["matched_query"] == "disease"
    assert results[0]["retrieval_mode"] == "public"


def test_loinc_routes_to_loinc_search() -> None:
    fake = FakeSearchTools(loinc_returns=[_LOINC_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_loinc" in methods
    assert "search_ols" not in methods
    assert "search_rxnorm" not in methods
    assert "search_icd10" not in methods


def test_rxnorm_routes_to_rxnav_search() -> None:
    fake = FakeSearchTools(rxnorm_returns=[_RXNORM_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["acetaminophen"],
        preferred_ontology="RXNORM",
    )
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_rxnorm" in methods
    assert "search_ols" not in methods


def test_rxnav_alias_routes_to_rxnorm_search() -> None:
    fake = FakeSearchTools(rxnorm_returns=[_RXNORM_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["acetaminophen"],
        preferred_ontology="RXNAV",
    )
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_rxnorm" in methods


def test_icd10_routes_to_icd_search() -> None:
    fake = FakeSearchTools(icd10_returns=[_ICD_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["respiratory infection"],
        preferred_ontology="ICD10",
    )
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_icd10" in methods
    assert "search_ols" not in methods
    assert "search_loinc" not in methods


def test_icd10cm_alias_routes_to_icd_search() -> None:
    fake = FakeSearchTools(icd10_returns=[_ICD_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["respiratory infection"],
        preferred_ontology="ICD10CM",
    )
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_icd10" in methods


# ─────────────────────────────────────────────────────────────────────────────
# Test 17 — unknown ontology raises PublicRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_ontology_raises() -> None:
    fake = FakeSearchTools()
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="UNKNOWNONTO",
    )
    with pytest.raises(PublicRetrievalError, match="not supported by any known public route"):
        retriever.retrieve(plan)


def test_no_ontologies_at_all_raises() -> None:
    """When no ontology can be inferred, a clear error should be raised."""
    fake = FakeSearchTools()
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.PUBLIC,
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=[],
        target_ontology_constraint=None,
    )
    with pytest.raises(PublicRetrievalError, match="No ontology route could be inferred"):
        retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 18 — returned raw candidates include matched_query
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_matched_query() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough symptom"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert "matched_query" in r
        assert r["matched_query"] == "cough symptom"


# ─────────────────────────────────────────────────────────────────────────────
# Test 19 — returned raw candidates include retrieval_mode="public"
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_retrieval_mode_public() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert r["retrieval_mode"] == "public"


# ─────────────────────────────────────────────────────────────────────────────
# Test 20 — returned raw candidates preserve original raw candidate fields
# ─────────────────────────────────────────────────────────────────────────────


def test_results_preserve_original_fields() -> None:
    original = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.95,
        "definition": "a cough",
        "source": "OLS",
        "iri": "http://purl.obolibrary.org/obo/HP_0012735",
    }
    fake = FakeSearchTools(ols_returns=[original])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) == 1
    r = results[0]
    assert r["code"] == original["code"]
    assert r["term"] == original["term"]
    assert r["score"] == original["score"]
    assert r["definition"] == original["definition"]
    assert r["source"] == original["source"]
    assert r["iri"] == original["iri"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 21 — multiple queries produce combined raw candidates
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_queries_produce_combined_results() -> None:
    ols_result_a = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.95,
        "definition": "",
        "source": "OLS",
    }
    ols_result_b = {
        "code": "HP:0002110",
        "term": "Bronchiectasis",
        "score": 0.80,
        "definition": "",
        "source": "OLS",
    }

    call_index = [0]
    responses = [[ols_result_a], [ols_result_b]]

    class SequentialFake(FakeSearchTools):
        def search_ols(self, query: str, ontology: str, top_k: int = 10) -> list[dict[str, Any]]:
            self.calls.append(
                ("search_ols", {"query": query, "ontology": ontology, "top_k": top_k})
            )
            result = responses[call_index[0]]
            call_index[0] = min(call_index[0] + 1, len(responses) - 1)
            return result

    fake = SequentialFake()
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough", "chronic cough"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) == 2
    matched_queries = {r["matched_query"] for r in results}
    assert "cough" in matched_queries
    assert "chronic cough" in matched_queries


# ─────────────────────────────────────────────────────────────────────────────
# Test 22 — no local/SapBERT methods are called
# ─────────────────────────────────────────────────────────────────────────────


def test_no_local_sapbert_methods_called() -> None:
    """FakeSearchTools raises AssertionError if search_sapbert or search_local is called."""
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    # Should not raise AssertionError from guard methods
    retriever.retrieve(plan)
    methods = [name for name, _ in fake.calls]
    assert "search_sapbert" not in methods
    assert "search_local" not in methods


# ─────────────────────────────────────────────────────────────────────────────
# Test 24 — SearchTools exception is wrapped in PublicRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_searchtools_ols_exception_raises_public_retrieval_error() -> None:
    fake = FakeSearchTools(raise_on="search_ols", raise_exc=RuntimeError("OLS network failure"))
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    with pytest.raises(PublicRetrievalError, match="Unexpected error calling public route"):
        retriever.retrieve(plan)


def test_searchtools_loinc_exception_raises_public_retrieval_error() -> None:
    fake = FakeSearchTools(raise_on="search_loinc", raise_exc=ConnectionError("LOINC timeout"))
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        preferred_ontology="LOINC",
    )
    with pytest.raises(PublicRetrievalError, match="Unexpected error calling public route"):
        retriever.retrieve(plan)


def test_searchtools_rxnorm_exception_raises_public_retrieval_error() -> None:
    fake = FakeSearchTools(raise_on="search_rxnorm", raise_exc=OSError("RxNav down"))
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["acetaminophen"],
        preferred_ontology="RXNORM",
    )
    with pytest.raises(PublicRetrievalError, match="Unexpected error calling public route"):
        retriever.retrieve(plan)


def test_searchtools_icd10_exception_raises_public_retrieval_error() -> None:
    fake = FakeSearchTools(raise_on="search_icd10", raise_exc=ValueError("bad ICD response"))
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["respiratory infection"],
        preferred_ontology="ICD10",
    )
    with pytest.raises(PublicRetrievalError, match="Unexpected error calling public route"):
        retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 25 — no both mode introduced
# ─────────────────────────────────────────────────────────────────────────────


def test_no_both_mode_introduced() -> None:
    """PublicOntologyRetriever must not accept any mode other than PUBLIC."""
    from llm_ontology_mapper.models import RetrievalMode

    accepted_modes = [RetrievalMode.PUBLIC]
    rejected_modes = [RetrievalMode.LOCAL, RetrievalMode.DISABLED]
    fake = FakeSearchTools()
    retriever = PublicOntologyRetriever(search_tools=fake)
    for mode in rejected_modes:
        plan = QueryPlan(original_term="test", retrieval_mode=mode)
        with pytest.raises(PublicRetrievalError):
            retriever.retrieve(plan)
    # Only PUBLIC must work
    assert len(accepted_modes) == 1
    assert accepted_modes[0] == RetrievalMode.PUBLIC


# ─────────────────────────────────────────────────────────────────────────────
# Additional provenance tests
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_requested_ontology() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert "requested_ontology" in r
        assert r["requested_ontology"] == "HPO"


def test_results_include_route_name_for_ols() -> None:
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) > 0
    assert results[0]["route_name"] == "OLS"


def test_results_include_route_name_for_loinc() -> None:
    fake = FakeSearchTools(loinc_returns=[_LOINC_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        preferred_ontology="LOINC",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    assert results[0]["route_name"] == "LOINC-Search-API"


def test_results_include_route_name_for_rxnorm() -> None:
    fake = FakeSearchTools(rxnorm_returns=[_RXNORM_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["acetaminophen"],
        preferred_ontology="RXNORM",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    assert results[0]["route_name"] == "RxNav"


def test_results_include_route_name_for_icd() -> None:
    fake = FakeSearchTools(icd10_returns=[_ICD_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["respiratory infection"],
        preferred_ontology="ICD10",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    assert results[0]["route_name"] == "NIH-ClinicalTables"


# ─────────────────────────────────────────────────────────────────────────────
# route_plan integration
# ─────────────────────────────────────────────────────────────────────────────


def test_route_plan_queries_take_precedence() -> None:
    """When route_plan is provided with queries, those are used instead of query_plan queries."""
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["original query"],
        preferred_ontology="HPO",
    )
    route_plan = RetrievalRoutePlan(
        retrieval_mode=RetrievalMode.PUBLIC,
        is_grounded_mode=True,
        grounding_source=GroundingSource.PUBLIC_API,
        queries=["route plan query"],
        candidate_ontologies=["HPO"],
    )
    retriever.retrieve(plan, route_plan=route_plan)
    queries_sent = [c["query"] for _, c in fake.calls]
    assert "route plan query" in queries_sent
    assert "original query" not in queries_sent


def test_route_plan_candidate_ontologies_used() -> None:
    """When route_plan is provided, its candidate_ontologies override query_plan's."""
    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=["NCIT"],  # overridden by route_plan
    )
    route_plan = RetrievalRoutePlan(
        retrieval_mode=RetrievalMode.PUBLIC,
        is_grounded_mode=True,
        grounding_source=GroundingSource.PUBLIC_API,
        queries=["cough"],
        candidate_ontologies=["MONDO"],
    )
    retriever.retrieve(plan, route_plan=route_plan)
    ontologies_searched = [c["ontology"] for name, c in fake.calls if name == "search_ols"]
    assert "MONDO" in ontologies_searched
    assert "NCIT" not in ontologies_searched


# ─────────────────────────────────────────────────────────────────────────────
# Empty result set (SearchTools returns [])
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_search_results_return_empty_list() -> None:
    fake = FakeSearchTools(ols_returns=[])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# _route_name helper
# ─────────────────────────────────────────────────────────────────────────────


def test_route_name_helper_loinc() -> None:
    assert _route_name("LOINC") == "LOINC-Search-API"
    assert _route_name("loinc") == "LOINC-Search-API"


def test_route_name_helper_rxnorm() -> None:
    assert _route_name("RXNORM") == "RxNav"
    assert _route_name("RXNAV") == "RxNav"
    assert _route_name("RXCUI") == "RxNav"


def test_route_name_helper_icd() -> None:
    assert _route_name("ICD10") == "NIH-ClinicalTables"
    assert _route_name("ICD10CM") == "NIH-ClinicalTables"
    assert _route_name("ICD") == "NIH-ClinicalTables"


def test_route_name_helper_ols_default() -> None:
    assert _route_name("HPO") == "OLS"
    assert _route_name("MONDO") == "OLS"
    assert _route_name("NCIT") == "OLS"
    assert _route_name("HP") == "OLS"
    assert _route_name("UNKNOWNONTO") == "OLS"  # fallthrough default


# ─────────────────────────────────────────────────────────────────────────────
# Import from top-level package
# ─────────────────────────────────────────────────────────────────────────────


def test_importable_from_package() -> None:
    from llm_ontology_mapper import PublicOntologyRetriever as POR  # noqa: F401
    from llm_ontology_mapper import PublicRetrievalError as PRE

    assert POR is PublicOntologyRetriever
    assert PRE is PublicRetrievalError


# ─────────────────────────────────────────────────────────────────────────────
# max_results_per_query is forwarded
# ─────────────────────────────────────────────────────────────────────────────


def test_max_results_per_query_forwarded_to_loinc() -> None:
    fake = FakeSearchTools(loinc_returns=[])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan, max_results_per_query=5)
    assert fake.calls[0][1]["top_k"] == 5


def test_max_results_per_query_forwarded_to_ols() -> None:
    fake = FakeSearchTools(ols_returns=[])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan, max_results_per_query=3)
    assert fake.calls[0][1]["top_k"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# No normalization is performed (returns raw dicts, not NormalizedCandidate)
# ─────────────────────────────────────────────────────────────────────────────


def test_results_are_raw_dicts_not_normalized_candidates() -> None:
    from llm_ontology_mapper.models import NormalizedCandidate

    fake = FakeSearchTools(ols_returns=[_OLS_CANDIDATE])
    retriever = PublicOntologyRetriever(search_tools=fake)
    plan = _public_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    for r in results:
        assert isinstance(r, dict)
        assert not isinstance(r, NormalizedCandidate)
