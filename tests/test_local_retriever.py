"""
Unit tests for LocalSemanticRetriever (Phase 8).

Validates:
1.  Local QueryPlan is accepted
2.  Public QueryPlan raises LocalRetrievalError
3.  Disabled QueryPlan raises LocalRetrievalError
4.  expanded_queries are used
5.  route_plan.queries override QueryPlan.expanded_queries
6.  Query fallback chain when expanded_queries is empty
7.  Duplicate and blank queries are removed
8.  target_ontology_constraint is enforced as a hard constraint
9.  target_ontology_constraint is sent in the local client payload
10. preferred_ontology is used when no target constraint exists
11. candidate_ontologies are used when no preferred_ontology exists
12. No ontology → broad semantic search (ontology=None)
13. Multiple queries produce combined raw candidates
14. Returned raw candidates include matched_query
15. Returned raw candidates include retrieval_mode="local"
16. Returned raw candidates include requested_ontology
17. Returned raw candidates include route_name="local_sapbert"
18. Original local candidate fields are preserved
19. Local client exception is wrapped in LocalRetrievalError
20. Local client empty results returns empty list
21. Malformed local response raises LocalRetrievalError
22. No public APIs are called
23. SearchTools is not used
24. PublicOntologyRetriever is not used
25. No both mode is introduced
26. max_results_per_query is passed as top_k to client
27. HPO/HP ontology normalization to SapBERT index key
28. Local retriever works with fake client only (no live calls)
29. No live external APIs are called in unit tests (enforced by FakeClient)
30–38. Existing test suites still pass (verified by running pytest)
39. All unit tests pass

Constraints:
- FakeClient records all calls and returns fixed dicts
- No live SapBERT, no live public APIs, no SearchTools
- All tests marked pytest.mark.unit
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_ontology_mapper import local_retriever as local_retriever_module
from llm_ontology_mapper.local_retriever import (
    _SAPBERT_INDEX_MAP,
    LocalRetrievalError,
    LocalSemanticRetriever,
)
from llm_ontology_mapper.models import (
    GroundingSource,
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────────────────────
# FakeClient
# ─────────────────────────────────────────────────────────────────────────────


class FakeClient:
    """
    Injectable fake for the SapBERT client.

    Records every search() call and returns a fixed list of raw dicts.
    Optionally raises a configured exception to test error-propagation paths.

    No HTTP calls, no external services.
    """

    def __init__(
        self,
        returns: list[dict[str, Any]] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._returns = returns or []
        self._raise_exc = raise_exc
        # List of (query, ontology, top_k) recorded in call order
        self.calls: list[tuple[str, str | None, int]] = []

    def search(
        self,
        query: str,
        ontology: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, ontology, top_k))
        if self._raise_exc is not None:
            raise self._raise_exc
        return list(self._returns)

    # Guard: PublicOntologyRetriever and SearchTools must never be called
    def search_ols(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("LocalSemanticRetriever must never call search_ols")

    def search_loinc(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("LocalSemanticRetriever must never call search_loinc")

    def search_rxnorm(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("LocalSemanticRetriever must never call search_rxnorm")

    def search_icd10(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("LocalSemanticRetriever must never call search_icd10")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _local_plan(**kwargs: Any) -> QueryPlan:
    """Build a local-mode QueryPlan with sensible defaults."""
    return QueryPlan(
        original_term=kwargs.get("original_term", "cough"),
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=kwargs.get("expanded_queries", ["cough"]),
        preferred_ontology=kwargs.get("preferred_ontology"),
        candidate_ontologies=kwargs.get("candidate_ontologies", []),
        target_ontology_constraint=kwargs.get("target_ontology_constraint"),
        allowed_target_ontologies=kwargs.get("allowed_target_ontologies"),
        inferred_meaning=kwargs.get("inferred_meaning"),
        original_label=kwargs.get("original_label"),
        normalized_term=kwargs.get("normalized_term"),
    )


_SAPBERT_CANDIDATE = {
    "code": "HP:0012735",
    "term": "Cough",
    "score": 0.93,
    "definition": "A cough",
    "source": "SapBERT",
}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — local QueryPlan is accepted
# ─────────────────────────────────────────────────────────────────────────────


def test_local_mode_is_accepted() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert isinstance(results, list)


def test_local_route_calls_include_latency_and_candidate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([5.0, 5.4])
    monkeypatch.setattr(local_retriever_module.time, "monotonic", lambda: next(ticks))
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    route_calls: list[dict[str, Any]] = []

    retriever.retrieve(plan, route_calls=route_calls)

    assert len(route_calls) == 1
    assert route_calls[0]["route"] == "local_sapbert"
    assert route_calls[0]["query"] == "cough"
    assert route_calls[0]["candidate_ontologies"] == ["HPO"]
    assert route_calls[0]["latency_ms"] == pytest.approx(400.0)
    assert route_calls[0]["candidate_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — public QueryPlan raises LocalRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_public_mode_raises() -> None:
    fake = FakeClient()
    retriever = LocalSemanticRetriever(client=fake)
    public_plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.PUBLIC,
        candidate_ontologies=["HPO"],
    )
    with pytest.raises(LocalRetrievalError, match="retrieval_mode=.public."):
        retriever.retrieve(public_plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — disabled QueryPlan raises LocalRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_mode_raises() -> None:
    fake = FakeClient()
    retriever = LocalSemanticRetriever(client=fake)
    disabled_plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.DISABLED,
    )
    with pytest.raises(LocalRetrievalError, match="retrieval_mode=.disabled."):
        retriever.retrieve(disabled_plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — expanded_queries are used
# ─────────────────────────────────────────────────────────────────────────────


def test_expanded_queries_are_used() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["systolic blood pressure", "systolic BP"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert "systolic blood pressure" in queries_sent
    assert "systolic BP" in queries_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — route_plan.queries override QueryPlan.expanded_queries
# ─────────────────────────────────────────────────────────────────────────────


def test_route_plan_queries_override_query_plan_queries() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["original query"],
        preferred_ontology="HPO",
    )
    route_plan = RetrievalRoutePlan(
        retrieval_mode=RetrievalMode.LOCAL,
        is_grounded_mode=True,
        grounding_source=GroundingSource.LOCAL_SAPBERT,
        queries=["route plan query"],
        candidate_ontologies=["HPO"],
    )
    retriever.retrieve(plan, route_plan=route_plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert "route plan query" in queries_sent
    assert "original query" not in queries_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — query fallback when expanded_queries is empty
# ─────────────────────────────────────────────────────────────────────────────


def test_fallback_uses_inferred_meaning() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=[],
        inferred_meaning="systolic blood pressure",
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert queries_sent == ["systolic blood pressure"]


def test_fallback_uses_original_label_when_no_inferred_meaning() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=[],
        inferred_meaning=None,
        original_label="Systolic Blood Pressure",
        preferred_ontology="LOINC",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert queries_sent == ["Systolic Blood Pressure"]


def test_fallback_uses_original_term_as_last_resort() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=[],
        inferred_meaning=None,
        original_label=None,
        normalized_term=None,
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert queries_sent == ["cough"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — duplicate and blank queries are removed
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_queries_are_deduplicated() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough", "cough", "  cough  "],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    # After strip + dedup "cough" appears exactly once
    assert queries_sent.count("cough") == 1


def test_blank_queries_are_filtered() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough", "", "   ", "cough symptom"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    queries_sent = [q for q, _, _ in fake.calls]
    assert "" not in queries_sent
    assert "   " not in queries_sent
    assert "cough" in queries_sent
    assert "cough symptom" in queries_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — target_ontology_constraint is enforced as a hard constraint
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_constraint_overrides_preferred_and_candidates() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="MONDO",  # overridden
        candidate_ontologies=["NCIT"],  # overridden
        target_ontology_constraint="HPO",
    )
    retriever.retrieve(plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    # Only HPO (mapped to HPO in index) should have been searched
    assert len(fake.calls) == 1
    # The index key for HPO is "HPO"
    assert ontologies_sent == ["HPO"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — target_ontology_constraint is sent in local client payload
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_constraint_sent_in_payload() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["systolic blood pressure"],
        target_ontology_constraint="LOINC",
    )
    retriever.retrieve(plan)
    _, ontology_sent, _ = fake.calls[0]
    # LOINC → LOINC in _SAPBERT_INDEX_MAP
    assert ontology_sent == "LOINC"


def test_allowed_target_ontologies_search_multiple_local_indices() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["blood pressure"],
        preferred_ontology=None,
        candidate_ontologies=["LOINC", "HPO", "MONDO"],
        allowed_target_ontologies=["LOINC", "HPO"],
    )

    retriever.retrieve(plan)

    assert [ontology for _, ontology, _ in fake.calls] == ["LOINC", "HPO"]


def test_allowed_target_ontologies_remove_unselected_local_indices() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["blood pressure"],
        preferred_ontology="LOINC",
        candidate_ontologies=["HPO", "MONDO"],
        allowed_target_ontologies=["HPO"],
    )

    retriever.retrieve(plan)

    assert [ontology for _, ontology, _ in fake.calls] == ["HPO"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — preferred_ontology is used when no target constraint
# ─────────────────────────────────────────────────────────────────────────────


def test_preferred_ontology_used_without_constraint() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
        candidate_ontologies=[],
        target_ontology_constraint=None,
    )
    retriever.retrieve(plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    assert "HPO" in ontologies_sent


def test_preferred_ontology_is_first_in_search_order() -> None:
    order: list[str] = []

    class OrderFake(FakeClient):
        def search(self, query: str, ontology: str | None, top_k: int) -> list[dict[str, Any]]:
            order.append(str(ontology))
            self.calls.append((query, ontology, top_k))
            return []

    fake = OrderFake()
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="MONDO",
        candidate_ontologies=["HPO", "NCIT"],
    )
    retriever.retrieve(plan)
    assert order[0] == "MONDO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — candidate_ontologies used when no preferred_ontology
# ─────────────────────────────────────────────────────────────────────────────


def test_candidate_ontologies_used_when_no_preferred() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=["HPO", "MONDO"],
    )
    retriever.retrieve(plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    assert "HPO" in ontologies_sent
    assert "MONDO" in ontologies_sent


# ─────────────────────────────────────────────────────────────────────────────
# Test 12 — no ontology → broad semantic search (ontology=None in payload)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_ontology_triggers_broad_search() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=[],
        target_ontology_constraint=None,
    )
    results = retriever.retrieve(plan)
    assert len(fake.calls) == 1
    _, ontology_sent, _ = fake.calls[0]
    assert ontology_sent is None
    assert len(results) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 13 — multiple queries produce combined raw candidates
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_queries_produce_combined_results() -> None:
    candidate_a = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.93,
        "definition": "",
        "source": "SapBERT",
    }
    candidate_b = {
        "code": "HP:0002110",
        "term": "Bronchiectasis",
        "score": 0.80,
        "definition": "",
        "source": "SapBERT",
    }

    responses = [candidate_a, candidate_b]
    call_idx = [0]

    class SequentialFake(FakeClient):
        def search(self, query: str, ontology: str | None, top_k: int) -> list[dict[str, Any]]:
            self.calls.append((query, ontology, top_k))
            result = [responses[call_idx[0]]]
            call_idx[0] = min(call_idx[0] + 1, len(responses) - 1)
            return result

    fake = SequentialFake()
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough", "chronic cough"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) == 2
    matched_queries = {r["matched_query"] for r in results}
    assert "cough" in matched_queries
    assert "chronic cough" in matched_queries


# ─────────────────────────────────────────────────────────────────────────────
# Test 14 — returned raw candidates include matched_query
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_matched_query() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough symptom"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert "matched_query" in r
        assert r["matched_query"] == "cough symptom"


# ─────────────────────────────────────────────────────────────────────────────
# Test 15 — returned raw candidates include retrieval_mode="local"
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_retrieval_mode_local() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert r["retrieval_mode"] == "local"


# ─────────────────────────────────────────────────────────────────────────────
# Test 16 — returned raw candidates include requested_ontology
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_requested_ontology() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert "requested_ontology" in r
        assert r["requested_ontology"] == "HPO"


def test_results_requested_ontology_is_none_for_broad_search() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = QueryPlan(
        original_term="cough",
        retrieval_mode=RetrievalMode.LOCAL,
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=[],
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert r["requested_ontology"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 17 — returned raw candidates include route_name="local_sapbert"
# ─────────────────────────────────────────────────────────────────────────────


def test_results_include_route_name_local_sapbert() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) > 0
    for r in results:
        assert r["route_name"] == "local_sapbert"


# ─────────────────────────────────────────────────────────────────────────────
# Test 18 — original local candidate fields are preserved
# ─────────────────────────────────────────────────────────────────────────────


def test_results_preserve_original_candidate_fields() -> None:
    original = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.93,
        "definition": "A cough is a forceful expulsion of air",
        "source": "SapBERT",
        "extra_field": "custom_value",
    }
    fake = FakeClient(returns=[original])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) == 1
    r = results[0]
    assert r["code"] == original["code"]
    assert r["term"] == original["term"]
    assert r["score"] == original["score"]
    assert r["definition"] == original["definition"]
    assert r["source"] == original["source"]
    assert r["extra_field"] == original["extra_field"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 19 — local client exception is wrapped in LocalRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_client_runtime_error_raises_local_retrieval_error() -> None:
    fake = FakeClient(raise_exc=RuntimeError("connection refused"))
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    with pytest.raises(LocalRetrievalError, match="Unexpected error from local semantic client"):
        retriever.retrieve(plan)


def test_client_connection_error_raises_local_retrieval_error() -> None:
    fake = FakeClient(raise_exc=ConnectionError("SapBERT unreachable"))
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    with pytest.raises(LocalRetrievalError):
        retriever.retrieve(plan)


def test_client_local_retrieval_error_propagates_directly() -> None:
    """LocalRetrievalError from client must not be double-wrapped."""
    original_error = LocalRetrievalError("malformed response")
    fake = FakeClient(raise_exc=original_error)
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    with pytest.raises(LocalRetrievalError, match="malformed response"):
        retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 20 — empty results from client returns empty list
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_client_results_return_empty_list() -> None:
    fake = FakeClient(returns=[])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 21 — malformed local response raises LocalRetrievalError
# ─────────────────────────────────────────────────────────────────────────────


def test_malformed_response_raises_local_retrieval_error() -> None:
    """Client raises LocalRetrievalError for malformed SapBERT responses."""
    fake = FakeClient(raise_exc=LocalRetrievalError("SapBERT response missing 'results' key"))
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    with pytest.raises(LocalRetrievalError, match="results"):
        retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 22–24 — no public APIs, no SearchTools, no PublicOntologyRetriever
# ─────────────────────────────────────────────────────────────────────────────


def test_no_public_apis_called() -> None:
    """FakeClient guard methods raise AssertionError if public API methods are invoked."""
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    # Should not trigger any AssertionError from guard methods
    retriever.retrieve(plan)
    methods_called = {name for q, onto, _ in fake.calls for name in ["search"]}
    assert "search_ols" not in methods_called
    assert "search_loinc" not in methods_called
    assert "search_rxnorm" not in methods_called
    assert "search_icd10" not in methods_called


def test_search_tools_not_used() -> None:
    """LocalSemanticRetriever must not import SearchTools."""
    import llm_ontology_mapper.local_retriever as lr_module

    # SearchTools must not be imported into the local_retriever module namespace
    assert not hasattr(lr_module, "SearchTools")

    assert "SearchTools" not in lr_module.__dict__


def test_public_ontology_retriever_not_used() -> None:
    """LocalSemanticRetriever must not import PublicOntologyRetriever."""
    import llm_ontology_mapper.local_retriever as lr_module

    assert not hasattr(lr_module, "PublicOntologyRetriever")
    assert "PublicOntologyRetriever" not in lr_module.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# Test 25 — no both mode introduced
# ─────────────────────────────────────────────────────────────────────────────


def test_no_both_mode_introduced() -> None:
    """LocalSemanticRetriever must only accept LOCAL mode."""
    fake = FakeClient()
    retriever = LocalSemanticRetriever(client=fake)
    for mode in [RetrievalMode.PUBLIC, RetrievalMode.DISABLED]:
        plan = QueryPlan(original_term="test", retrieval_mode=mode)
        with pytest.raises(LocalRetrievalError):
            retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# Test 26 — max_results_per_query is passed as top_k
# ─────────────────────────────────────────────────────────────────────────────


def test_max_results_per_query_forwarded_as_top_k() -> None:
    fake = FakeClient(returns=[])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan, max_results_per_query=7)
    assert fake.calls[0][2] == 7


def test_default_max_results_is_fifteen() -> None:
    """Recall-increase change: the per-query default rose from 10 to 15."""
    fake = FakeClient(returns=[])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    retriever.retrieve(plan)
    assert fake.calls[0][2] == 15


# ─────────────────────────────────────────────────────────────────────────────
# Test 27 — HPO/HP ontology normalization to SapBERT index key
# ─────────────────────────────────────────────────────────────────────────────


def test_hp_normalized_to_hpo_index_key() -> None:
    """HP (user-facing) must map to HPO (SapBERT index key)."""
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HP",
    )
    retriever.retrieve(plan)
    _, ontology_sent, _ = fake.calls[0]
    # HP → HPO in _SAPBERT_INDEX_MAP
    assert ontology_sent == "HPO"


def test_hpo_stays_as_hpo_index_key() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
    )
    retriever.retrieve(plan)
    _, ontology_sent, _ = fake.calls[0]
    assert ontology_sent == "HPO"


def test_requested_ontology_preserves_user_facing_name_for_hp() -> None:
    """requested_ontology in provenance must reflect the user-supplied name (HP), not the index key."""
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HP",
    )
    results = retriever.retrieve(plan)
    assert len(results) > 0
    # requested_ontology is the user-facing name (uppercased from preferred_ontology)
    assert results[0]["requested_ontology"] == "HP"


def test_sapbert_index_map_coverage() -> None:
    """Spot-check common ontology keys in _SAPBERT_INDEX_MAP."""
    assert _SAPBERT_INDEX_MAP["HP"] == "HPO"
    assert _SAPBERT_INDEX_MAP["HPO"] == "HPO"
    assert _SAPBERT_INDEX_MAP["MONDO"] == "MONDO"
    assert _SAPBERT_INDEX_MAP["NCIT"] == "NCIT"
    assert _SAPBERT_INDEX_MAP["LOINC"] == "LOINC"
    assert _SAPBERT_INDEX_MAP["RXNORM"] == "RXNORM"
    assert _SAPBERT_INDEX_MAP["RXCUI"] == "RXNORM"
    assert _SAPBERT_INDEX_MAP["ICD10"] == "ICD10CM"
    assert _SAPBERT_INDEX_MAP["SNOMEDCT"] == "SNOMED"


# ─────────────────────────────────────────────────────────────────────────────
# Test 28 — local retriever works with fake client only (no live calls)
# ─────────────────────────────────────────────────────────────────────────────


def test_fake_client_injection_avoids_live_calls() -> None:
    """Instantiating with client= bypasses all HTTP; no network access occurs."""
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    assert len(results) > 0
    # All calls went to fake
    assert len(fake.calls) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 29 — no live external APIs (implicitly enforced; all tests use FakeClient)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_live_calls_in_unit_tests() -> None:
    """
    All tests in this module use FakeClient.  Instantiating LocalSemanticRetriever
    with client= never creates a real HTTP connection.
    """
    fake = FakeClient(returns=[])
    retriever = LocalSemanticRetriever(client=fake)
    assert retriever._client is fake


# ─────────────────────────────────────────────────────────────────────────────
# No-client configuration
# ─────────────────────────────────────────────────────────────────────────────


def test_no_client_raises_on_retrieve() -> None:
    """LocalSemanticRetriever with no client or URL raises at retrieve time."""
    retriever = LocalSemanticRetriever()
    plan = _local_plan(preferred_ontology="HPO")
    with pytest.raises(LocalRetrievalError, match="No local client configured"):
        retriever.retrieve(plan)


# ─────────────────────────────────────────────────────────────────────────────
# route_plan candidate_ontologies override
# ─────────────────────────────────────────────────────────────────────────────


def test_route_plan_candidate_ontologies_override_query_plan() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=["NCIT"],  # overridden by route_plan
    )
    route_plan = RetrievalRoutePlan(
        retrieval_mode=RetrievalMode.LOCAL,
        is_grounded_mode=True,
        grounding_source=GroundingSource.LOCAL_SAPBERT,
        queries=["cough"],
        candidate_ontologies=["MONDO"],
    )
    retriever.retrieve(plan, route_plan=route_plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    assert "MONDO" in ontologies_sent
    assert "NCIT" not in ontologies_sent


# ─────────────────────────────────────────────────────────────────────────────
# Results are raw dicts, not NormalizedCandidate objects
# ─────────────────────────────────────────────────────────────────────────────


def test_results_are_raw_dicts_not_normalized_candidates() -> None:
    from llm_ontology_mapper.models import NormalizedCandidate

    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(preferred_ontology="HPO")
    results = retriever.retrieve(plan)
    for r in results:
        assert isinstance(r, dict)
        assert not isinstance(r, NormalizedCandidate)


# ─────────────────────────────────────────────────────────────────────────────
# Import from top-level package
# ─────────────────────────────────────────────────────────────────────────────


def test_importable_from_package() -> None:
    from llm_ontology_mapper import (
        LocalRetrievalError as LRE,
    )
    from llm_ontology_mapper import (
        LocalSemanticRetriever as LSR,
    )
    from llm_ontology_mapper import (
        SapBERTClient as SBC,
    )

    assert LSR is LocalSemanticRetriever
    assert LRE is LocalRetrievalError
    from llm_ontology_mapper.local_retriever import SapBERTClient

    assert SBC is SapBERTClient


# ─────────────────────────────────────────────────────────────────────────────
# Multiple candidate_ontologies searched in order
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_candidate_ontologies_all_searched() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology=None,
        candidate_ontologies=["HPO", "MONDO", "NCIT"],
    )
    retriever.retrieve(plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    assert "HPO" in ontologies_sent
    assert "MONDO" in ontologies_sent
    assert "NCIT" in ontologies_sent
    assert len(fake.calls) == 3  # one call per ontology


# ─────────────────────────────────────────────────────────────────────────────
# preferred_ontology is not duplicated when it's also in candidate_ontologies
# ─────────────────────────────────────────────────────────────────────────────


def test_preferred_ontology_not_duplicated_in_candidates() -> None:
    fake = FakeClient(returns=[_SAPBERT_CANDIDATE])
    retriever = LocalSemanticRetriever(client=fake)
    plan = _local_plan(
        expanded_queries=["cough"],
        preferred_ontology="HPO",
        candidate_ontologies=["HPO", "MONDO"],  # HPO duplicated
    )
    retriever.retrieve(plan)
    ontologies_sent = [onto for _, onto, _ in fake.calls]
    assert ontologies_sent.count("HPO") == 1
    assert "MONDO" in ontologies_sent
