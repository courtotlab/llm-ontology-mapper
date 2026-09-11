"""
Unit tests for RetrievalRouter (retrieval_router.py).

No external APIs are called — all inputs are constructed QueryPlan objects.

Run with:  pytest tests/test_retrieval_router.py -v -m unit
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.models import (
    GroundingSource,
    LogicType,
    MappingResult,
    QueryPlan,
    RetrievalMode,
    RetrievalRoutePlan,
)
from llm_ontology_mapper.retrieval_router import RetrievalRouter

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def router() -> RetrievalRouter:
    return RetrievalRouter()


def _public_plan(**kwargs) -> QueryPlan:
    defaults = dict(
        original_term="sys_bp",
        expanded_queries=["systolic blood pressure", "systolic BP"],
        candidate_ontologies=["LOINC", "HPO"],
        preferred_ontology="LOINC",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def _local_plan(**kwargs) -> QueryPlan:
    defaults = dict(
        original_term="sys_bp",
        expanded_queries=["systolic blood pressure", "systolic BP"],
        candidate_ontologies=["LOINC", "HPO"],
        preferred_ontology="LOINC",
        retrieval_mode=RetrievalMode.LOCAL,
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def _disabled_plan(**kwargs) -> QueryPlan:
    defaults = dict(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.DISABLED,
        retrieval_disabled_reason="Retrieval disabled by caller",
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 1–3: basic mode routing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_public_plan_returns_public_route_plan(router: RetrievalRouter) -> None:
    plan = _public_plan()
    route = router.route(plan)

    assert isinstance(route, RetrievalRoutePlan)
    assert route.retrieval_mode == RetrievalMode.PUBLIC
    assert route.is_grounded_mode is True
    assert route.retrieval_skipped is False


@pytest.mark.unit
def test_local_plan_returns_local_route_plan(router: RetrievalRouter) -> None:
    plan = _local_plan()
    route = router.route(plan)

    assert isinstance(route, RetrievalRoutePlan)
    assert route.retrieval_mode == RetrievalMode.LOCAL
    assert route.is_grounded_mode is True
    assert route.retrieval_skipped is False


@pytest.mark.unit
def test_disabled_plan_returns_skipped_ungrounded_route_plan(router: RetrievalRouter) -> None:
    plan = _disabled_plan()
    route = router.route(plan)

    assert isinstance(route, RetrievalRoutePlan)
    assert route.retrieval_mode == RetrievalMode.DISABLED
    assert route.is_grounded_mode is False
    assert route.retrieval_skipped is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests 4–5: expanded_queries pass through
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_public_mode_uses_expanded_queries(router: RetrievalRouter) -> None:
    plan = _public_plan(expanded_queries=["systolic blood pressure", "systolic BP"])
    route = router.route(plan)

    assert route.queries == ["systolic blood pressure", "systolic BP"]


@pytest.mark.unit
def test_local_mode_uses_expanded_queries(router: RetrievalRouter) -> None:
    plan = _local_plan(expanded_queries=["systolic blood pressure", "SBP"])
    route = router.route(plan)

    assert route.queries == ["systolic blood pressure", "SBP"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: disabled mode has empty route_calls
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disabled_mode_has_empty_route_calls(router: RetrievalRouter) -> None:
    route = router.route(_disabled_plan())

    assert route.route_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: no both mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_both_mode(router: RetrievalRouter) -> None:
    # RetrievalMode enum must not have BOTH
    assert not hasattr(RetrievalMode, "BOTH")

    # "both" is not a valid retrieval mode value
    with pytest.raises(ValueError):
        RetrievalMode("both")

    # A public route plan uses only public_api, never local_sapbert
    route = router.route(_public_plan())
    assert route.grounding_source == GroundingSource.PUBLIC_API
    assert route.grounding_source != GroundingSource.LOCAL_SAPBERT

    # A local route plan uses only local_sapbert, never public_api
    route_local = router.route(_local_plan())
    assert route_local.grounding_source == GroundingSource.LOCAL_SAPBERT
    assert route_local.grounding_source != GroundingSource.PUBLIC_API


# ─────────────────────────────────────────────────────────────────────────────
# Tests 8–9: target_ontology_constraint is a hard constraint
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_restricts_candidate_ontologies(router: RetrievalRouter) -> None:
    plan = _public_plan(
        candidate_ontologies=["LOINC", "HPO"],
        target_ontology_constraint="HPO",
    )
    route = router.route(plan)

    assert route.candidate_ontologies == ["HPO"]
    assert "LOINC" not in route.candidate_ontologies
    assert route.target_ontology_constraint == "HPO"


@pytest.mark.unit
def test_target_ontology_preserved_in_route_calls(router: RetrievalRouter) -> None:
    plan = _public_plan(
        expanded_queries=["systolic blood pressure"],
        target_ontology_constraint="HPO",
        allowed_target_ontologies=["HPO"],
    )
    route = router.route(plan)

    assert route.route_calls
    assert route.allowed_target_ontologies == ["HPO"]
    for call in route.route_calls:
        assert call["target_ontology"] == "HPO"
        assert call["allowed_target_ontologies"] == ["HPO"]
        assert call["candidate_ontologies"] == ["HPO"]


@pytest.mark.unit
def test_allowed_target_ontologies_restrict_candidates(router: RetrievalRouter) -> None:
    plan = _public_plan(
        candidate_ontologies=["LOINC", "HPO", "MONDO"],
        target_ontology_constraint=None,
        allowed_target_ontologies=["HPO", "MONDO"],
    )
    route = router.route(plan)

    assert route.allowed_target_ontologies == ["HPO", "MONDO"]
    assert route.candidate_ontologies == ["HPO", "MONDO"]
    for call in route.route_calls:
        assert call["allowed_target_ontologies"] == ["HPO", "MONDO"]
        assert call["candidate_ontologies"] == ["HPO", "MONDO"]


@pytest.mark.unit
def test_allowed_target_ontologies_fall_back_when_no_candidate_survives(
    router: RetrievalRouter,
) -> None:
    plan = _public_plan(
        candidate_ontologies=["LOINC"],
        target_ontology_constraint=None,
        allowed_target_ontologies=["HPO", "MONDO"],
    )
    route = router.route(plan)

    assert route.candidate_ontologies == ["HPO", "MONDO"]
    for call in route.route_calls:
        assert call["candidate_ontologies"] == ["HPO", "MONDO"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests 10–11: query fallback
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_query_fallback_when_expanded_queries_empty(router: RetrievalRouter) -> None:
    # No expanded_queries — should fall back to inferred_meaning first
    plan = QueryPlan(
        original_term="sys_bp",
        inferred_meaning="systolic blood pressure",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    route = router.route(plan)

    assert route.queries == ["systolic blood pressure"]


@pytest.mark.unit
def test_query_fallback_priority_order(router: RetrievalRouter) -> None:
    # inferred_meaning takes priority over original_label and original_term
    plan = QueryPlan(
        original_term="sbp",
        original_label="Systolic Label",
        normalized_term="sbp normalized",
        inferred_meaning="systolic blood pressure",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    route = router.route(plan)

    assert route.queries == ["systolic blood pressure"]


@pytest.mark.unit
def test_query_fallback_uses_original_label_if_no_inferred_meaning(
    router: RetrievalRouter,
) -> None:
    plan = QueryPlan(
        original_term="sbp",
        original_label="Systolic BP reading",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    route = router.route(plan)

    assert route.queries == ["Systolic BP reading"]


@pytest.mark.unit
def test_query_fallback_uses_original_term_as_last_resort(
    router: RetrievalRouter,
) -> None:
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.PUBLIC,
    )
    route = router.route(plan)

    assert route.queries == ["sys_bp"]


@pytest.mark.unit
def test_query_fallback_produces_unique_nonblank_queries(router: RetrievalRouter) -> None:
    plan = _public_plan(
        expanded_queries=[
            "blood pressure",
            "blood pressure",  # duplicate
            "systolic BP",
            "",  # blank
            "   ",  # whitespace-only
        ],
    )
    route = router.route(plan)

    assert "" not in route.queries
    for q in route.queries:
        assert q.strip() == q and q  # no blanks, no leading/trailing whitespace
    assert route.queries.count("blood pressure") == 1
    assert len(route.queries) == 2  # "blood pressure" and "systolic BP"


# ─────────────────────────────────────────────────────────────────────────────
# Tests 12–14: grounding_source per mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_public_route_grounding_source_is_public_api(router: RetrievalRouter) -> None:
    route = router.route(_public_plan())
    assert route.grounding_source == GroundingSource.PUBLIC_API


@pytest.mark.unit
def test_local_route_grounding_source_is_local_sapbert(router: RetrievalRouter) -> None:
    route = router.route(_local_plan())
    assert route.grounding_source == GroundingSource.LOCAL_SAPBERT


@pytest.mark.unit
def test_disabled_route_grounding_source_is_none(router: RetrievalRouter) -> None:
    route = router.route(_disabled_plan())
    assert route.grounding_source == GroundingSource.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Tests 15–17: retrieval_skipped and is_grounded_mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disabled_retrieval_skipped(router: RetrievalRouter) -> None:
    route = router.route(_disabled_plan())
    assert route.retrieval_skipped is True


@pytest.mark.unit
def test_disabled_is_grounded_mode_false(router: RetrievalRouter) -> None:
    route = router.route(_disabled_plan())
    assert route.is_grounded_mode is False


@pytest.mark.unit
@pytest.mark.parametrize("plan_fn", [_public_plan, _local_plan])
def test_public_local_is_grounded_mode_true(router: RetrievalRouter, plan_fn) -> None:
    route = router.route(plan_fn())
    assert route.is_grounded_mode is True
    assert route.retrieval_skipped is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: no external API calls
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_external_api_calls(router: RetrievalRouter) -> None:
    # Routing must complete without network access.
    # route_calls are descriptions only — they carry no live response data.
    route = router.route(_public_plan())

    assert route.route_calls
    for call in route.route_calls:
        assert "route" in call
        assert "query" in call
        # No response payload — just a plan
        assert "response" not in call
        assert "status_code" not in call
        assert "candidates" not in call


# ─────────────────────────────────────────────────────────────────────────────
# Tests 19–20: existing tests unaffected
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_query_planner_imports_unaffected() -> None:
    from llm_ontology_mapper.query_planner import QueryPlanner, QueryPlanningError

    assert QueryPlanner is not None
    assert QueryPlanningError is not None


@pytest.mark.unit
def test_existing_model_imports_unaffected() -> None:

    # Phase 1 models should still construct cleanly
    plan = QueryPlan(original_term="cough", retrieval_mode=RetrievalMode.PUBLIC)
    assert plan.route_public_apis is True

    r = MappingResult(
        source_term="cough",
        target_code="HP:0012735",
        target_term="Cough",
        ontology="HPO",
        confidence=0.93,
        logic_type=LogicType.RAG,
    )
    assert r.target_code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge-case tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_route_calls_contain_route_name_public(router: RetrievalRouter) -> None:
    route = router.route(_public_plan(expanded_queries=["systolic blood pressure"]))

    assert len(route.route_calls) == 1
    assert route.route_calls[0]["route"] == "public_api"
    assert route.route_calls[0]["query"] == "systolic blood pressure"


@pytest.mark.unit
def test_route_calls_contain_route_name_local(router: RetrievalRouter) -> None:
    route = router.route(_local_plan(expanded_queries=["systolic blood pressure"]))

    assert len(route.route_calls) == 1
    assert route.route_calls[0]["route"] == "local_sapbert"
    assert route.route_calls[0]["query"] == "systolic blood pressure"


@pytest.mark.unit
def test_one_route_call_per_query(router: RetrievalRouter) -> None:
    plan = _public_plan(expanded_queries=["systolic blood pressure", "systolic BP", "SBP"])
    route = router.route(plan)

    assert len(route.route_calls) == 3
    queries_in_calls = [rc["query"] for rc in route.route_calls]
    assert queries_in_calls == ["systolic blood pressure", "systolic BP", "SBP"]


@pytest.mark.unit
def test_disabled_reason_preserved_from_query_plan(router: RetrievalRouter) -> None:
    plan = _disabled_plan(retrieval_disabled_reason="user opted out of retrieval")
    route = router.route(plan)

    assert route.retrieval_disabled_reason == "user opted out of retrieval"


@pytest.mark.unit
def test_disabled_reason_set_when_absent(router: RetrievalRouter) -> None:
    plan = QueryPlan(
        original_term="sys_bp",
        retrieval_mode=RetrievalMode.DISABLED,
        # No retrieval_disabled_reason supplied
    )
    route = router.route(plan)

    assert route.retrieval_disabled_reason is not None
    assert len(route.retrieval_disabled_reason) > 0


@pytest.mark.unit
def test_candidate_ontologies_used_when_no_target_constraint(
    router: RetrievalRouter,
) -> None:
    plan = _public_plan(
        candidate_ontologies=["LOINC", "HPO", "MONDO"],
        target_ontology_constraint=None,
    )
    route = router.route(plan)

    assert route.candidate_ontologies == ["LOINC", "HPO", "MONDO"]
    assert route.target_ontology_constraint is None
    assert route.allowed_target_ontologies is None


@pytest.mark.unit
def test_route_plan_serialises_to_json(router: RetrievalRouter) -> None:
    route = router.route(_public_plan())
    dumped = route.model_dump(mode="json")

    assert isinstance(dumped, dict)
    assert dumped["retrieval_mode"] == "public"
    assert dumped["grounding_source"] == "public_api"
    assert isinstance(dumped["route_calls"], list)
