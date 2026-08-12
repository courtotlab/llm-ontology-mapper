"""
Unit tests for CandidateMerger (candidate_merger.py).

No external APIs are called — all inputs are NormalizedCandidate objects
constructed directly from field values.

Run with:  pytest tests/test_candidate_merger.py -v -m unit
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_ontology_mapper.candidate_merger import CandidateMergeError, CandidateMerger
from llm_ontology_mapper.models import NormalizedCandidate, RetrievalMode

# ─────────────────────────────────────────────────────────────────────────────
# Shared factory helper
# ─────────────────────────────────────────────────────────────────────────────


def _c(
    code: str = "HP:0012735",
    term: str = "Cough",
    ontology: str = "HPO",
    source: str = "OLS",
    matched_query: str = "cough",
    retrieval_mode: RetrievalMode = RetrievalMode.PUBLIC,
    raw_score: float | None = None,
    normalized_score: float | None = None,
    definition: str | None = None,
    provenance: dict | None = None,
    retrieved_from_ontologies: list[str] | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        code=code,
        term=term,
        ontology=ontology,
        source=source,
        matched_query=matched_query,
        retrieval_mode=retrieval_mode,
        raw_score=raw_score,
        normalized_score=normalized_score,
        definition=definition,
        provenance=provenance,
        retrieved_from_ontologies=retrieved_from_ontologies or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def merger() -> CandidateMerger:
    return CandidateMerger()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: merge returns candidates unchanged when there are no duplicates
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_duplicates_returns_all(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", term="Cough")
    b = _c(code="HP:0002110", term="Bronchiectasis")
    c = _c(code="HP:0001250", term="Seizure")

    result = merger.merge([a, b, c])

    assert len(result) == 3
    # Original objects are returned when there is no merging to do
    codes = {r.code for r in result}
    assert codes == {"HP:0012735", "HP:0002110", "HP:0001250"}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: merge deduplicates same ontology + code
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_deduplicates_same_ontology_and_code(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.9)
    b = _c(code="HP:0012735", normalized_score=0.7)

    result = merger.merge([a, b])

    assert len(result) == 1
    assert result[0].code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: merge treats ontology casing as equivalent
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ontology_casing_equivalent(merger: CandidateMerger) -> None:
    # NormalizedCandidate uppercases ontology on construction, so "hpo" → "HPO"
    a = _c(ontology="hpo", code="HP:0012735", normalized_score=0.9)
    b = _c(ontology="HPO", code="HP:0012735", normalized_score=0.7)

    result = merger.merge([a, b])

    assert len(result) == 1
    assert result[0].ontology == "HPO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: merge treats code casing as equivalent when appropriate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_code_casing_equivalent(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.9)
    b = _c(code="hp:0012735", normalized_score=0.7)  # lowercase code

    result = merger.merge([a, b])

    assert len(result) == 1
    # Winner is a (higher score), so its code form is kept
    assert result[0].code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: merge does NOT dedupe same term with different code
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_dedup_same_term_different_code(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", term="Cough")
    b = _c(code="HP:0002835", term="Cough")  # different code, same term

    result = merger.merge([a, b])

    assert len(result) == 2
    codes = {r.code for r in result}
    assert codes == {"HP:0012735", "HP:0002835"}


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: merge does NOT dedupe same code across different ontologies
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_dedup_same_code_different_ontologies(merger: CandidateMerger) -> None:
    a = _c(ontology="HPO", code="HP:0012735")
    b = _c(ontology="LOINC", code="8480-6")

    result = merger.merge([a, b])

    assert len(result) == 2
    ontologies = {r.ontology for r in result}
    assert ontologies == {"HPO", "LOINC"}


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: duplicate keeps higher normalized_score
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_duplicate_keeps_higher_normalized_score(merger: CandidateMerger) -> None:
    low = _c(code="HP:0012735", normalized_score=0.6, term="Cough A")
    high = _c(code="HP:0012735", normalized_score=0.9, term="Cough B")

    result = merger.merge([low, high])

    assert len(result) == 1
    assert result[0].normalized_score == pytest.approx(0.9)
    assert result[0].term == "Cough B"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: duplicate keeps higher raw_score when normalized_score is missing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_duplicate_keeps_higher_raw_score_when_no_normalized(
    merger: CandidateMerger,
) -> None:
    low = _c(code="HP:0012735", raw_score=0.5, term="Cough A")
    high = _c(code="HP:0012735", raw_score=0.8, term="Cough B")

    result = merger.merge([low, high])

    assert len(result) == 1
    assert result[0].raw_score == pytest.approx(0.8)
    assert result[0].term == "Cough B"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: duplicate keeps candidate with definition when scores tie
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_duplicate_keeps_candidate_with_definition_on_score_tie(
    merger: CandidateMerger,
) -> None:
    no_def = _c(code="HP:0012735", normalized_score=0.8, definition=None, term="Cough A")
    has_def = _c(
        code="HP:0012735",
        normalized_score=0.8,
        definition="A cough.",
        term="Cough B",
    )

    result = merger.merge([no_def, has_def])

    assert len(result) == 1
    assert result[0].definition == "A cough."
    assert result[0].term == "Cough B"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: duplicate keeps earlier candidate when all else ties
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_duplicate_keeps_earlier_candidate_when_all_tied(
    merger: CandidateMerger,
) -> None:
    first = _c(code="HP:0012735", normalized_score=0.8, definition="Def.", term="First")
    second = _c(code="HP:0012735", normalized_score=0.8, definition="Def.", term="Second")

    result = merger.merge([first, second])

    assert len(result) == 1
    assert result[0].term == "First"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: merged provenance includes both original provenance records
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_merged_provenance_includes_both_provenances(merger: CandidateMerger) -> None:
    prov_a = {"raw_candidate": {"code": "HP:0012735"}, "retrieval_mode": "public"}
    prov_b = {"raw_candidate": {"code": "HP:0012735"}, "retrieval_mode": "local"}

    a = _c(code="HP:0012735", normalized_score=0.9, provenance=prov_a)
    b = _c(code="HP:0012735", normalized_score=0.7, provenance=prov_b)

    result = merger.merge([a, b])
    assert len(result) == 1

    prov = result[0].provenance
    assert isinstance(prov, dict)
    assert prov.get("merged") is True

    sources = prov["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 2

    all_provenances = [s["provenance"] for s in sources]
    assert prov_a in all_provenances
    assert prov_b in all_provenances


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: merged provenance preserves matched_query from discarded duplicate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_merged_provenance_preserves_discarded_matched_query(
    merger: CandidateMerger,
) -> None:
    winner = _c(
        code="HP:0012735",
        normalized_score=0.9,
        matched_query="cough",
    )
    discarded = _c(
        code="HP:0012735",
        normalized_score=0.5,
        matched_query="expulsive cough reflex",
    )

    result = merger.merge([winner, discarded])
    assert len(result) == 1

    prov = result[0].provenance
    assert isinstance(prov, dict)
    matched_queries = prov["matched_queries"]
    assert "cough" in matched_queries
    assert "expulsive cough reflex" in matched_queries


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: sorting uses normalized_score descending
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sorting_by_normalized_score_descending(merger: CandidateMerger) -> None:
    low = _c(code="HP:0000001", term="A", normalized_score=0.3)
    mid = _c(code="HP:0000002", term="B", normalized_score=0.7)
    high = _c(code="HP:0000003", term="C", normalized_score=0.95)

    result = merger.merge([low, mid, high])

    assert [r.normalized_score for r in result] == [
        pytest.approx(0.95),
        pytest.approx(0.7),
        pytest.approx(0.3),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: sorting falls back to raw_score descending
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sorting_falls_back_to_raw_score(merger: CandidateMerger) -> None:
    low = _c(code="HP:0000001", term="A", raw_score=0.2)
    high = _c(code="HP:0000002", term="B", raw_score=0.9)

    result = merger.merge([low, high])

    assert result[0].raw_score == pytest.approx(0.9)
    assert result[1].raw_score == pytest.approx(0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: sorting handles None scores (scored candidates sort before None)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sorting_handles_none_scores(merger: CandidateMerger) -> None:
    no_score = _c(code="HP:0000001", term="A")
    scored = _c(code="HP:0000002", term="B", normalized_score=0.5)
    also_no_score = _c(code="HP:0000003", term="C")

    result = merger.merge([no_score, scored, also_no_score])

    # Scored candidate comes first; None-scored come after, alphabetically
    assert result[0].normalized_score == pytest.approx(0.5)
    assert result[1].term == "A"
    assert result[2].term == "C"


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: target_ontology_constraint filters candidates
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_filters(merger: CandidateMerger) -> None:
    hpo = _c(ontology="HPO", code="HP:0012735", term="Cough")
    loinc = _c(ontology="LOINC", code="8480-6", term="Systolic BP")
    mondo = _c(ontology="MONDO", code="MONDO:0005180", term="Parkinson")

    result = merger.merge([hpo, loinc, mondo], target_ontology_constraint="LOINC")

    assert len(result) == 1
    assert result[0].ontology == "LOINC"
    assert result[0].code == "8480-6"


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: target_ontology_constraint is case-insensitive
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("constraint", ["hpo", "HPO", "Hpo", "hPO"])
def test_target_ontology_constraint_case_insensitive(
    merger: CandidateMerger, constraint: str
) -> None:
    hpo = _c(ontology="HPO", code="HP:0012735")
    loinc = _c(ontology="LOINC", code="8480-6")

    result = merger.merge([hpo, loinc], target_ontology_constraint=constraint)

    assert len(result) == 1
    assert result[0].ontology == "HPO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: target_ontology_constraint can return an empty list
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_target_ontology_can_return_empty_list(merger: CandidateMerger) -> None:
    hpo = _c(ontology="HPO", code="HP:0012735")
    loinc = _c(ontology="LOINC", code="8480-6")

    result = merger.merge([hpo, loinc], target_ontology_constraint="MONDO")

    assert result == []


@pytest.mark.unit
def test_allowed_target_ontologies_keep_several_selected_candidates(
    merger: CandidateMerger,
) -> None:
    hpo = _c(code="HP:0012735", ontology="HPO")
    loinc = _c(code="LOINC:8480-6", ontology="LOINC")
    mondo = _c(code="MONDO:0000001", ontology="MONDO")

    result = merger.merge(
        [hpo, loinc, mondo],
        allowed_target_ontologies=["HPO", "LOINC"],
    )

    assert {candidate.ontology for candidate in result} == {"HPO", "LOINC"}


@pytest.mark.unit
def test_allowed_target_ontologies_remove_unselected_candidates(
    merger: CandidateMerger,
) -> None:
    hpo = _c(code="HP:0012735", ontology="HPO")
    loinc = _c(code="LOINC:8480-6", ontology="LOINC")

    result = merger.merge(
        [hpo, loinc],
        allowed_target_ontologies=["LOINC"],
    )

    assert result == [loinc]


@pytest.mark.unit
def test_allowed_target_ontologies_none_is_unrestricted(
    merger: CandidateMerger,
) -> None:
    hpo = _c(code="HP:0012735", ontology="HPO")
    loinc = _c(code="LOINC:8480-6", ontology="LOINC")

    result = merger.merge(
        [hpo, loinc],
        allowed_target_ontologies=None,
    )

    assert {candidate.ontology for candidate in result} == {"HPO", "LOINC"}


@pytest.mark.unit
def test_hp_candidate_from_mondo_scoped_search_removed_when_only_mondo_allowed(
    merger: CandidateMerger,
) -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", term="Asthma")

    result = merger.merge([candidate], allowed_target_ontologies=["MONDO"])

    assert result == []


@pytest.mark.unit
def test_hp_candidate_from_mondo_scoped_search_remains_when_hpo_allowed(
    merger: CandidateMerger,
) -> None:
    candidate = _c(code="HP:0002099", ontology="HPO", term="Asthma")

    result = merger.merge([candidate], allowed_target_ontologies=["MONDO", "HPO"])

    assert result == [candidate]


@pytest.mark.unit
def test_efo_target_keeps_imported_mondo_candidate_retrieved_from_efo(
    merger: CandidateMerger,
) -> None:
    candidate = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["EFO"],
    )

    result = merger.merge([candidate], target_ontology_constraint="EFO")

    assert result == [candidate]


@pytest.mark.unit
def test_efo_target_keeps_imported_hpo_candidate_retrieved_from_efo(
    merger: CandidateMerger,
) -> None:
    candidate = _c(
        code="HP:0002099",
        ontology="HPO",
        term="Asthma",
        retrieved_from_ontologies=["EFO"],
    )

    result = merger.merge([candidate], allowed_target_ontologies=["EFO"])

    assert result == [candidate]


@pytest.mark.unit
def test_efo_target_removes_mondo_candidate_not_retrieved_from_efo(
    merger: CandidateMerger,
) -> None:
    candidate = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["MONDO"],
    )

    result = merger.merge([candidate], target_ontology_constraint="EFO")

    assert result == []


@pytest.mark.unit
def test_mondo_target_still_removes_hpo_candidate_retrieved_from_mondo(
    merger: CandidateMerger,
) -> None:
    candidate = _c(
        code="HP:0002099",
        ontology="HPO",
        term="Asthma",
        retrieved_from_ontologies=["MONDO"],
    )

    result = merger.merge([candidate], target_ontology_constraint="MONDO")

    assert result == []


@pytest.mark.unit
def test_multi_target_keeps_native_and_efo_retrieved_imported_candidates(
    merger: CandidateMerger,
) -> None:
    native_loinc = _c(code="LOINC:8480-6", ontology="LOINC", term="Systolic BP")
    native_efo = _c(code="EFO:0000408", ontology="EFO", term="disease")
    imported_mondo = _c(
        code="MONDO:0004975",
        ontology="MONDO",
        term="asthma",
        retrieved_from_ontologies=["EFO"],
    )
    unrelated_mondo = _c(
        code="MONDO:0000001",
        ontology="MONDO",
        term="unrelated disease",
        retrieved_from_ontologies=["MONDO"],
    )

    result = merger.merge(
        [native_loinc, native_efo, imported_mondo, unrelated_mondo],
        allowed_target_ontologies=["EFO", "LOINC"],
    )

    assert {candidate.code for candidate in result} == {
        "LOINC:8480-6",
        "EFO:0000408",
        "MONDO:0004975",
    }


@pytest.mark.unit
def test_duplicate_merge_preserves_retrieved_from_ontologies(
    merger: CandidateMerger,
) -> None:
    from_efo = _c(
        code="HP:0002099",
        ontology="HPO",
        term="Asthma",
        normalized_score=0.8,
        retrieved_from_ontologies=["EFO"],
    )
    from_hpo = _c(
        code="HP:0002099",
        ontology="HPO",
        term="Asthma",
        normalized_score=0.7,
        retrieved_from_ontologies=["HPO"],
    )

    result = merger.merge([from_efo, from_hpo])

    assert len(result) == 1
    assert result[0].retrieved_from_ontologies == ["EFO", "HPO"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: max_candidates limits after merge/sort/filter
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_max_candidates_limits_result(merger: CandidateMerger) -> None:
    candidates = [
        _c(code=f"HP:{i:07d}", term=f"Term {i}", normalized_score=1.0 - (i * 0.1)) for i in range(6)
    ]

    result = merger.merge(candidates, max_candidates=3)

    assert len(result) == 3
    # First three should be the highest-scoring ones
    assert result[0].normalized_score == pytest.approx(1.0)
    assert result[1].normalized_score == pytest.approx(0.9)
    assert result[2].normalized_score == pytest.approx(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: empty input returns empty list
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_empty_input_returns_empty_list(merger: CandidateMerger) -> None:
    result = merger.merge([])

    assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: invalid input object raises a clear error
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_input",
    [
        "not a candidate",
        42,
        {"code": "HP:0012735", "term": "Cough"},  # raw dict, not NormalizedCandidate
        None,
    ],
)
def test_invalid_input_raises(merger: CandidateMerger, bad_input) -> None:
    valid = _c()
    with pytest.raises(CandidateMergeError, match="NormalizedCandidate"):
        merger.merge([valid, bad_input])


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: no both mode is introduced
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_both_mode_introduced(merger: CandidateMerger) -> None:
    from llm_ontology_mapper.models import RetrievalMode

    assert not hasattr(RetrievalMode, "BOTH")

    public = _c(retrieval_mode=RetrievalMode.PUBLIC)
    local = _c(
        code="HP:0002110",
        term="Bronchiectasis",
        retrieval_mode=RetrievalMode.LOCAL,
    )

    result = merger.merge([public, local])
    assert len(result) == 2

    modes = {r.retrieval_mode for r in result}
    assert modes == {RetrievalMode.PUBLIC, RetrievalMode.LOCAL}
    assert RetrievalMode.DISABLED not in modes


# ─────────────────────────────────────────────────────────────────────────────
# Test 23: public and local candidates are both accepted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_accepts_public_and_local_candidates(merger: CandidateMerger) -> None:
    public = _c(
        code="HP:0012735",
        retrieval_mode=RetrievalMode.PUBLIC,
        source="OLS",
        normalized_score=0.9,
    )
    local = _c(
        code="HP:0002110",
        term="Bronchiectasis",
        retrieval_mode=RetrievalMode.LOCAL,
        source="SapBERT",
        normalized_score=0.85,
    )

    result = merger.merge([public, local])

    assert len(result) == 2
    assert result[0].retrieval_mode == RetrievalMode.PUBLIC  # higher score first
    assert result[1].retrieval_mode == RetrievalMode.LOCAL


# ─────────────────────────────────────────────────────────────────────────────
# Test 24: merger makes no external calls
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_external_calls(merger: CandidateMerger) -> None:
    candidates = [_c(code=f"HP:{i:07d}", term=f"Term {i}") for i in range(5)]

    # If any external call were made this would fail without network access
    result = merger.merge(candidates)

    assert len(result) == 5
    for r in result:
        assert isinstance(r, NormalizedCandidate)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 25–28: existing pipeline tests unaffected
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_normalizer_imports_unaffected() -> None:
    from llm_ontology_mapper.candidate_normalizer import (
        CandidateNormalizationError,
        CandidateNormalizer,
    )

    assert CandidateNormalizer is not None
    assert CandidateNormalizationError is not None


@pytest.mark.unit
def test_existing_retrieval_router_imports_unaffected() -> None:
    from llm_ontology_mapper.retrieval_router import RetrievalRouter

    assert RetrievalRouter is not None


@pytest.mark.unit
def test_existing_query_planner_imports_unaffected() -> None:
    from llm_ontology_mapper.query_planner import QueryPlanner, QueryPlanningError

    assert QueryPlanner is not None
    assert QueryPlanningError is not None


@pytest.mark.unit
def test_existing_model_imports_unaffected() -> None:
    from llm_ontology_mapper.models import (
        LogicType,
        MappingResult,
        QueryPlan,
        RetrievalMode,
    )

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
def test_three_way_duplicate_resolved_to_best(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.4)
    b = _c(code="HP:0012735", normalized_score=0.95)
    c = _c(code="HP:0012735", normalized_score=0.7)

    result = merger.merge([a, b, c])

    assert len(result) == 1
    assert result[0].normalized_score == pytest.approx(0.95)


@pytest.mark.unit
def test_three_way_duplicate_provenance_has_count(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.9)
    b = _c(code="HP:0012735", normalized_score=0.7)
    c = _c(code="HP:0012735", normalized_score=0.5)

    result = merger.merge([a, b, c])

    prov = result[0].provenance
    assert isinstance(prov, dict)
    assert prov["duplicate_count"] == 3
    assert len(prov["sources"]) == 3


@pytest.mark.unit
def test_max_candidates_zero_returns_empty(merger: CandidateMerger) -> None:
    candidates = [_c(code=f"HP:{i:07d}", term=f"Term {i}") for i in range(3)]

    result = merger.merge(candidates, max_candidates=0)

    assert result == []


@pytest.mark.unit
def test_sort_tiebreaker_is_alphabetical_term(merger: CandidateMerger) -> None:
    # Both have same score — should be sorted by term alphabetically
    z = _c(code="HP:0000001", term="Zebra", normalized_score=0.5)
    a = _c(code="HP:0000002", term="Apple", normalized_score=0.5)
    m = _c(code="HP:0000003", term="Mango", normalized_score=0.5)

    result = merger.merge([z, a, m])

    assert [r.term for r in result] == ["Apple", "Mango", "Zebra"]


@pytest.mark.unit
def test_merged_candidate_is_frozen(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.9)
    b = _c(code="HP:0012735", normalized_score=0.5)

    result = merger.merge([a, b])

    with pytest.raises(ValidationError):
        result[0].code = "CHANGED"  # type: ignore[misc]


@pytest.mark.unit
def test_merged_candidate_serialises_to_json(merger: CandidateMerger) -> None:
    a = _c(code="HP:0012735", normalized_score=0.9, matched_query="cough")
    b = _c(code="HP:0012735", normalized_score=0.5, matched_query="dry cough")

    result = merger.merge([a, b])
    assert len(result) == 1

    dumped = result[0].model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert dumped["code"] == "HP:0012735"
    assert dumped["retrieval_mode"] == "public"
    assert dumped["provenance"]["merged"] is True


@pytest.mark.unit
def test_filter_and_dedup_and_limit_combined(merger: CandidateMerger) -> None:
    # Mix of HPO (with a duplicate) and LOINC, requesting HPO with max 1
    hpo_a = _c(ontology="HPO", code="HP:0012735", term="Cough", normalized_score=0.9)
    hpo_b = _c(ontology="HPO", code="HP:0012735", term="Cough", normalized_score=0.6)
    hpo_c = _c(ontology="HPO", code="HP:0001250", term="Seizure", normalized_score=0.7)
    loinc = _c(ontology="LOINC", code="8480-6", term="SBP", normalized_score=0.95)

    result = merger.merge(
        [hpo_a, hpo_b, hpo_c, loinc],
        target_ontology_constraint="HPO",
        max_candidates=1,
    )

    # After dedup: 2 HPO + 1 LOINC → filter → 2 HPO → sort → top 1
    assert len(result) == 1
    assert result[0].ontology == "HPO"
    assert result[0].code == "HP:0012735"
    assert result[0].normalized_score == pytest.approx(0.9)


@pytest.mark.unit
def test_single_candidate_passes_through_unchanged(merger: CandidateMerger) -> None:
    c = _c(code="HP:0012735", normalized_score=0.8)

    result = merger.merge([c])

    assert len(result) == 1
    assert result[0] is c  # same object — no unnecessary reconstruction


@pytest.mark.unit
def test_normalizer_and_merger_integration(merger: CandidateMerger) -> None:
    from llm_ontology_mapper.candidate_normalizer import CandidateNormalizer

    normalizer = CandidateNormalizer()

    raw_candidates = [
        {
            "code": "HP:0012735",
            "term": "Cough",
            "score": 0.9,
            "definition": "A cough.",
            "source": "OLS",
        },
        {
            "code": "HP:0012735",
            "term": "Cough",  # duplicate from second query
            "score": 0.75,
            "source": "OLS",
        },
        {"code": "HP:0001250", "term": "Seizure", "score": 0.85, "source": "OLS"},
    ]

    normalized_a, normalized_b, normalized_c = [
        normalizer.normalize(
            r,
            retrieval_mode=RetrievalMode.PUBLIC,
            matched_query="cough" if i < 2 else "seizure",
            default_ontology="HPO",
        )
        for i, r in enumerate(raw_candidates)
    ]

    merged = merger.merge([normalized_a, normalized_b, normalized_c])

    # Duplicate merged, leaving 2 candidates
    assert len(merged) == 2
    # Best-scored cough wins
    assert merged[0].code == "HP:0012735"
    assert merged[0].normalized_score == pytest.approx(0.9)
    assert merged[1].code == "HP:0001250"
