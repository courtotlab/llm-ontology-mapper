"""
Unit tests for CandidateNormalizer (candidate_normalizer.py).

No external APIs are called — all inputs are constructed raw dicts.

Run with:  pytest tests/test_candidate_normalizer.py -v -m unit
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_ontology_mapper.candidate_normalizer import (
    CandidateNormalizationError,
    CandidateNormalizer,
)
from llm_ontology_mapper.models import NormalizedCandidate, RetrievalMode

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def normalizer() -> CandidateNormalizer:
    return CandidateNormalizer()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: normalizes public candidate with code/term/ontology/source/score
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalizes_public_candidate(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": 0.95,
        "definition": "A cough.",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.PUBLIC,
        matched_query="cough",
    )

    assert isinstance(candidate, NormalizedCandidate)
    assert candidate.code == "HP:0012735"
    assert candidate.term == "Cough"
    assert candidate.ontology == "HPO"
    assert candidate.source == "OLS"
    assert candidate.raw_score == 0.95
    assert candidate.definition == "A cough."
    assert candidate.retrieval_mode == RetrievalMode.PUBLIC


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: normalizes local candidate with code/term/ontology/source/similarity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalizes_local_candidate(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "SapBERT",
        "similarity": 0.88,
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.LOCAL,
        matched_query="cough",
    )

    assert isinstance(candidate, NormalizedCandidate)
    assert candidate.retrieval_mode == RetrievalMode.LOCAL
    assert candidate.source == "SapBERT"
    assert candidate.raw_score == 0.88


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: rejects disabled retrieval_mode
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("mode", [RetrievalMode.DISABLED, "disabled"])
def test_rejects_disabled_retrieval_mode(normalizer: CandidateNormalizer, mode) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}
    with pytest.raises(CandidateNormalizationError, match="disabled"):
        normalizer.normalize(raw, retrieval_mode=mode, matched_query="cough")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: rejects missing code
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rejects_missing_code(normalizer: CandidateNormalizer) -> None:
    raw = {"term": "Cough", "ontology": "HPO", "source": "OLS"}
    with pytest.raises(CandidateNormalizationError, match="code"):
        normalizer.normalize(raw, retrieval_mode="public", matched_query="cough")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: rejects missing term
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rejects_missing_term(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "ontology": "HPO", "source": "OLS"}
    with pytest.raises(CandidateNormalizationError, match="term"):
        normalizer.normalize(raw, retrieval_mode="public", matched_query="cough")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: rejects missing ontology when no default_ontology is provided
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rejects_missing_ontology_with_no_default(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "0012735", "term": "Cough", "source": "OLS"}
    with pytest.raises(CandidateNormalizationError, match="ontology"):
        normalizer.normalize(raw, retrieval_mode="public", matched_query="cough")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: uses default_ontology when ontology is missing from raw dict
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_uses_default_ontology(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "source": "OLS"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
        default_ontology="HPO",
    )

    assert candidate.ontology == "HPO"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: uses default_source when source is missing from raw dict
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_uses_default_source(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
        default_source="MyRetriever",
    )

    assert candidate.source == "MyRetriever"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: infers source for public mode when missing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_infers_source_public_mode(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.PUBLIC,
        matched_query="cough",
    )

    assert candidate.source == "public_api"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: infers source for local mode when missing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_infers_source_local_mode(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.LOCAL,
        matched_query="cough",
    )

    assert candidate.source == "local_sapbert"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: converts numeric string score to float
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_converts_numeric_string_score(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": "0.87",  # string, not float
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.raw_score == pytest.approx(0.87)
    assert isinstance(candidate.raw_score, float)


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: leaves invalid score as None
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_leaves_invalid_score_as_none(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": "not-a-number",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.raw_score is None
    assert candidate.normalized_score is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: uses raw score as normalized_score when between 0 and 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_uses_raw_score_as_normalized_score_in_range(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": 0.75,
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.raw_score == pytest.approx(0.75)
    assert candidate.normalized_score == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: does not use raw score as normalized_score when outside 0..1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("score", [1.5, -0.1, 100.0, -5.0])
def test_does_not_use_out_of_range_score_as_normalized(
    normalizer: CandidateNormalizer, score: float
) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": score,
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.raw_score == pytest.approx(score)
    assert candidate.normalized_score is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: extracts first definition from a definitions list
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extracts_first_definition_from_list(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "definitions": ["A cough is an expulsive reflex.", "Secondary definition."],
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.definition == "A cough is an expulsive reflex."


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: preserves matched_query
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_preserves_matched_query(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="expulsive cough reflex",
    )

    assert candidate.matched_query == "expulsive cough reflex"


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: preserves raw candidate in provenance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_preserves_raw_candidate_in_provenance(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": 0.9,
        "extra_field": "keep_me",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert isinstance(candidate.provenance, dict)
    assert candidate.provenance["raw_candidate"] is raw
    assert candidate.provenance["raw_candidate"]["extra_field"] == "keep_me"
    assert candidate.provenance["retrieval_mode"] == "public"
    assert candidate.provenance["matched_query"] == "cough"


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: normalizes ontology casing to uppercase
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ontology_input", "code", "expected"),
    [
        ("hpo", "HP:0012735", "HPO"),
        ("HPO", "HP:0012735", "HPO"),
        ("Hpo", "HP:0012735", "HPO"),
        ("loinc", "LOINC:8480-6", "LOINC"),
        ("mondo", "MONDO:0004979", "MONDO"),
    ],
)
def test_normalizes_ontology_casing(
    normalizer: CandidateNormalizer,
    ontology_input: str,
    code: str,
    expected: str,
) -> None:
    raw = {
        "code": code,
        "term": "Cough",
        "ontology": ontology_input,
        "source": "OLS",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.ontology == expected


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: supports code alias 'id'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_supports_code_alias_id(normalizer: CandidateNormalizer) -> None:
    raw = {
        "id": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.code == "HP:0012735"


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: supports code alias 'curie'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_supports_code_alias_curie(normalizer: CandidateNormalizer) -> None:
    raw = {
        "curie": "MONDO:0005180",
        "term": "Parkinson disease",
        "ontology": "MONDO",
        "source": "OLS",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="Parkinson disease",
    )

    assert candidate.code == "MONDO:0005180"


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: supports term alias 'label'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_supports_term_alias_label(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "label": "Cough",
        "ontology": "HPO",
        "source": "OLS",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.term == "Cough"


# ─────────────────────────────────────────────────────────────────────────────
# Test 22: supports term alias 'display'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_supports_term_alias_display(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "LOINC:8480-6",
        "display": "Systolic blood pressure",
        "ontology": "LOINC",
        "source": "LOINC-Search-API",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="systolic blood pressure",
    )

    assert candidate.term == "Systolic blood pressure"


# ─────────────────────────────────────────────────────────────────────────────
# Test 23: normalize_many returns a list of NormalizedCandidate objects
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_many_returns_list(normalizer: CandidateNormalizer) -> None:
    raws = [
        {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS", "score": 0.95},
        {
            "code": "HP:0002110",
            "term": "Bronchiectasis",
            "ontology": "HPO",
            "source": "OLS",
            "score": 0.80,
        },
        {
            "code": "HP:0001650",
            "term": "Aortic regurgitation",
            "ontology": "HPO",
            "source": "OLS",
            "score": 0.72,
        },
    ]
    results = normalizer.normalize_many(
        raws,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, NormalizedCandidate) for r in results)
    assert [r.code for r in results] == ["HP:0012735", "HP:0002110", "HP:0001650"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 24: normalize_many raises clearly if one candidate is invalid
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_many_raises_on_invalid_candidate(normalizer: CandidateNormalizer) -> None:
    raws = [
        {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"},
        {"term": "Cough", "ontology": "HPO", "source": "OLS"},  # missing code
        {"code": "HP:0002110", "term": "Bronchiectasis", "ontology": "HPO", "source": "OLS"},
    ]
    with pytest.raises(CandidateNormalizationError, match="code"):
        normalizer.normalize_many(raws, retrieval_mode="public", matched_query="cough")


# ─────────────────────────────────────────────────────────────────────────────
# Test 25: no both mode is accepted or introduced
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_both_mode_accepted(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}

    # 'both' is not a valid retrieval mode string
    with pytest.raises(CandidateNormalizationError):
        normalizer.normalize(raw, retrieval_mode="both", matched_query="cough")

    # CandidateNormalizer produces no BOTH mode in its output
    public_candidate = normalizer.normalize(raw, retrieval_mode="public", matched_query="cough")
    assert public_candidate.retrieval_mode != "both"
    assert not hasattr(RetrievalMode, "BOTH")


# ─────────────────────────────────────────────────────────────────────────────
# Tests 26–28: existing pipeline tests still importable and not broken
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_existing_retrieval_router_imports_unaffected() -> None:
    from llm_ontology_mapper.models import RetrievalRoutePlan
    from llm_ontology_mapper.retrieval_router import RetrievalRouter

    assert RetrievalRouter is not None
    assert RetrievalRoutePlan is not None


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
def test_provenance_includes_default_ontology_when_provided(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {"code": "8480-6", "term": "Systolic BP", "source": "LOINC-Search-API"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="systolic blood pressure",
        default_ontology="LOINC",
    )

    assert candidate.provenance["default_ontology"] == "LOINC"


@pytest.mark.unit
def test_provenance_includes_default_source_when_provided(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
        default_source="CustomRetriever",
    )

    assert candidate.provenance["default_source"] == "CustomRetriever"


@pytest.mark.unit
def test_provenance_omits_default_ontology_when_not_provided(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert "default_ontology" not in candidate.provenance


@pytest.mark.unit
def test_explicit_normalized_score_used_when_present(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "HPO",
        "source": "OLS",
        "score": 2.5,  # out of range — would not be promoted
        "normalized_score": 0.82,  # explicit — should be used
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    assert candidate.raw_score == pytest.approx(2.5)
    assert candidate.normalized_score == pytest.approx(0.82)


@pytest.mark.unit
def test_ols_style_candidate_without_ontology_uses_default(
    normalizer: CandidateNormalizer,
) -> None:
    # OLS does not always include an 'ontology' key; callers pass default_ontology
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.95,
        "definition": "A cough.",
        "source": "OLS",
        "iri": "http://purl.obolibrary.org/obo/HP_0012735",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.PUBLIC,
        matched_query="cough",
        default_ontology="HPO",
    )

    assert candidate.code == "HP:0012735"
    assert candidate.ontology == "HPO"
    assert candidate.source == "OLS"


@pytest.mark.unit
def test_code_namespace_overrides_requested_ontology_fallback(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {
        "code": "HP:0002099",
        "term": "Asthma",
        "source": "OLS",
        "requested_ontology": "MONDO",
        "iri": "http://purl.obolibrary.org/obo/HP_0002099",
    }

    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.PUBLIC,
        matched_query="asthma",
        default_ontology="MONDO",
    )

    assert candidate.code == "HP:0002099"
    assert candidate.ontology == "HPO"


@pytest.mark.unit
def test_malformed_nested_curie_candidate_is_rejected(
    normalizer: CandidateNormalizer,
) -> None:
    raw = {
        "code": "MONDO:HP:0002099",
        "term": "Asthma",
        "ontology": "MONDO",
        "source": "OLS",
    }

    with pytest.raises(CandidateNormalizationError, match="identity"):
        normalizer.normalize(
            raw,
            retrieval_mode=RetrievalMode.PUBLIC,
            matched_query="asthma",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected_code", "expected_ontology"),
    [
        (
            {
                "code": "MONDO:0004979",
                "term": "asthma",
                "ontology": "MONDO",
                "source": "OLS",
            },
            "MONDO:0004979",
            "MONDO",
        ),
        (
            {
                "code": "HP:0002099",
                "term": "Asthma",
                "ontology": "HPO",
                "source": "OLS",
            },
            "HP:0002099",
            "HPO",
        ),
        (
            {
                "code": "LOINC:8480-6",
                "term": "Systolic blood pressure",
                "ontology": "LOINC",
                "source": "LOINC-Search-API",
            },
            "LOINC:8480-6",
            "LOINC",
        ),
    ],
)
def test_valid_mondo_hpo_and_loinc_candidates_remain_unchanged(
    normalizer: CandidateNormalizer,
    raw: dict[str, str],
    expected_code: str,
    expected_ontology: str,
) -> None:
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.PUBLIC,
        matched_query="query",
    )

    assert candidate.code == expected_code
    assert candidate.ontology == expected_ontology


@pytest.mark.unit
def test_sapbert_style_candidate(normalizer: CandidateNormalizer) -> None:
    # SapBERT returns: code, term, score, definition, source="SapBERT" — no ontology
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "score": 0.91,
        "definition": "",
        "source": "SapBERT",
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode=RetrievalMode.LOCAL,
        matched_query="cough",
        default_ontology="HPO",
    )

    assert candidate.retrieval_mode == RetrievalMode.LOCAL
    assert candidate.source == "SapBERT"
    assert candidate.ontology == "HPO"
    assert candidate.raw_score == pytest.approx(0.91)
    assert candidate.normalized_score == pytest.approx(0.91)


@pytest.mark.unit
def test_definition_blank_value_is_none(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "RXNORM:1234",
        "term": "Aspirin",
        "ontology": "RXNORM",
        "source": "RxNav",
        "definition": "",  # empty string — should produce None
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="aspirin",
    )

    assert candidate.definition is None


@pytest.mark.unit
def test_string_retrieval_mode_public(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",  # string, not enum
        matched_query="cough",
    )

    assert candidate.retrieval_mode == RetrievalMode.PUBLIC


@pytest.mark.unit
def test_string_retrieval_mode_local(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "SapBERT"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="local",  # string, not enum
        matched_query="cough",
    )

    assert candidate.retrieval_mode == RetrievalMode.LOCAL


@pytest.mark.unit
def test_candidate_is_frozen(normalizer: CandidateNormalizer) -> None:
    raw = {"code": "HP:0012735", "term": "Cough", "ontology": "HPO", "source": "OLS"}
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )

    with pytest.raises(ValidationError):
        candidate.code = "CHANGED"  # type: ignore[misc]


@pytest.mark.unit
def test_candidate_serialises_to_json(normalizer: CandidateNormalizer) -> None:
    raw = {
        "code": "HP:0012735",
        "term": "Cough",
        "ontology": "hpo",
        "source": "OLS",
        "score": 0.9,
    }
    candidate = normalizer.normalize(
        raw,
        retrieval_mode="public",
        matched_query="cough",
    )
    dumped = candidate.model_dump(mode="json")

    assert isinstance(dumped, dict)
    assert dumped["code"] == "HP:0012735"
    assert dumped["ontology"] == "HPO"  # uppercased
    assert dumped["retrieval_mode"] == "public"
