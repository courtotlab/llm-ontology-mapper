"""
Scenario 2 grounding-extraction tests (Part 30, items 32-35).

Builds MappingResult.metadata.rag_debug.candidates_retrieved[0] in the exact
shape mapping_result_builder._build_metadata / disabled_mapping._build_metadata
actually produce, and checks extract_grounding_info() reads real evidence --
never inferring grounding from retrieval_mode or logic_type alone.
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scenario2_grounding import extract_grounding_info
from llm_ontology_mapper.models import LogicType, MappingMetadata, MappingResult, RAGDebugInfo

pytestmark = pytest.mark.unit


def _grounded_result(*, candidate_codes: list[str], selected_code: str, is_grounded: bool = True) -> MappingResult:
    pipeline_info = {
        "retrieval_mode": "public",
        "is_grounded": is_grounded,
        "grounding_source": "public_api",
        "policy": "production_grounded",
        "candidate_count": len(candidate_codes),
        "retrieval_trace": {"retrieval_skipped": False},
        "candidate_score_provenance": [
            {"code": code, "ontology": "HPO"} for code in candidate_codes
        ],
    }
    rag_debug = RAGDebugInfo(query_sent="q", candidates_retrieved=[pipeline_info], top_k=len(candidate_codes))
    return MappingResult(
        source_term="q",
        target_code=selected_code,
        target_term="t",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=rag_debug),
    )


def _disabled_result(selected_code: str) -> MappingResult:
    pipeline_info = {
        "retrieval_mode": "disabled",
        "is_grounded": False,
        "grounding_source": "none",
        "policy": "disabled_llm_only",
        "retrieval_skipped": True,
        "candidate_count": 0,
    }
    rag_debug = RAGDebugInfo(query_sent="q", candidates_retrieved=[pipeline_info], top_k=0)
    return MappingResult(
        source_term="q",
        target_code=selected_code,
        target_term="t",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.LLM,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=rag_debug),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 32. public selected retrieved candidate -> grounded
# ─────────────────────────────────────────────────────────────────────────────


def test_public_selected_code_in_candidates_is_grounded() -> None:
    result = _grounded_result(candidate_codes=["HP:0002110", "HP:0000001"], selected_code="HP:0002110")
    info = extract_grounding_info(result, is_mapped=True, mapped_code_normalized="HP:0002110")
    assert info.is_grounded is True
    assert info.grounding_source == "public_api"
    assert info.selected_code_was_retrieved is True
    assert info.retrieval_skipped is False


# ─────────────────────────────────────────────────────────────────────────────
# 33. local selected retrieved candidate -> grounded
# ─────────────────────────────────────────────────────────────────────────────


def test_local_selected_code_in_candidates_is_grounded() -> None:
    pipeline_info = {
        "retrieval_mode": "local",
        "is_grounded": True,
        "grounding_source": "local_sapbert",
        "retrieval_trace": {"retrieval_skipped": False},
        "candidate_count": 1,
        "candidate_score_provenance": [{"code": "MONDO:0000123", "ontology": "MONDO"}],
    }
    rag_debug = RAGDebugInfo(query_sent="q", candidates_retrieved=[pipeline_info], top_k=1)
    result = MappingResult(
        source_term="q",
        target_code="MONDO:0000123",
        target_term="t",
        ontology="MONDO",
        confidence=0.85,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=rag_debug),
    )
    info = extract_grounding_info(result, is_mapped=True, mapped_code_normalized="MONDO:0000123")
    assert info.is_grounded is True
    assert info.grounding_source == "local_sapbert"
    assert info.selected_code_was_retrieved is True


# ─────────────────────────────────────────────────────────────────────────────
# 34. absent selected candidate -> anomaly (is_grounded flag disagrees with
# candidate-provenance evidence -- extraction reports the EVIDENCE, not the
# flag, surfacing the anomaly rather than trusting is_grounded blindly).
# ─────────────────────────────────────────────────────────────────────────────


def test_selected_code_absent_from_candidates_is_flagged_false_even_if_is_grounded_true() -> None:
    result = _grounded_result(candidate_codes=["HP:0000001", "HP:0000002"], selected_code="HP:9999999", is_grounded=True)
    info = extract_grounding_info(result, is_mapped=True, mapped_code_normalized="HP:9999999")
    # The pipeline claimed is_grounded=True, but the selected code is not
    # actually present in the candidates supplied to the reranker -- this is
    # exactly the anomaly grounding_rate must be able to surface.
    assert info.is_grounded is True
    assert info.selected_code_was_retrieved is False


# ─────────────────────────────────────────────────────────────────────────────
# 35. disabled -> ungrounded
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_mode_is_ungrounded_and_retrieval_skipped() -> None:
    result = _disabled_result("HP:0002110")
    info = extract_grounding_info(result, is_mapped=True, mapped_code_normalized="HP:0002110")
    assert info.is_grounded is False
    assert info.grounding_source == "none"
    assert info.retrieval_skipped is True
    assert info.selected_code_was_retrieved is False


def test_unmapped_row_selected_code_was_retrieved_is_false() -> None:
    pipeline_info = {
        "retrieval_mode": "public",
        "is_grounded": False,
        "grounding_source": "public_api",
        "retrieval_trace": {"retrieval_skipped": False},
        "candidate_count": 1,
        "candidate_score_provenance": [{"code": "HP:0000001", "ontology": "HPO"}],
    }
    rag_debug = RAGDebugInfo(query_sent="q", candidates_retrieved=[pipeline_info], top_k=1)
    result = MappingResult(
        source_term="q",
        target_code="UNKNOWN:UNMAPPED",
        target_term="UNMAPPED",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=rag_debug),
    )
    info = extract_grounding_info(result, is_mapped=False, mapped_code_normalized=None)
    assert info.selected_code_was_retrieved is False
