"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_runner.

Uses stub mappers / mocked HTTP -- no network, no real OpenAI or SapBERT
calls.

Run with:  pytest tests/benchmarking/test_scenario1_runner.py -v -m unit
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.benchmarking.scenario1_dataset import CanonicalQuery
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    STRICT_TARGET_ONTOLOGY,
    SapBertHealthError,
    check_sapbert_health,
    execute_query,
)
from llm_ontology_mapper.models import (
    AlternativeMapping,
    LogicType,
    MappingMetadata,
    MappingResult,
)

pytestmark = pytest.mark.unit


def _cq(**overrides) -> CanonicalQuery:
    defaults = dict(
        query_id=1,
        source_query="headache disorder",
        gold_codes=["EFO:0000001"],
        gold_labels=["Headache disorder"],
        gold_first_row_indices=[0],
        original_row_indices=[0],
    )
    defaults.update(overrides)
    return CanonicalQuery(**defaults)


def _alt(code: str, ontology: str, confidence: float = 0.5) -> AlternativeMapping:
    return AlternativeMapping(code=code, term=f"term for {code}", ontology=ontology, confidence=confidence)


# ─────────────────────────────────────────────────────────────────────────────
# 7. non-strict EFO is explicitly passed to mapper
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_query_passes_strict_target_ontology_false_explicitly() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="headache disorder",
        target_code="EFO:0000001",
        target_term="Headache disorder",
        ontology="EFO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )

    execute_query(mapper=mapper, cq=_cq(), pricing=None)

    assert STRICT_TARGET_ONTOLOGY is False
    _, kwargs = mapper.map_term.call_args
    assert kwargs["strict_target_ontology"] is False
    assert kwargs["source_term"] == "headache disorder"


# ─────────────────────────────────────────────────────────────────────────────
# 8. rank 1 + four alternatives preserved in returned order (never reranked)
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_query_preserves_returned_rank_order() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="q",
        target_code="UBERON:0000001",  # rank 1 is deliberately non-EFO
        target_term="rank1 term",
        ontology="UBERON",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[
            _alt("EFO:0000001", "EFO", confidence=0.8),
            _alt("MONDO:0000001", "MONDO", confidence=0.7),
            _alt("EFO:0000002", "EFO", confidence=0.6),
        ],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )

    row = execute_query(mapper=mapper, cq=_cq(gold_codes=["EFO:0000001"]), pricing=None)

    assert row.rank_codes == ("UBERON:0000001", "EFO:0000001", "MONDO:0000001", "EFO:0000002", None)
    # Never reordered so the higher-scoring EFO alternative isn't promoted to rank 1.
    assert row.rank_codes[0] == "UBERON:0000001"


def test_execute_query_pads_missing_alternatives_with_none_never_fabricated() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="q",
        target_code="EFO:0000001",
        target_term="t",
        ontology="EFO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    row = execute_query(mapper=mapper, cq=_cq(), pricing=None)
    assert row.rank_codes == ("EFO:0000001", None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# 9/10. non-EFO predictions preserved; no free exact-match credit
# ─────────────────────────────────────────────────────────────────────────────


def test_non_efo_top1_prediction_is_preserved_not_dropped_or_relabeled() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="q",
        target_code="MONDO:0000001",
        target_term="mondo term",
        ontology="MONDO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    row = execute_query(mapper=mapper, cq=_cq(gold_codes=["EFO:0000001"]), pricing=None)
    assert row.status == "mapped"
    assert row.mapped_code == "MONDO:0000001"
    assert row.mapped_ontology == "MONDO"


def test_non_efo_prediction_gets_no_exact_credit_unless_codes_match() -> None:
    from llm_ontology_mapper.benchmarking.scenario1_metrics import (
        PredictionRecord,
        score_prediction,
    )

    rec = PredictionRecord(
        query_id=1,
        query="q",
        gold_codes=("EFO:0000001",),
        status="mapped",
        ranks=("MONDO:0000001", None, None, None, None),
    )
    rm = score_prediction(rec)
    assert rm.top1_hit is False
    assert rm.gold_rank is None

    # But an exact code match (even outside EFO's own namespace, in principle)
    # does receive credit -- the rule is code equality, not ontology identity.
    rec2 = PredictionRecord(
        query_id=2,
        query="q",
        gold_codes=("EFO:0000001",),
        status="mapped",
        ranks=("EFO:0000001", None, None, None, None),
    )
    assert score_prediction(rec2).top1_hit is True


# ─────────────────────────────────────────────────────────────────────────────
# Unmapped / execution-error rows
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_query_unmapped_sentinel_recorded_as_unmapped() -> None:
    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term="q",
        target_code="UNKNOWN:UNMAPPED",
        target_term="UNMAPPED",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )
    row = execute_query(mapper=mapper, cq=_cq(), pricing=None)
    assert row.status == "unmapped"
    assert row.mapped_code is None
    assert row.rank_codes == (None, None, None, None, None)


def test_execute_query_execution_error_captured_not_scored_as_unmapped() -> None:
    mapper = MagicMock()
    mapper.map_term.side_effect = RuntimeError("boom")
    row = execute_query(mapper=mapper, cq=_cq(), pricing=None)
    assert row.status == "error"
    assert row.error_type == "RuntimeError"
    assert "boom" in (row.error_message or "")


# ─────────────────────────────────────────────────────────────────────────────
# 34/35. SapBERT health validation + EFO required
# ─────────────────────────────────────────────────────────────────────────────


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_check_sapbert_health_ok_with_efo_loaded() -> None:
    payload = {
        "status": "ok",
        "model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "loaded_indexes": ["EFO", "HPO"],
        "available_indexes": ["EFO", "HPO"],
        "lazy_load": True,
    }
    with patch("requests.get", return_value=_mock_response(payload)):
        health = check_sapbert_health("http://localhost:8765")
    assert health.status == "ok"
    assert "EFO" in health.loaded_indexes
    assert health.model == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def test_check_sapbert_health_rejects_non_ok_status() -> None:
    payload = {"status": "degraded", "loaded_indexes": ["EFO"], "available_indexes": ["EFO"]}
    with (
        patch("requests.get", return_value=_mock_response(payload)),
        pytest.raises(SapBertHealthError, match="status != 'ok'"),
    ):
        check_sapbert_health("http://localhost:8765")


def test_check_sapbert_health_requires_efo_present() -> None:
    payload = {"status": "ok", "loaded_indexes": ["HPO"], "available_indexes": ["HPO"]}
    with (
        patch("requests.get", return_value=_mock_response(payload)),
        pytest.raises(SapBertHealthError, match="EFO"),
    ):
        check_sapbert_health("http://localhost:8765")


def test_check_sapbert_health_efo_in_available_but_not_loaded_is_ok() -> None:
    payload = {"status": "ok", "loaded_indexes": [], "available_indexes": ["EFO"]}
    with patch("requests.get", return_value=_mock_response(payload)):
        health = check_sapbert_health("http://localhost:8765")
    assert "EFO" in health.available_indexes
