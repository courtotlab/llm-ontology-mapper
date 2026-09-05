"""
Scenario 2 disabled-mode ranked-alternatives integration tests.

Proves that once DisabledMappingRunner populates MappingResult.alternatives
(see disabled_mapping.py / test_disabled_mapping.py), the EXISTING generic
Scenario 2 serialization (execute_row -> Scenario2RowResult.rank_codes) and
scoring (scenario2_metrics.score_prediction) pick up gold hits at ranks 2-5
with no disabled-specific code path -- exactly the same execute_row() and
score_prediction() used for public/local.

No network, no real LLM/OpenAI/SapBERT calls -- mapper.map_term is a
MagicMock returning a hand-built disabled-shaped MappingResult.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.model_registry import get_model_config
from llm_ontology_mapper.benchmarking.pricing import get_pricing
from llm_ontology_mapper.benchmarking.scenario2_metrics import PredictionRecord, score_prediction
from llm_ontology_mapper.benchmarking.scenario2_runner import Scenario2RunConfig, execute_row
from llm_ontology_mapper.models import (
    AlternativeMapping,
    GroundingSource,
    LogicType,
    MappingMetadata,
    MappingResult,
    RAGDebugInfo,
)

pytestmark = pytest.mark.unit


def _row(**overrides) -> BenchmarkRow:
    defaults = dict(
        input_row=1,
        source_variable="sys_bp",
        source_label="Systolic blood pressure",
        source_description=None,
        target_ontology="LOINC",
        gold_code_raw="LOINC:8459-0",
        gold_codes=["LOINC:8459-0"],
        gold_target_term="Systolic blood pressure--sitting",
    )
    defaults.update(overrides)
    return BenchmarkRow(**defaults)


def _disabled_metadata() -> MappingMetadata:
    return MappingMetadata(
        model="stub-model",
        provider="stub",
        rag_debug=RAGDebugInfo(
            query_sent="sys_bp",
            candidates_retrieved=[
                {
                    "retrieval_mode": "disabled",
                    "is_grounded": False,
                    "grounding_source": "none",
                    "policy": "disabled_llm_only",
                    "retrieval_skipped": True,
                    "candidate_count": 0,
                }
            ],
            top_k=0,
        ),
    )


def _disabled_result_with_alternatives(alt_codes: list[str]) -> MappingResult:
    alternatives = [
        AlternativeMapping(
            code=code,
            term=f"term for {code}",
            ontology="LOINC",
            confidence=round(0.5 - 0.05 * i, 2),
            source="llm",
        )
        for i, code in enumerate(alt_codes)
    ]
    return MappingResult(
        source_term="sys_bp",
        target_code="LOINC:8480-6",
        target_term="Systolic blood pressure",
        ontology="LOINC",
        confidence=0.6,
        logic_type=LogicType.LLM,
        alternatives=alternatives,
        metadata=_disabled_metadata(),
    )


def _execute(alt_codes: list[str], *, gold_codes: list[str]) -> tuple:
    mapper = MagicMock()
    mapper.map_term.return_value = _disabled_result_with_alternatives(alt_codes)
    row = _row(gold_codes=gold_codes, gold_code_raw=gold_codes[0])
    model_cfg = get_model_config("gpt-5.6-luna")
    run_config = Scenario2RunConfig(model_config=model_cfg, retrieval_mode="disabled", max_alternatives=4)
    pricing = get_pricing("gpt-5.6-luna")
    row_result = execute_row(mapper=mapper, row=row, run_config=run_config, pricing=pricing)
    record = PredictionRecord(
        row_id=row_result.input_row,
        status=row_result.mapped_status,
        gold_codes=tuple(row_result.gold_codes_normalized),
        ranks=row_result.rank_codes,
    )
    return row_result, score_prediction(record)


# ─────────────────────────────────────────────────────────────────────────────
# 20. rank serialization includes disabled rank 2-5
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_alternatives_serialize_into_rank_2_through_5() -> None:
    row_result, _ = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8478-0"],
    )
    assert row_result.rank_codes[0] == "LOINC:8480-6"  # selected
    assert row_result.rank_codes[1] == "LOINC:8459-0"  # alt 1 -> rank 2
    assert row_result.rank_codes[2] == "LOINC:8460-8"  # alt 2 -> rank 3
    assert row_result.rank_codes[4] == "LOINC:8478-0"  # alt 4 -> rank 5


def test_disabled_grounding_fields_unaffected_by_alternatives() -> None:
    row_result, _ = _execute(["LOINC:8459-0"], gold_codes=["LOINC:8459-0"])
    assert row_result.is_grounded is False
    assert row_result.grounding_source == GroundingSource.NONE.value
    assert row_result.retrieval_skipped is True
    assert row_result.selected_code_was_retrieved is False


# ─────────────────────────────────────────────────────────────────────────────
# 21-23. gold at rank 2 / 3 / 5
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_gold_at_rank_2() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8459-0"],
    )
    assert metrics.gold_rank == 2
    assert metrics.top1_hit is False


def test_disabled_gold_at_rank_3() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8460-8"],
    )
    assert metrics.gold_rank == 3


def test_disabled_gold_at_rank_5() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8478-0"],
    )
    assert metrics.gold_rank == 5


# ─────────────────────────────────────────────────────────────────────────────
# 24. Top-3/Top-5/MRR respond correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_gold_at_rank_2_hits_top3_and_top5_not_top1() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8459-0"],
    )
    assert metrics.top1_hit is False
    assert metrics.top3_hit is True
    assert metrics.top5_hit is True
    assert metrics.reciprocal_rank == pytest.approx(0.5)


def test_disabled_gold_at_rank_5_misses_top3_hits_top5() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8", "LOINC:8461-6", "LOINC:8478-0"],
        gold_codes=["LOINC:8478-0"],
    )
    assert metrics.top3_hit is False
    assert metrics.top5_hit is True
    assert metrics.reciprocal_rank == pytest.approx(0.2)


def test_disabled_gold_not_present_scores_zero() -> None:
    _, metrics = _execute(
        ["LOINC:8459-0", "LOINC:8460-8"],
        gold_codes=["LOINC:9999-9"],
    )
    assert metrics.gold_rank is None
    assert metrics.top1_hit is False
    assert metrics.top5_hit is False
    assert metrics.reciprocal_rank == 0.0
