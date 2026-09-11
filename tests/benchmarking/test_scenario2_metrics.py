"""
Scenario 2 mapping-performance metric tests (Part 30, items 12-19).

Pure logic over scenario2_metrics.PredictionRecord/score_prediction/aggregate
-- no network, no mapper.
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scenario2_metrics import (
    AbstentionStats,
    PredictionRecord,
    abstention_stats,
    aggregate,
    score_prediction,
)

pytestmark = pytest.mark.unit


def _record(*, row_id=1, status="mapped", gold_codes=(), ranks=None) -> PredictionRecord:
    ranks = ranks if ranks is not None else (None, None, None, None, None)
    return PredictionRecord(row_id=row_id, status=status, gold_codes=gold_codes, ranks=ranks)


# ─────────────────────────────────────────────────────────────────────────────
# 12. rank1 exact gold -> correct
# ─────────────────────────────────────────────────────────────────────────────


def test_rank1_exact_gold_is_correct() -> None:
    rec = _record(gold_codes=("HP:0002110",), ranks=("HP:0002110", None, None, None, None))
    metrics = score_prediction(rec)
    assert metrics.semantic_correctness is True
    assert metrics.top1_hit is True
    assert metrics.gold_rank == 1


# ─────────────────────────────────────────────────────────────────────────────
# 13. either multi-gold target -> correct
# ─────────────────────────────────────────────────────────────────────────────


def test_either_multi_gold_target_counts_as_correct() -> None:
    rec_a = _record(gold_codes=("MONDO:0000001", "MONDO:0000002"), ranks=("MONDO:0000001", None, None, None, None))
    rec_b = _record(gold_codes=("MONDO:0000001", "MONDO:0000002"), ranks=("MONDO:0000002", None, None, None, None))
    assert score_prediction(rec_a).semantic_correctness is True
    assert score_prediction(rec_b).semantic_correctness is True


# ─────────────────────────────────────────────────────────────────────────────
# 14. valid but wrong ontology code -> incorrect
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_but_wrong_code_is_incorrect() -> None:
    rec = _record(gold_codes=("HP:0002110",), ranks=("HP:9999999", None, None, None, None))
    metrics = score_prediction(rec)
    assert metrics.semantic_correctness is False
    assert metrics.gold_rank is None


# ─────────────────────────────────────────────────────────────────────────────
# 15. unmapped -> zero Top-k credit
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_row_gets_zero_topk_credit() -> None:
    rec = _record(status="unmapped", gold_codes=("HP:0002110",), ranks=(None, None, None, None, None))
    metrics = score_prediction(rec)
    assert metrics.top1_hit is False
    assert metrics.top3_hit is False
    assert metrics.top5_hit is False
    assert metrics.reciprocal_rank == 0.0
    assert metrics.semantic_correctness is False


# ─────────────────────────────────────────────────────────────────────────────
# 16. error -> zero Top-k credit, still counted in N
# ─────────────────────────────────────────────────────────────────────────────


def test_error_row_gets_zero_topk_credit_but_stays_in_n() -> None:
    rec = _record(status="error", gold_codes=("HP:0002110",), ranks=(None, None, None, None, None))
    metrics = score_prediction(rec)
    assert metrics.top1_hit is False
    assert metrics.semantic_correctness is False

    agg = aggregate([metrics])
    assert agg.n == 1
    assert agg.top1 == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 17. Top-1/3/5 fixtures
# ─────────────────────────────────────────────────────────────────────────────


def test_top135_fixture_across_ranks() -> None:
    gold = ("HP:0002110",)
    rank1 = score_prediction(_record(row_id=1, gold_codes=gold, ranks=("HP:0002110", None, None, None, None)))
    rank3 = score_prediction(
        _record(row_id=2, gold_codes=gold, ranks=("HP:0000001", "HP:0000002", "HP:0002110", None, None))
    )
    rank5 = score_prediction(
        _record(
            row_id=3,
            gold_codes=gold,
            ranks=("HP:0000001", "HP:0000002", "HP:0000003", "HP:0000004", "HP:0002110"),
        )
    )
    absent = score_prediction(
        _record(row_id=4, gold_codes=gold, ranks=("HP:0000001", "HP:0000002", "HP:0000003", "HP:0000004", "HP:0000005"))
    )

    assert (rank1.top1_hit, rank1.top3_hit, rank1.top5_hit) == (True, True, True)
    assert (rank3.top1_hit, rank3.top3_hit, rank3.top5_hit) == (False, True, True)
    assert (rank5.top1_hit, rank5.top3_hit, rank5.top5_hit) == (False, False, True)
    assert (absent.top1_hit, absent.top3_hit, absent.top5_hit) == (False, False, False)

    agg = aggregate([rank1, rank3, rank5, absent])
    assert agg.top1 == pytest.approx(1 / 4)
    assert agg.top3 == pytest.approx(2 / 4)
    assert agg.top5 == pytest.approx(3 / 4)


# ─────────────────────────────────────────────────────────────────────────────
# 18. MRR fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_mrr_fixture() -> None:
    gold = ("HP:0002110",)
    rank1 = score_prediction(_record(row_id=1, gold_codes=gold, ranks=("HP:0002110", None, None, None, None)))
    rank2 = score_prediction(_record(row_id=2, gold_codes=gold, ranks=("HP:0000001", "HP:0002110", None, None, None)))
    rank4 = score_prediction(
        _record(row_id=3, gold_codes=gold, ranks=("HP:0000001", "HP:0000002", "HP:0000003", "HP:0002110", None))
    )
    absent = score_prediction(
        _record(row_id=4, gold_codes=gold, ranks=("HP:0000001", "HP:0000002", "HP:0000003", "HP:0000004", "HP:0000005"))
    )

    assert rank1.reciprocal_rank == pytest.approx(1.0)
    assert rank2.reciprocal_rank == pytest.approx(0.5)
    assert rank4.reciprocal_rank == pytest.approx(0.25)
    assert absent.reciprocal_rank == 0.0

    agg = aggregate([rank1, rank2, rank4, absent])
    assert agg.mrr == pytest.approx((1.0 + 0.5 + 0.25 + 0.0) / 4)


# ─────────────────────────────────────────────────────────────────────────────
# 19. Recall@GT fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_recall_at_gt_fixture_single_gold() -> None:
    rec = score_prediction(
        _record(row_id=1, gold_codes=("HP:0002110",), ranks=("HP:0002110", None, None, None, None))
    )
    assert rec.recall_at_gt == pytest.approx(1.0)


def test_recall_at_gt_fixture_double_gold_one_recovered() -> None:
    # n = 2 gold codes -> top-2 predictions checked; only one of the two golds appears.
    rec = score_prediction(
        _record(
            row_id=1,
            gold_codes=("MONDO:0000001", "MONDO:0000002"),
            ranks=("MONDO:0000001", "MONDO:0000099", None, None, None),
        )
    )
    assert rec.recall_at_gt == pytest.approx(0.5)


def test_recall_at_gt_fixture_double_gold_both_recovered() -> None:
    rec = score_prediction(
        _record(
            row_id=1,
            gold_codes=("MONDO:0000001", "MONDO:0000002"),
            ranks=("MONDO:0000001", "MONDO:0000002", None, None, None),
        )
    )
    assert rec.recall_at_gt == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 30/31. abstention counted for unmapped, not for errors
# ─────────────────────────────────────────────────────────────────────────────


def test_abstention_counts_unmapped_and_unmapped_sentinel_not_errors() -> None:
    records = [
        _record(row_id=1, status="unmapped", gold_codes=("HP:0002110",)),
        _record(row_id=2, status="mapped", gold_codes=("HP:0002110",), ranks=("HP:0002110", None, None, None, None)),
        _record(row_id=3, status="error", gold_codes=("HP:0002110",)),
        _record(row_id=4, status="mapped", gold_codes=("HP:0002110",), ranks=("HP:0002110", None, None, None, None)),
    ]
    mapped_codes = [None, "HP:0002110", None, "UNKNOWN:UNMAPPED"]
    stats: AbstentionStats = abstention_stats(records, mapped_codes)
    # row 1 (unmapped) and row 4 (mapped-but-sentinel) abstain; row 3 (error) does not.
    assert stats.total == 4
    assert stats.abstention_count == 2
    assert stats.abstention_rate == pytest.approx(0.5)
