"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_metrics.

Run with:  pytest tests/benchmarking/test_scenario1_metrics.py -v -m unit
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scenario1_metrics import (
    FN,
    FP_ERROR,
    STATUS_OK,
    TP_IDENTICAL,
    TP_RELATED,
    PredictionRecord,
    aggregate,
    aggregate_tp_taxonomy,
    classify_tp_taxonomy_row,
    execution_diagnostics,
    first_gold_rank,
    namespace_distribution,
    recall_at_gt,
    reciprocal_rank,
    score_prediction,
    top_k_hit,
)

pytestmark = pytest.mark.unit


def _ranks(*codes: str | None) -> tuple[str | None, ...]:
    padded = list(codes) + [None] * (5 - len(codes))
    return tuple(padded[:5])


def _record(**overrides) -> PredictionRecord:
    defaults = dict(query_id=1, query="q", gold_codes=("EFO:1",), status="mapped", ranks=_ranks("EFO:1"))
    defaults.update(overrides)
    return PredictionRecord(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 11/12/13. Top-1 single gold, Top-3, Top-5
# ─────────────────────────────────────────────────────────────────────────────


def test_top1_single_gold_hit() -> None:
    rank = first_gold_rank(_ranks("EFO:1"), ("EFO:1",))
    assert rank == 1
    assert top_k_hit(rank, 1) is True


def test_top3_success_gold_at_rank_3() -> None:
    rank = first_gold_rank(_ranks("X:1", "X:2", "EFO:1"), ("EFO:1",))
    assert rank == 3
    assert top_k_hit(rank, 1) is False
    assert top_k_hit(rank, 3) is True


def test_top5_success_gold_at_rank_5() -> None:
    rank = first_gold_rank(_ranks("X:1", "X:2", "X:3", "X:4", "EFO:1"), ("EFO:1",))
    assert rank == 5
    assert top_k_hit(rank, 3) is False
    assert top_k_hit(rank, 5) is True


def test_no_hit_rank_is_none() -> None:
    rank = first_gold_rank(_ranks("X:1", "X:2"), ("EFO:1",))
    assert rank is None
    assert top_k_hit(rank, 5) is False


# ─────────────────────────────────────────────────────────────────────────────
# 14/15. MRR + no-hit MRR=0
# ─────────────────────────────────────────────────────────────────────────────


def test_mrr_reciprocal_of_rank() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(4) == 0.25


def test_mrr_no_hit_is_zero() -> None:
    assert reciprocal_rank(None) == 0.0


def test_aggregate_mrr_over_multiple_rows() -> None:
    rows = [
        score_prediction(_record(query_id=1, ranks=_ranks("EFO:1"))),  # rank 1
        score_prediction(_record(query_id=2, ranks=_ranks("X", "EFO:1"))),  # rank 2
        score_prediction(_record(query_id=3, ranks=_ranks("X", "Y"))),  # no hit
    ]
    agg = aggregate(rows)
    assert agg.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)


# ─────────────────────────────────────────────────────────────────────────────
# 16. multiple-gold Top-k
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_gold_topk_hits_on_any_acceptable_code() -> None:
    rank = first_gold_rank(_ranks("X:1", "EFO:2"), ("EFO:1", "EFO:2"))
    assert rank == 2


# ─────────────────────────────────────────────────────────────────────────────
# 17. Recall@GT
# ─────────────────────────────────────────────────────────────────────────────


def test_recall_at_gt_all_recovered() -> None:
    r = recall_at_gt(_ranks("EFO:1", "EFO:2"), ("EFO:1", "EFO:2"))
    assert r == 1.0


def test_recall_at_gt_partial() -> None:
    r = recall_at_gt(_ranks("EFO:1", "X:9"), ("EFO:1", "EFO:2"))
    assert r == 0.5


def test_recall_at_gt_none_when_no_gold() -> None:
    assert recall_at_gt(_ranks("EFO:1"), ()) is None


def test_recall_at_gt_caps_window_at_gold_set_size() -> None:
    # gold set size 1 -> only rank-1 slot counted as the "top-n" window.
    r = recall_at_gt(_ranks("X:9", "EFO:1"), ("EFO:1",))
    assert r == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 18/19. unmapped/execution-error rows remain in the denominator
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_row_scores_zero_but_counted() -> None:
    rec = _record(status="unmapped", ranks=_ranks())
    rm = score_prediction(rec)
    assert rm.top1_hit is False
    assert rm.reciprocal_rank == 0.0

    agg = aggregate([rm])
    assert agg.n == 1
    assert agg.top1 == 0.0


def test_execution_error_row_scores_zero_but_counted() -> None:
    rec = _record(status="error", ranks=_ranks())
    rm = score_prediction(rec)
    agg = aggregate([rm, score_prediction(_record(query_id=2))])
    assert agg.n == 2
    assert agg.top1 == 0.5  # one hit, one error-as-zero


# ─────────────────────────────────────────────────────────────────────────────
# Namespace distribution (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────


def test_namespace_distribution_counts_rank1_ontology() -> None:
    records = [
        _record(query_id=1, rank_ontologies=("EFO", None, None, None, None)),
        _record(query_id=2, rank_ontologies=("UBERON", None, None, None, None)),
        _record(query_id=3, rank_ontologies=("EFO", None, None, None, None)),
    ]
    dist = namespace_distribution(records)
    assert dist == {"EFO": 2, "UBERON": 1}


def test_execution_diagnostics_counts() -> None:
    records = [
        _record(query_id=1, status="mapped"),
        _record(query_id=2, status="unmapped"),
        _record(query_id=3, status="error"),
        _record(query_id=4, status="mapped"),
    ]
    diag = execution_diagnostics(records)
    assert diag.total == 4
    assert diag.mapped_count == 2
    assert diag.unmapped_count == 1
    assert diag.error_count == 1
    assert diag.error_rate == 0.25


# ─────────────────────────────────────────────────────────────────────────────
# TP taxonomy (Part 15/16 -- reliability-audit follow-up): fully automatic,
# no manual review required or consulted. See classify_tp_taxonomy_row.
# ─────────────────────────────────────────────────────────────────────────────


def test_same_is_tp_identical() -> None:
    """1. Same -> TP-Identical."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="mapped",
        rank1_code="EFO:1",
        gold_codes=("EFO:1",),
        graph_relationship="Same",
    )
    assert row.category == TP_IDENTICAL


def test_more_specific_is_tp_related_automatically() -> None:
    """2. More Specific -> TP-Related, with no manual_review_decision input."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="mapped",
        rank1_code="MONDO:1",
        gold_codes=("EFO:1",),
        graph_relationship="More Specific",
    )
    assert row.category == TP_RELATED


def test_more_general_is_tp_related_automatically() -> None:
    """3. More General -> TP-Related, with no manual_review_decision input."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="mapped",
        rank1_code="MONDO:1",
        gold_codes=("EFO:1",),
        graph_relationship="More General",
    )
    assert row.category == TP_RELATED


def test_sibling_is_tp_related_automatically() -> None:
    """4. Sibling -> TP-Related, with no manual_review_decision input."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="mapped",
        rank1_code="MONDO:1",
        gold_codes=("EFO:1",),
        graph_relationship="Sibling",
    )
    assert row.category == TP_RELATED


def test_unrelated_is_fp_error() -> None:
    """5. Unrelated -> FP-Error."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="mapped",
        rank1_code="EFO:9",
        gold_codes=("EFO:1",),
        graph_relationship="Unrelated",
    )
    assert row.category == FP_ERROR


def test_unmapped_with_gold_is_fn() -> None:
    """6. unmapped -> FN."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="unmapped",
        rank1_code=None,
        gold_codes=("EFO:1",),
        graph_relationship=None,
    )
    assert row.category == FN


def test_execution_error_with_gold_is_fn_zero_credit() -> None:
    """Locked behavior: an execution error earns the SAME zero TP-taxonomy
    credit as a genuine unmapped row -- FN, not TP, not silently dropped."""
    row = classify_tp_taxonomy_row(
        query_id=1,
        status="error",
        rank1_code=None,
        gold_codes=("EFO:1",),
        graph_relationship=None,
    )
    assert row.category == FN


def test_graph_related_row_requires_no_manual_review_input() -> None:
    """7. classify_tp_taxonomy_row takes no manual_review_decision parameter
    at all -- graph-related rows are classified from graph_relationship alone."""
    import inspect

    params = inspect.signature(classify_tp_taxonomy_row).parameters
    assert "manual_review_decision" not in params


def test_taxonomy_precision_recall_f1_numeric_without_manual_review() -> None:
    """8. Precision/Recall/F1 are always numeric (STATUS_OK) -- no pending/
    REQUIRES_MANUAL_REVIEW gate -- once graph classification is available."""
    rows = [
        classify_tp_taxonomy_row(
            query_id=1, status="mapped", rank1_code="EFO:1", gold_codes=("EFO:1",),
            graph_relationship="Same",
        ),
        classify_tp_taxonomy_row(
            query_id=2, status="mapped", rank1_code="MONDO:1", gold_codes=("EFO:2",),
            graph_relationship="More Specific",
        ),
        classify_tp_taxonomy_row(
            query_id=3, status="mapped", rank1_code="MONDO:2", gold_codes=("EFO:3",),
            graph_relationship="More General",
        ),
        classify_tp_taxonomy_row(
            query_id=4, status="mapped", rank1_code="MONDO:3", gold_codes=("EFO:4",),
            graph_relationship="Sibling",
        ),
        classify_tp_taxonomy_row(
            query_id=5, status="mapped", rank1_code="EFO:9", gold_codes=("EFO:5",),
            graph_relationship="Unrelated",
        ),
        classify_tp_taxonomy_row(
            query_id=6, status="unmapped", rank1_code=None, gold_codes=("EFO:6",),
            graph_relationship=None,
        ),
    ]
    result = aggregate_tp_taxonomy(rows)
    assert result.status == STATUS_OK
    assert isinstance(result.precision, float)
    assert isinstance(result.recall, float)
    assert isinstance(result.f1, float)
    # TP-Identical=1, TP-Related=3, FP-Error=1, FN=1
    assert result.counts[TP_IDENTICAL] == 1
    assert result.counts[TP_RELATED] == 3
    assert result.counts[FP_ERROR] == 1
    assert result.counts[FN] == 1
    assert result.precision == pytest.approx(4 / 5)  # (1+3) / (1+3+1)
    assert result.recall == pytest.approx(4 / 5)  # (1+3) / (1+3+1)
    expected_f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
    assert result.f1 == pytest.approx(expected_f1)
