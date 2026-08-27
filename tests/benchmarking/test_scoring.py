"""
Unit tests for llm_ontology_mapper.benchmarking.scoring.

Pure logic -- no network, no pandas, no LLM/provider imports.

Run with:  pytest tests/benchmarking/test_scoring.py -v -m unit
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scoring import (
    RowScore,
    aggregate_scores,
    find_gold_rank,
    parse_gold_codes,
    score_row,
    top1_exact_agreement,
    top5_set_agreement,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Gold-code parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_single_gold_code() -> None:
    assert parse_gold_codes("HP:0000245") == ["HP:0000245"]


def test_parse_multiple_pipe_separated_gold_codes() -> None:
    assert parse_gold_codes("HP:0000245 | HP:0001742") == ["HP:0000245", "HP:0001742"]


def test_parse_gold_codes_trims_whitespace() -> None:
    assert parse_gold_codes("  HP:0000245  |  HP:0001742  ") == ["HP:0000245", "HP:0001742"]


def test_parse_gold_codes_blank_returns_empty_list() -> None:
    assert parse_gold_codes(None) == []
    assert parse_gold_codes("") == []
    assert parse_gold_codes("   ") == []


def test_parse_gold_codes_drops_empty_segments() -> None:
    assert parse_gold_codes("HP:0000245 | | HP:0001742") == ["HP:0000245", "HP:0001742"]


# ─────────────────────────────────────────────────────────────────────────────
# find_gold_rank: highest-ranked occurrence of any acceptable gold code
# ─────────────────────────────────────────────────────────────────────────────


def test_find_gold_rank_uses_highest_ranked_occurrence() -> None:
    # gold codes A | B; B appears at rank 2, A appears at rank 4 -> rank 2 wins.
    ranked = ["X:1", "B", "X:3", "A"]
    assert find_gold_rank(ranked, ["A", "B"]) == 2


def test_find_gold_rank_none_when_absent() -> None:
    ranked = ["X:1", "X:2", "X:3", "X:4", "X:5"]
    assert find_gold_rank(ranked, ["A", "B"]) is None


def test_find_gold_rank_ignores_none_slots() -> None:
    ranked = [None, None, "A", None, None]
    assert find_gold_rank(ranked, ["A"]) == 3


def test_find_gold_rank_only_considers_first_five() -> None:
    ranked = ["X:1", "X:2", "X:3", "X:4", "X:5", "A"]
    assert find_gold_rank(ranked, ["A"]) is None


# ─────────────────────────────────────────────────────────────────────────────
# Locked rank-based scoring
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rank,expected_tp,expected_fp,expected_fn",
    [
        (1, 1.0, 0.0, 0.0),
        (2, 0.5, 0.25, 0.25),
        (3, 0.25, 0.375, 0.375),
        (4, 0.125, 0.4375, 0.4375),
        (5, 0.0625, 0.46875, 0.46875),
    ],
)
def test_rank_scoring(rank: int, expected_tp: float, expected_fp: float, expected_fn: float) -> None:
    ranked = [None] * 5
    ranked[rank - 1] = "GOLD"
    result = score_row(is_mapped=True, ranked_codes=ranked, gold_codes=["GOLD"])
    assert result.gold_rank == rank
    assert result.tp == pytest.approx(expected_tp)
    assert result.fp == pytest.approx(expected_fp)
    assert result.fn == pytest.approx(expected_fn)
    assert result.tn == 0.0
    assert result.top1_correct == (rank == 1)
    assert result.top5_hit is True


def test_mapped_but_gold_not_in_top5() -> None:
    ranked = ["X:1", "X:2", "X:3", "X:4", "X:5"]
    result = score_row(is_mapped=True, ranked_codes=ranked, gold_codes=["GOLD"])
    assert result.gold_rank is None
    assert result.tp == 0.0
    assert result.fp == 0.5
    assert result.fn == 0.5
    assert result.tn == 0.0
    assert result.top1_correct is False
    assert result.top5_hit is False


def test_unmapped_with_gold_mapping_present() -> None:
    result = score_row(is_mapped=False, ranked_codes=[], gold_codes=["GOLD"])
    assert result.gold_rank is None
    assert result.tp == 0.0
    assert result.fp == 0.0
    assert result.fn == 1.0
    assert result.tn == 0.0
    assert result.top1_correct is False
    assert result.top5_hit is False


def test_unmapped_with_no_gold_is_true_negative() -> None:
    # Not expected in this dataset, but the scoring contract defines it.
    result = score_row(is_mapped=False, ranked_codes=[], gold_codes=[])
    assert result.tp == 0.0
    assert result.fp == 0.0
    assert result.fn == 0.0
    assert result.tn == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Weighted aggregate metrics: sum fractional counts before dividing
# ─────────────────────────────────────────────────────────────────────────────


def test_fractional_totals_summed_before_precision_recall_f1() -> None:
    # Row 1: rank 1 (tp=1, fp=0, fn=0). Row 2: rank 2 (tp=0.5, fp=0.25, fn=0.25).
    rows = [
        RowScore(gold_rank=1, top1_correct=True, top5_hit=True, tp=1.0, fp=0.0, fn=0.0, tn=0.0),
        RowScore(gold_rank=2, top1_correct=False, top5_hit=True, tp=0.5, fp=0.25, fn=0.25, tn=0.0),
    ]
    metrics = aggregate_scores(rows)
    assert metrics.tp_total == pytest.approx(1.5)
    assert metrics.fp_total == pytest.approx(0.25)
    assert metrics.fn_total == pytest.approx(0.25)
    expected_precision = 1.5 / (1.5 + 0.25)
    expected_recall = 1.5 / (1.5 + 0.25)
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert metrics.weighted_precision == pytest.approx(expected_precision)
    assert metrics.weighted_recall == pytest.approx(expected_recall)
    assert metrics.weighted_f1 == pytest.approx(expected_f1)


def test_aggregate_scores_handles_zero_denominators() -> None:
    rows = [RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=0.0, fp=0.0, fn=0.0, tn=1.0)]
    metrics = aggregate_scores(rows)
    assert metrics.weighted_precision == 0.0
    assert metrics.weighted_recall == 0.0
    assert metrics.weighted_f1 == 0.0


def test_aggregate_scores_empty_rows() -> None:
    metrics = aggregate_scores([])
    assert metrics.rows_evaluated == 0
    assert metrics.top1_accuracy == 0.0
    assert metrics.top5_hit_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Top-1 accuracy / Top-5 hit rate
# ─────────────────────────────────────────────────────────────────────────────


def test_top1_accuracy() -> None:
    rows = [
        RowScore(gold_rank=1, top1_correct=True, top5_hit=True, tp=1.0, fp=0.0, fn=0.0, tn=0.0),
        RowScore(gold_rank=2, top1_correct=False, top5_hit=True, tp=0.5, fp=0.25, fn=0.25, tn=0.0),
        RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=0.0, fp=0.5, fn=0.5, tn=0.0),
    ]
    metrics = aggregate_scores(rows)
    assert metrics.top1_correct_count == 1
    assert metrics.top1_accuracy == pytest.approx(1 / 3)


def test_top5_hit_rate() -> None:
    rows = [
        RowScore(gold_rank=1, top1_correct=True, top5_hit=True, tp=1.0, fp=0.0, fn=0.0, tn=0.0),
        RowScore(gold_rank=5, top1_correct=False, top5_hit=True, tp=0.0625, fp=0.46875, fn=0.46875, tn=0.0),
        RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=0.0, fp=0.5, fn=0.5, tn=0.0),
    ]
    metrics = aggregate_scores(rows)
    assert metrics.top5_hit_count == 2
    assert metrics.top5_hit_rate == pytest.approx(2 / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Two-run reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def test_top1_exact_agreement() -> None:
    pairs = [("HP:1", "HP:1"), ("HP:2", "HP:3"), (None, None)]
    assert top1_exact_agreement(pairs) == pytest.approx(2 / 3)


def test_top1_exact_agreement_empty() -> None:
    assert top1_exact_agreement([]) == 0.0


def test_top5_set_agreement_identical_sets_different_order() -> None:
    pairs = [
        (["HP:1", "HP:2", None, None, None], ["HP:2", "HP:1", None, None, None]),
    ]
    assert top5_set_agreement(pairs) == pytest.approx(1.0)


def test_top5_set_agreement_different_sets() -> None:
    pairs = [
        (["HP:1", "HP:2"], ["HP:1", "HP:3"]),
    ]
    assert top5_set_agreement(pairs) == pytest.approx(0.0)
