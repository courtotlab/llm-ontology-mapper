"""
Scenario 2 calibration tests (Part 30, items 20-29).

Pure logic -- no network, no mapper, no scipy live services (scipy.stats is
a local computation, not a network call).
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.scenario2_calibration import (
    ECE_BIN_EDGES,
    NOT_COMPUTABLE,
    STATUS_OK,
    WILCOXON_RANK_SUM_TEST_NAME,
    CalibrationDataQualityError,
    build_calibration_pairs,
    confidence_separation_stats,
    expected_calibration_error,
    roc_auc,
)
from llm_ontology_mapper.benchmarking.scenario2_calibration import (
    brier_score as compute_brier_score,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# 20. calibration correctness comes from gold (semantic_correctness), never
# from validation_status -- build_calibration_pairs never even reads it.
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_pairs_ignore_validation_status() -> None:
    rows = [
        {"status": "mapped", "confidence": "0.9", "semantic_correctness": "True", "validation_status": "INVALID"},
        {"status": "mapped", "confidence": "0.2", "semantic_correctness": "False", "validation_status": "VALID"},
    ]
    y_true, y_score = build_calibration_pairs(rows)
    # y is driven purely by semantic_correctness (gold match), regardless of
    # validation_status (hallucination) on the same row.
    assert y_true == [1, 0]
    assert y_score == [0.9, 0.2]


def test_calibration_pairs_exclude_unmapped_and_error_rows() -> None:
    rows = [
        {"status": "mapped", "confidence": "0.8", "semantic_correctness": "True"},
        {"status": "unmapped", "confidence": "", "semantic_correctness": "False"},
        {"status": "error", "confidence": "", "semantic_correctness": "False"},
    ]
    y_true, y_score = build_calibration_pairs(rows)
    assert y_true == [1]
    assert y_score == [0.8]


# ─────────────────────────────────────────────────────────────────────────────
# 21. AUC known fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_auc_perfect_separation() -> None:
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.3, 0.1]
    result = roc_auc(y_true, y_score)
    assert result.status == STATUS_OK
    assert result.value == pytest.approx(1.0)


def test_auc_worst_case_inverted() -> None:
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]
    result = roc_auc(y_true, y_score)
    assert result.value == pytest.approx(0.0)


def test_auc_random_chance_fixture() -> None:
    # Standard textbook fixture: two positives, two negatives, one tie pair
    # each side -> AUC = 0.5.
    y_true = [1, 0, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.5]
    result = roc_auc(y_true, y_score)
    assert result.value == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 22. Brier score known fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_brier_score_fixture() -> None:
    y_true = [1, 0]
    y_score = [1.0, 0.0]
    assert compute_brier_score(y_true, y_score) == pytest.approx(0.0)

    y_true2 = [1, 0]
    y_score2 = [0.0, 1.0]
    assert compute_brier_score(y_true2, y_score2) == pytest.approx(1.0)

    y_true3 = [1, 0]
    y_score3 = [0.5, 0.5]
    assert compute_brier_score(y_true3, y_score3) == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────────────────────
# 23. ECE known fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_ece_fixture_perfect_calibration() -> None:
    # 10 items at confidence=0.9, exactly 9/10 correct -> mean_confidence ==
    # empirical_accuracy == 0.9 in the one occupied bin -> ECE == 0.
    y_true = [1] * 9 + [0]
    y_score = [0.9] * 10
    result = expected_calibration_error(y_true, y_score)
    assert result.ece == pytest.approx(0.0)


def test_ece_fixture_miscalibrated() -> None:
    # confidence=0.9 but only 50% correct -> gap of 0.4 in that bin, all mass
    # in one bin -> ECE == 0.4.
    y_true = [1, 0]
    y_score = [0.9, 0.9]
    result = expected_calibration_error(y_true, y_score)
    assert result.ece == pytest.approx(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# 24. confidence=1.0 belongs in the final bin
# ─────────────────────────────────────────────────────────────────────────────


def test_confidence_1_0_in_final_bin() -> None:
    result = expected_calibration_error([1], [1.0])
    final_bin = result.bins[-1]
    assert final_bin.bin_lower == pytest.approx(0.9)
    assert final_bin.bin_upper == pytest.approx(1.0)
    assert final_bin.count == 1
    # No other bin received this point.
    assert sum(b.count for b in result.bins[:-1]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 25. fixed bins identical across modes
# ─────────────────────────────────────────────────────────────────────────────


def test_bin_edges_are_fixed_and_shared() -> None:
    assert tuple(i / 10 for i in range(11)) == ECE_BIN_EDGES
    result_a = expected_calibration_error([1, 0], [0.95, 0.05])
    result_b = expected_calibration_error([0, 1], [0.15, 0.75])
    edges_a = [(b.bin_lower, b.bin_upper) for b in result_a.bins]
    edges_b = [(b.bin_lower, b.bin_upper) for b in result_b.bins]
    assert edges_a == edges_b


# ─────────────────────────────────────────────────────────────────────────────
# 26. one-class AUC unavailable -- never fabricated as 0.5/1.0
# ─────────────────────────────────────────────────────────────────────────────


def test_auc_not_computable_single_class() -> None:
    result = roc_auc([1, 1, 1], [0.9, 0.8, 0.7])
    assert result.status == NOT_COMPUTABLE
    assert result.value is None


# ─────────────────────────────────────────────────────────────────────────────
# 27. rank-sum correct/incorrect groups
# ─────────────────────────────────────────────────────────────────────────────


def test_rank_sum_test_name_and_groups() -> None:
    correct = [0.9, 0.85, 0.95]
    incorrect = [0.2, 0.3, 0.25]
    stats = confidence_separation_stats(correct, incorrect)
    assert stats.test_name == WILCOXON_RANK_SUM_TEST_NAME
    assert "ranksums" in stats.test_name
    assert stats.n_correct == 3
    assert stats.n_incorrect == 3
    assert stats.statistic is not None
    assert stats.p_value is not None
    assert stats.status == STATUS_OK


# ─────────────────────────────────────────────────────────────────────────────
# 28. Cohen's d fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_cohens_d_fixture() -> None:
    # mean diff = 0.4, equal variance (sd=0.1 each) -> pooled sd = 0.1 -> d = 4.0
    correct = [0.8, 0.9, 1.0]
    incorrect = [0.4, 0.5, 0.6]
    stats = confidence_separation_stats(correct, incorrect)
    assert stats.mean_confidence_correct == pytest.approx(0.9)
    assert stats.mean_confidence_incorrect == pytest.approx(0.5)
    assert stats.cohens_d is not None
    assert stats.cohens_d > 0


def test_cohens_d_insufficient_samples() -> None:
    stats = confidence_separation_stats([0.9], [0.2])
    assert stats.cohens_d is None
    assert stats.note is not None


def test_cohens_d_zero_variance() -> None:
    stats = confidence_separation_stats([0.9, 0.9], [0.2, 0.2])
    assert stats.cohens_d is None
    assert "zero" in (stats.note or "").lower()


def test_separation_not_computable_when_group_empty() -> None:
    stats = confidence_separation_stats([], [0.5])
    assert stats.status == NOT_COMPUTABLE
    assert stats.statistic is None
    assert stats.p_value is None


# ─────────────────────────────────────────────────────────────────────────────
# 29. (see calibration-pairs tests above) + confidence range surfaced as error
# ─────────────────────────────────────────────────────────────────────────────


def test_out_of_range_confidence_raises_not_clipped() -> None:
    with pytest.raises(CalibrationDataQualityError):
        roc_auc([1, 0], [1.5, 0.2])
    with pytest.raises(CalibrationDataQualityError):
        compute_brier_score([1, 0], [-0.1, 0.2])
