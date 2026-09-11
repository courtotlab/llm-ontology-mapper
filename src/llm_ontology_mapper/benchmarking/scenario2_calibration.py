"""
Scenario 2 (retrieval-mode ablation) confidence-calibration analysis: ROC AUC,
Brier score, Expected Calibration Error (10 fixed bins), and confidence
separation statistics (Wilcoxon rank-sum test + Cohen's d).

Calibration is scientifically evaluable here (unlike a gold-free HostSeq-style
dataset) because Top-1 correctness (Part 6/8) is directly derivable from the
workbook's gold labels. y = Top-1 exact-gold correctness; score = mapper
confidence. Unmapped/execution-error rows carry no ordinary prediction
confidence and MUST be excluded by the caller before these functions are
invoked (Part 11) -- this module does not know about status, only about
already-filtered (y, score) pairs.

Pure logic -- no network, no mapper. Re-computable from a saved
predictions.csv with zero mapper/LLM calls (--evaluate-existing).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

NOT_COMPUTABLE = "NOT_COMPUTABLE"
STATUS_OK = "OK"

WILCOXON_RANK_SUM_TEST_NAME = "scipy.stats.ranksums (Wilcoxon rank-sum test)"

# Ten fixed equal-width bins shared across every mode -- [0.0,0.1) .. [0.9,1.0].
# confidence=1.0 belongs in the final bin (Part 14).
ECE_BIN_EDGES: tuple[float, ...] = tuple(i / 10 for i in range(11))
ECE_NUM_BINS = 10


class CalibrationDataQualityError(ValueError):
    """Raised when a confidence score is outside [0, 1] -- never silently
    clipped (Part 13)."""


STATUS_MAPPED = "mapped"
_TRUE_STRINGS = {"true", "1", "yes"}


def build_calibration_pairs(rows: list[dict[str, str]]) -> tuple[list[int], list[float]]:
    """Build (y_true, y_score) from persisted predictions.csv-shaped row
    dicts (status, confidence, semantic_correctness), excluding every
    unmapped/execution-error row (Part 11/29) -- calibration correctness (y)
    comes from the gold-based semantic_correctness column ONLY, never from
    validation_status (hallucination is orthogonal to calibration)."""
    y_true: list[int] = []
    y_score: list[float] = []
    for row in rows:
        if row.get("status") != STATUS_MAPPED:
            continue
        confidence = row.get("confidence")
        if not confidence:
            continue
        semantic_correct = str(row.get("semantic_correctness", "")).strip().lower() in _TRUE_STRINGS
        y_true.append(1 if semantic_correct else 0)
        y_score.append(float(confidence))
    return y_true, y_score


def validate_confidences(scores: list[float]) -> None:
    for score in scores:
        if not (0.0 <= score <= 1.0):
            raise CalibrationDataQualityError(
                f"Mapper confidence={score!r} is outside the valid [0.0, 1.0] range; "
                "refusing to silently clip it. This indicates a data-quality problem "
                "upstream (a malformed prediction row), not a calibration bug."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Part 12 -- ROC AUC (Mann-Whitney U / rank-sum formulation; tie-aware)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RocAucResult:
    value: float | None
    status: str  # STATUS_OK | NOT_COMPUTABLE
    n_positive: int
    n_negative: int


def roc_auc(y_true: list[int], y_score: list[float]) -> RocAucResult:
    """AUC via the Mann-Whitney U / rank-sum identity:
        AUC = (sum(ranks of positive-class scores) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    Tie-aware via scipy.stats.rankdata (average-rank tie handling), so this is
    numerically identical to sklearn.metrics.roc_auc_score without adding a
    scikit-learn dependency. Returns NOT_COMPUTABLE (never a fabricated
    0.5/1.0) when only one outcome class is present (Part 12)."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    validate_confidences(y_score)

    n_positive = sum(1 for y in y_true if y == 1)
    n_negative = sum(1 for y in y_true if y == 0)
    if n_positive == 0 or n_negative == 0:
        return RocAucResult(value=None, status=NOT_COMPUTABLE, n_positive=n_positive, n_negative=n_negative)

    ranks = _scipy_stats.rankdata(y_score)
    rank_sum_positive = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    auc = (rank_sum_positive - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return RocAucResult(value=float(auc), status=STATUS_OK, n_positive=n_positive, n_negative=n_negative)


# ─────────────────────────────────────────────────────────────────────────────
# Part 13 -- Brier score
# ─────────────────────────────────────────────────────────────────────────────


def brier_score(y_true: list[int], y_score: list[float]) -> float | None:
    """mean((confidence - y)^2) over mapped, non-error, non-abstained rows.
    None only when there is no data (Part 13)."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if not y_true:
        return None
    validate_confidences(y_score)
    return statistics.fmean((score - y) ** 2 for y, score in zip(y_true, y_score, strict=True))


# ─────────────────────────────────────────────────────────────────────────────
# Part 14 -- Expected Calibration Error, 10 fixed equal-width bins
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EceBin:
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    calibration_gap: float | None  # |mean_confidence - empirical_accuracy|


@dataclass(frozen=True)
class EceResult:
    ece: float | None
    n: int
    bins: list[EceBin]


def _bin_index(score: float) -> int:
    """confidence=1.0 belongs in the final bin (index 9), never a bin 10."""
    idx = int(score * ECE_NUM_BINS)
    return min(idx, ECE_NUM_BINS - 1)


def expected_calibration_error(y_true: list[int], y_score: list[float]) -> EceResult:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    validate_confidences(y_score)

    n = len(y_true)
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(ECE_NUM_BINS)]
    for y, score in zip(y_true, y_score, strict=True):
        buckets[_bin_index(score)].append((y, score))

    bins: list[EceBin] = []
    ece_total = 0.0
    for i in range(ECE_NUM_BINS):
        items = buckets[i]
        lower, upper = ECE_BIN_EDGES[i], ECE_BIN_EDGES[i + 1]
        if not items:
            bins.append(
                EceBin(
                    bin_lower=lower,
                    bin_upper=upper,
                    count=0,
                    mean_confidence=None,
                    empirical_accuracy=None,
                    calibration_gap=None,
                )
            )
            continue
        count = len(items)
        mean_conf = statistics.fmean(score for _, score in items)
        emp_acc = statistics.fmean(y for y, _ in items)
        gap = abs(mean_conf - emp_acc)
        bins.append(
            EceBin(
                bin_lower=lower,
                bin_upper=upper,
                count=count,
                mean_confidence=mean_conf,
                empirical_accuracy=emp_acc,
                calibration_gap=gap,
            )
        )
        if n:
            ece_total += (count / n) * gap

    return EceResult(ece=ece_total if n else None, n=n, bins=bins)


# ─────────────────────────────────────────────────────────────────────────────
# Part 15 -- confidence separation: Wilcoxon rank-sum test + Cohen's d
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeparationStats:
    n_correct: int
    n_incorrect: int
    test_name: str
    statistic: float | None
    p_value: float | None
    mean_confidence_correct: float | None
    mean_confidence_incorrect: float | None
    sd_confidence_correct: float | None
    sd_confidence_incorrect: float | None
    cohens_d: float | None
    status: str  # STATUS_OK | NOT_COMPUTABLE
    note: str | None = None


def confidence_separation_stats(
    correct_scores: list[float], incorrect_scores: list[float]
) -> SeparationStats:
    validate_confidences(correct_scores)
    validate_confidences(incorrect_scores)
    n_correct, n_incorrect = len(correct_scores), len(incorrect_scores)

    if n_correct == 0 or n_incorrect == 0:
        return SeparationStats(
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            test_name=WILCOXON_RANK_SUM_TEST_NAME,
            statistic=None,
            p_value=None,
            mean_confidence_correct=statistics.fmean(correct_scores) if correct_scores else None,
            mean_confidence_incorrect=statistics.fmean(incorrect_scores) if incorrect_scores else None,
            sd_confidence_correct=None,
            sd_confidence_incorrect=None,
            cohens_d=None,
            status=NOT_COMPUTABLE,
            note="One of the two groups (correct/incorrect) is empty.",
        )

    mean_correct = statistics.fmean(correct_scores)
    mean_incorrect = statistics.fmean(incorrect_scores)
    sd_correct = statistics.stdev(correct_scores) if n_correct >= 2 else None
    sd_incorrect = statistics.stdev(incorrect_scores) if n_incorrect >= 2 else None

    result = _scipy_stats.ranksums(correct_scores, incorrect_scores)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    cohens_d: float | None
    note: str | None = None
    if n_correct < 2 or n_incorrect < 2:
        cohens_d = None
        note = "Cohen's d requires at least 2 samples in both groups to estimate variance."
    else:
        assert sd_correct is not None and sd_incorrect is not None  # guaranteed by n >= 2 above
        pooled_var = ((n_correct - 1) * sd_correct**2 + (n_incorrect - 1) * sd_incorrect**2) / (
            n_correct + n_incorrect - 2
        )
        pooled_sd = math.sqrt(pooled_var)
        if pooled_sd == 0.0:
            cohens_d = None
            note = "Pooled standard deviation is zero (no variance in either group); Cohen's d is undefined."
        else:
            cohens_d = (mean_correct - mean_incorrect) / pooled_sd

    return SeparationStats(
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        test_name=WILCOXON_RANK_SUM_TEST_NAME,
        statistic=statistic,
        p_value=p_value,
        mean_confidence_correct=mean_correct,
        mean_confidence_incorrect=mean_incorrect,
        sd_confidence_correct=sd_correct,
        sd_confidence_incorrect=sd_incorrect,
        cohens_d=cohens_d,
        status=STATUS_OK,
        note=note,
    )
