"""
Pure rank-based scoring logic for the model benchmark.

No network, no pandas, no LLM/provider imports — this module only operates on
plain strings/lists so it is trivially unit-testable and independent of how
codes were retrieved or normalized.

Locked scoring contract (rank = position of the first acceptable gold code
among the 5 candidate ranks: 1=selected result, 2-5=alternatives[0..3]):

    rank 1                          -> TP=1,     FP=0,       FN=0
    rank 2                          -> TP=0.5,   FP=0.25,    FN=0.25
    rank 3                          -> TP=0.25,  FP=0.375,   FN=0.375
    rank 4                          -> TP=0.125, FP=0.4375,  FN=0.4375
    rank 5                          -> TP=0.0625,FP=0.46875, FN=0.46875
    mapped, gold absent from ranks  -> TP=0,     FP=0.5,     FN=0.5
    unmapped, gold exists           -> TP=0,     FP=0,       FN=1

TN is always 0 for this dataset (no intended gold-negative examples); the
`tn` field is kept for output-schema completeness only, per the benchmark
spec, and is not aggregated into any TN-dependent metric.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_RANK = 5

# rank -> (tp, fp, fn)
_RANK_SCORES: dict[int, tuple[float, float, float]] = {
    1: (1.0, 0.0, 0.0),
    2: (0.5, 0.25, 0.25),
    3: (0.25, 0.375, 0.375),
    4: (0.125, 0.4375, 0.4375),
    5: (0.0625, 0.46875, 0.46875),
}

# (tp, fp, fn) when the mapper returned a mapping but no gold code appears in ranks 1-5
_MAPPED_GOLD_MISSING = (0.0, 0.5, 0.5)

# (tp, fp, fn) when the mapper returned unmapped while a gold mapping exists
_UNMAPPED_WITH_GOLD = (0.0, 0.0, 1.0)


def parse_gold_codes(raw: str | None) -> list[str]:
    """Split a '|'-separated gold-code cell into trimmed, non-empty codes.

    A single code with no '|' returns a one-item list. Blank/None input
    returns an empty list.
    """
    if raw is None:
        return []
    text = str(raw)
    return [part.strip() for part in text.split("|") if part.strip()]


def find_gold_rank(
    ranked_codes: Sequence[str | None],
    gold_codes: Iterable[str],
) -> int | None:
    """Return the 1-based rank of the first (highest-ranked) occurrence of
    any acceptable gold code among ranked_codes[:5], or None if absent.
    """
    gold_set = {code for code in gold_codes if code}
    if not gold_set:
        return None
    for rank, code in enumerate(ranked_codes[:MAX_RANK], start=1):
        if code and code in gold_set:
            return rank
    return None


@dataclass(frozen=True)
class RowScore:
    """Scoring outcome for a single benchmark row."""

    gold_rank: int | None
    top1_correct: bool
    top5_hit: bool
    tp: float
    fp: float
    fn: float
    tn: float


def score_row(
    *,
    is_mapped: bool,
    ranked_codes: Sequence[str | None] = (),
    gold_codes: Iterable[str] = (),
) -> RowScore:
    """
    Score one row under the locked rank-based scheme.

    Args:
        is_mapped: True when the mapper returned a real mapping (not the
            UNMAPPED sentinel) and the call completed without an execution
            error. Execution errors must never be routed through this
            function as is_mapped=False -- they are a separate outcome.
        ranked_codes: up to 5 candidate codes, index 0 = rank 1 (selected
            result), indices 1-4 = alternatives 1-4. Missing alternative
            slots should be None, never fabricated.
        gold_codes: acceptable gold codes for this row (already parsed via
            parse_gold_codes and, by convention, normalized by the caller).
    """
    gold_list = list(gold_codes)

    if not is_mapped:
        if gold_list:
            tp, fp, fn = _UNMAPPED_WITH_GOLD
            return RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=tp, fp=fp, fn=fn, tn=0.0)
        # Not expected in this dataset (no intended gold-negative rows), but
        # defined for completeness: an unmapped row with no gold at all is a
        # true negative.
        return RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=0.0, fp=0.0, fn=0.0, tn=1.0)

    rank = find_gold_rank(ranked_codes, gold_list)
    if rank is None:
        tp, fp, fn = _MAPPED_GOLD_MISSING
        return RowScore(gold_rank=None, top1_correct=False, top5_hit=False, tp=tp, fp=fp, fn=fn, tn=0.0)

    tp, fp, fn = _RANK_SCORES[rank]
    return RowScore(
        gold_rank=rank,
        top1_correct=rank == 1,
        top5_hit=True,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=0.0,
    )


@dataclass(frozen=True)
class RunMetrics:
    """Weighted aggregate + conventional accuracy metrics for one run."""

    rows_evaluated: int
    tp_total: float
    fp_total: float
    fn_total: float
    tn_total: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    top1_correct_count: int
    top1_accuracy: float
    top5_hit_count: int
    top5_hit_rate: float


def aggregate_scores(row_scores: Sequence[RowScore]) -> RunMetrics:
    """
    Sum fractional TP/FP/FN/TN across rows first, then derive weighted
    precision/recall/F1 from the summed totals (never average per-row
    ratios) and conventional top1/top5 rates over the evaluated rows.
    """
    tp_total = sum(r.tp for r in row_scores)
    fp_total = sum(r.fp for r in row_scores)
    fn_total = sum(r.fn for r in row_scores)
    tn_total = sum(r.tn for r in row_scores)

    weighted_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    weighted_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    weighted_f1 = (
        2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        if (weighted_precision + weighted_recall) > 0
        else 0.0
    )

    rows_evaluated = len(row_scores)
    top1_correct_count = sum(1 for r in row_scores if r.top1_correct)
    top5_hit_count = sum(1 for r in row_scores if r.top5_hit)
    top1_accuracy = top1_correct_count / rows_evaluated if rows_evaluated else 0.0
    top5_hit_rate = top5_hit_count / rows_evaluated if rows_evaluated else 0.0

    return RunMetrics(
        rows_evaluated=rows_evaluated,
        tp_total=tp_total,
        fp_total=fp_total,
        fn_total=fn_total,
        tn_total=tn_total,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        top1_correct_count=top1_correct_count,
        top1_accuracy=top1_accuracy,
        top5_hit_count=top5_hit_count,
        top5_hit_rate=top5_hit_rate,
    )


def top1_exact_agreement(pairs: Sequence[tuple[str | None, str | None]]) -> float:
    """Fraction of rows where run-1 rank-1 code == run-2 rank-1 code.

    Callers must pass already ontology-normalized codes (or None for
    unmapped) so formatting differences don't register as disagreement.
    """
    if not pairs:
        return 0.0
    agree = sum(1 for run1, run2 in pairs if run1 == run2)
    return agree / len(pairs)


def top5_set_agreement(
    pairs: Sequence[tuple[Sequence[str | None], Sequence[str | None]]],
) -> float:
    """Fraction of rows where the run-1 and run-2 rank1-5 code sets are identical.

    None slots (missing alternatives) are dropped before comparison; callers
    must pass already ontology-normalized codes.
    """
    if not pairs:
        return 0.0
    agree = 0
    for run1_codes, run2_codes in pairs:
        set1 = {c for c in run1_codes if c}
        set2 = {c for c in run2_codes if c}
        if set1 == set2:
            agree += 1
    return agree / len(pairs)
