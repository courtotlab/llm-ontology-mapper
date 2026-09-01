"""
Scenario 2 (retrieval-mode ablation) mapping-performance metrics: Top-1/3/5,
MRR, Recall@GT, semantic correctness, and abstention/execution diagnostics.

Rank-based scoring (first_gold_rank / top_k_hit / reciprocal_rank /
recall_at_gt) is imported verbatim from scenario1_metrics -- it is generic
over any 5-slot ranked-code prediction and gold-code set (nothing EFO- or
graph-distance-specific), so Scenario 2 reuses the SAME validated
implementation rather than re-deriving it (Part 7/8/9).

Pure logic over plain records -- no network, no mapper, no pandas dependency
on a *live* run, so every metric here is re-computable from a saved
predictions.csv with zero mapper/LLM calls (--evaluate-existing).
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_ontology_mapper.benchmarking.scenario1_metrics import (
    MAX_RANK,
    first_gold_rank,
    recall_at_gt,
    reciprocal_rank,
    top_k_hit,
)

STATUS_MAPPED = "mapped"
STATUS_UNMAPPED = "unmapped"
STATUS_ERROR = "error"

UNMAPPED_SENTINEL = "UNKNOWN:UNMAPPED"

__all__ = [
    "MAX_RANK",
    "STATUS_ERROR",
    "STATUS_MAPPED",
    "STATUS_UNMAPPED",
    "UNMAPPED_SENTINEL",
    "AbstentionStats",
    "AggregateMetrics",
    "ExecutionDiagnostics",
    "PredictionRecord",
    "RowMetrics",
    "abstention_stats",
    "aggregate",
    "execution_diagnostics",
    "score_prediction",
]


# ─────────────────────────────────────────────────────────────────────────────
# Record shape shared by a live run and a reloaded predictions.csv
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionRecord:
    row_id: int
    status: str  # "mapped" | "unmapped" | "error"
    gold_codes: tuple[str, ...]
    ranks: tuple[str | None, ...]  # length 5: rank1..rank5 codes, in returned order

    def __post_init__(self) -> None:
        if len(self.ranks) != MAX_RANK:
            raise ValueError(f"ranks must have exactly {MAX_RANK} slots, got {len(self.ranks)}")

    @property
    def rank1_code(self) -> str | None:
        return self.ranks[0]


# ─────────────────────────────────────────────────────────────────────────────
# Part 6/8/9 -- per-row Top-k / MRR / Recall@GT / semantic correctness
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RowMetrics:
    row_id: int
    gold_rank: int | None
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    reciprocal_rank: float
    recall_at_gt: float | None
    semantic_correctness: bool  # locked identical to top1_hit -- Part 6


def score_prediction(record: PredictionRecord) -> RowMetrics:
    """Score one row. Execution errors and unmapped rows fall straight through
    (ranks are all-None or partially None) and score 0 on every rank-based
    metric -- they are never dropped from the N=218 denominator (Part 7/22)."""
    gold_rank = first_gold_rank(record.ranks, record.gold_codes)
    return RowMetrics(
        row_id=record.row_id,
        gold_rank=gold_rank,
        top1_hit=top_k_hit(gold_rank, 1),
        top3_hit=top_k_hit(gold_rank, 3),
        top5_hit=top_k_hit(gold_rank, 5),
        reciprocal_rank=reciprocal_rank(gold_rank),
        recall_at_gt=recall_at_gt(record.ranks, record.gold_codes),
        semantic_correctness=top_k_hit(gold_rank, 1),
    )


@dataclass(frozen=True)
class AggregateMetrics:
    n: int
    top1: float
    top3: float
    top5: float
    mrr: float
    recall_at_gt: float
    recall_at_gt_n: int


def aggregate(row_metrics: list[RowMetrics]) -> AggregateMetrics:
    n = len(row_metrics)
    if n == 0:
        return AggregateMetrics(n=0, top1=0.0, top3=0.0, top5=0.0, mrr=0.0, recall_at_gt=0.0, recall_at_gt_n=0)
    top1 = sum(1 for r in row_metrics if r.top1_hit) / n
    top3 = sum(1 for r in row_metrics if r.top3_hit) / n
    top5 = sum(1 for r in row_metrics if r.top5_hit) / n
    mrr = sum(r.reciprocal_rank for r in row_metrics) / n
    recall_values = [r.recall_at_gt for r in row_metrics if r.recall_at_gt is not None]
    recall_at_gt_mean = sum(recall_values) / len(recall_values) if recall_values else 0.0
    return AggregateMetrics(
        n=n,
        top1=top1,
        top3=top3,
        top5=top5,
        mrr=mrr,
        recall_at_gt=recall_at_gt_mean,
        recall_at_gt_n=len(recall_values),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 10 -- abstention (unmapped OR mapped-but-UNKNOWN:UNMAPPED); errors excluded
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AbstentionStats:
    total: int
    abstention_count: int

    @property
    def abstention_rate(self) -> float:
        return self.abstention_count / self.total if self.total else 0.0


def is_abstention(*, status: str, mapped_code: str | None) -> bool:
    """status="unmapped" OR mapped code == UNKNOWN:UNMAPPED (Part 10).
    Execution errors are NEVER abstentions -- they are a distinct outcome."""
    if status == STATUS_ERROR:
        return False
    if status == STATUS_UNMAPPED:
        return True
    normalized_code = (mapped_code or "").strip().upper()
    return normalized_code == UNMAPPED_SENTINEL


def abstention_stats(records: list[PredictionRecord], mapped_codes: list[str | None]) -> AbstentionStats:
    """mapped_codes must be the same length/order as records (rank1_code is
    already covered by record.rank1_code, but that is the *normalized* code;
    callers pass the raw mapped_code here for the exact sentinel check)."""
    if len(records) != len(mapped_codes):
        raise ValueError("records and mapped_codes must have the same length")
    total = len(records)
    count = sum(
        1
        for record, mapped_code in zip(records, mapped_codes, strict=True)
        if is_abstention(status=record.status, mapped_code=mapped_code)
    )
    return AbstentionStats(total=total, abstention_count=count)


# ─────────────────────────────────────────────────────────────────────────────
# Part 22 -- execution-error / mapped / unmapped diagnostics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionDiagnostics:
    total: int
    mapped_count: int
    unmapped_count: int
    error_count: int

    @property
    def mapped_rate(self) -> float:
        return self.mapped_count / self.total if self.total else 0.0

    @property
    def unmapped_rate(self) -> float:
        return self.unmapped_count / self.total if self.total else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total if self.total else 0.0


def execution_diagnostics(records: list[PredictionRecord]) -> ExecutionDiagnostics:
    total = len(records)
    mapped = sum(1 for r in records if r.status == STATUS_MAPPED)
    unmapped = sum(1 for r in records if r.status == STATUS_UNMAPPED)
    error = sum(1 for r in records if r.status == STATUS_ERROR)
    return ExecutionDiagnostics(total=total, mapped_count=mapped, unmapped_count=unmapped, error_count=error)
