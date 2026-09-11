"""
Scenario 1 (OLS-EFO) metric computation: Top-k, MRR, Recall@GT, graph-distance
summary, namespace distribution, and the TP taxonomy (Parts 10-16, 21).

Pure logic over plain records -- no network, no mapper, no pandas dependency
on a *live* run. Operates identically whether the records were just produced
by scenario1_runner or reloaded from a saved predictions.csv, which is what
lets `--evaluate-existing` recompute every metric without repeating a single
LLM call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

MAX_RANK = 5
UNMAPPED_SENTINEL = "UNKNOWN:UNMAPPED"

STATUS_MAPPED = "mapped"
STATUS_UNMAPPED = "unmapped"
STATUS_ERROR = "error"

EVAL_UNIT_UNIQUE_QUERY = "unique_query"
EVAL_UNIT_MAPPING_PAIR = "mapping_pair"

STATUS_OK = "OK"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Record shape shared by a live run and a reloaded predictions.csv
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionRecord:
    query_id: int
    query: str
    gold_codes: tuple[str, ...]
    status: str  # "mapped" | "unmapped" | "error"
    ranks: tuple[str | None, ...]  # length 5: rank1..rank5 codes, in returned order
    rank_ontologies: tuple[str | None, ...] = field(default=())  # length 5

    def __post_init__(self) -> None:
        if len(self.ranks) != MAX_RANK:
            raise ValueError(f"ranks must have exactly {MAX_RANK} slots, got {len(self.ranks)}")

    @property
    def rank1_code(self) -> str | None:
        return self.ranks[0]

    @property
    def rank1_ontology(self) -> str | None:
        return self.rank_ontologies[0] if self.rank_ontologies else None


# ─────────────────────────────────────────────────────────────────────────────
# Part 10/11/12 -- per-row Top-k / MRR / Recall@GT
# ─────────────────────────────────────────────────────────────────────────────


def first_gold_rank(ranks: tuple[str | None, ...], gold_codes: tuple[str, ...]) -> int | None:
    """1-based rank of the first ranked slot containing any acceptable gold
    code, or None if the row is unmapped/errored or no gold code appears."""
    gold_set = {c for c in gold_codes if c}
    if not gold_set:
        return None
    for i, code in enumerate(ranks[:MAX_RANK], start=1):
        if code and code in gold_set:
            return i
    return None


def top_k_hit(gold_rank: int | None, k: int) -> bool:
    return gold_rank is not None and gold_rank <= k


def reciprocal_rank(gold_rank: int | None) -> float:
    return 0.0 if gold_rank is None else 1.0 / gold_rank


def recall_at_gt(ranks: tuple[str | None, ...], gold_codes: tuple[str, ...]) -> float | None:
    """Fraction of the query's distinct gold codes recovered within the top-n
    predictions, n = |gold_codes| (Part 12). None when there are no gold
    codes (not expected in OLS-EFO, but never fabricated as 0)."""
    gold_set = {c for c in gold_codes if c}
    if not gold_set:
        return None
    n = min(len(gold_set), MAX_RANK)
    top_n = {c for c in ranks[:n] if c}
    recovered = len(gold_set & top_n)
    return recovered / len(gold_set)


@dataclass(frozen=True)
class RowMetrics:
    query_id: int
    gold_rank: int | None
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    reciprocal_rank: float
    recall_at_gt: float | None


def score_prediction(record: PredictionRecord) -> RowMetrics:
    """Score one row. Execution errors and unmapped rows fall straight through
    (ranks are all-None or partially None) and score 0 on every rank-based
    metric -- they are never dropped from the denominator (Part 10/11/17)."""
    gold_rank = first_gold_rank(record.ranks, record.gold_codes)
    return RowMetrics(
        query_id=record.query_id,
        gold_rank=gold_rank,
        top1_hit=top_k_hit(gold_rank, 1),
        top3_hit=top_k_hit(gold_rank, 3),
        top5_hit=top_k_hit(gold_rank, 5),
        reciprocal_rank=reciprocal_rank(gold_rank),
        recall_at_gt=recall_at_gt(record.ranks, record.gold_codes),
    )


@dataclass(frozen=True)
class AggregateMetrics:
    n: int
    top1: float
    top3: float
    top5: float
    mrr: float
    recall_at_gt: float
    recall_at_gt_n: int  # number of rows with a defined (non-None) recall_at_gt


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
# Part 14 -- namespace distribution (diagnostic only)
# ─────────────────────────────────────────────────────────────────────────────


def namespace_distribution(records: list[PredictionRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        ontology = r.rank1_ontology or "NONE"
        counts[ontology] += 1
    return dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
# Part 13/14 -- graph-distance summary
# ─────────────────────────────────────────────────────────────────────────────


def graph_relationship_distribution(relationships: list[str]) -> dict[str, int]:
    return dict(Counter(relationships))


def graph_relationship_percentages(
    relationships: list[str], *, denominator: int
) -> dict[str, float]:
    if denominator == 0:
        return {}
    counts = graph_relationship_distribution(relationships)
    return {k: v / denominator for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Part 15/16 -- TP taxonomy (fully automatic -- no manual review gate)
#
# Graph-distance classification (scenario1_graph_distance.EfoGraphIndex.classify)
# is itself fully automatic and deterministic, so every row maps to exactly
# one taxonomy category with no human input required:
#
#     Same                                -> TP-Identical
#     More Specific / More General / Sibling -> TP-Related   (automatic)
#     Unrelated                          -> FP-Error
#     unmapped / execution-error, gold present -> FN  (locked: an execution
#         error earns the SAME zero TP-taxonomy credit as a genuine unmapped
#         row -- consistent with score_prediction(), which also scores an
#         error row as 0 on every rank-based metric. This does not replace
#         the separate execution_diagnostics()/error_count reporting; it
#         only fixes how much TP-taxonomy credit an error row earns: zero.)
#
# manual_review_required.csv, if generated, is diagnostic-only -- it is never
# read back into this classification.
# ─────────────────────────────────────────────────────────────────────────────

TP_IDENTICAL = "TP-Identical"
TP_RELATED = "TP-Related"
TP_CONTRIBUTION = "TP-Contribution"  # not applicable to OLS-EFO -- see module note below
FN = "FN"
FP_ERROR = "FP-Error"
FP_INCORRECT_CONTRIBUTION = "FP-Incorrect-Contribution"  # not applicable to OLS-EFO
NOT_APPLICABLE = "NOT-APPLICABLE"  # gold-negative row: not modeled in OLS-EFO (every query has gold)

_GRAPH_RELATED_RELATIONSHIPS = frozenset({"More Specific", "More General", "Sibling"})


@dataclass(frozen=True)
class TpTaxonomyRow:
    query_id: int
    category: str  # one of the constants above


@dataclass(frozen=True)
class TpTaxonomyResult:
    counts: dict[str, int]
    precision: float | None
    recall: float | None
    f1: float | None
    status: str  # STATUS_OK


def classify_tp_taxonomy_row(
    *,
    query_id: int,
    status: str,
    rank1_code: str | None,
    gold_codes: tuple[str, ...],
    graph_relationship: str | None,
) -> TpTaxonomyRow:
    """Classify exactly one query's outcome into the Part 15 taxonomy.

    Fully automatic -- see the module note above. TP-Contribution and
    FP-Incorrect-Contribution are part of the taxonomy vocabulary but are
    never produced here: for OLS-EFO every evaluated query has at least one
    gold mapping, so the has_gold=False branches below (which return
    NOT_APPLICABLE) are not expected to be reached; they exist only so this
    function does not crash on a hypothetical zero-gold row.
    """
    has_gold = bool(gold_codes)

    if status in (STATUS_UNMAPPED, STATUS_ERROR) or rank1_code is None:
        if has_gold:
            return TpTaxonomyRow(query_id, FN)
        return TpTaxonomyRow(query_id, NOT_APPLICABLE)

    if has_gold and rank1_code in gold_codes:
        return TpTaxonomyRow(query_id, TP_IDENTICAL)

    if has_gold:
        if graph_relationship in _GRAPH_RELATED_RELATIONSHIPS:
            return TpTaxonomyRow(query_id, TP_RELATED)
        return TpTaxonomyRow(query_id, FP_ERROR)

    return TpTaxonomyRow(query_id, NOT_APPLICABLE)


def aggregate_tp_taxonomy(rows: list[TpTaxonomyRow]) -> TpTaxonomyResult:
    """Aggregate the fully-automatic Part 15/16 classification. There is no
    pending/manual-review gate: Precision/Recall/F1 are always numeric
    (STATUS_OK) whenever graph classification itself is available -- which
    it always is, since EfoGraphIndex.classify() fails loudly rather than
    silently degrading (see run_scenario1_ols_efo.py _finalize_outputs).
    NOT_APPLICABLE rows (see classify_tp_taxonomy_row) are counted but
    excluded from tp/fp/fn, same as before.
    """
    counts: Counter[str] = Counter(r.category for r in rows)

    tp = counts.get(TP_IDENTICAL, 0) + counts.get(TP_RELATED, 0) + counts.get(TP_CONTRIBUTION, 0)
    fp = counts.get(FP_ERROR, 0) + counts.get(FP_INCORRECT_CONTRIBUTION, 0)
    fn = counts.get(FN, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return TpTaxonomyResult(
        counts=dict(counts),
        precision=precision,
        recall=recall,
        f1=f1,
        status=STATUS_OK,
    )


def exact_only_diagnostic(rows: list[TpTaxonomyRow]) -> dict[str, float]:
    """An explicitly-labelled diagnostic that treats every graph-related
    (TP-Related) row as FP-Error instead (i.e. exact-match-only, no graph
    credit at all). Never the official TP-taxonomy result (Part 16) --
    callers must label it as such."""
    counts: Counter[str] = Counter()
    for r in rows:
        category = FP_ERROR if r.category == TP_RELATED else r.category
        counts[category] += 1
    tp = counts.get(TP_IDENTICAL, 0) + counts.get(TP_CONTRIBUTION, 0)
    fp = counts.get(FP_ERROR, 0) + counts.get(FP_INCORRECT_CONTRIBUTION, 0)
    fn = counts.get(FN, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Part 17 -- execution-error / mapped / unmapped diagnostics
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


# ─────────────────────────────────────────────────────────────────────────────
# Part 21 -- Scenario 1 metric table
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricTableRow:
    metric: str
    value: float | str | None
    numerator: float | None
    denominator: int | None
    evaluation_unit: str
    status: str


def build_metric_table(
    *,
    unique_query_agg: AggregateMetrics,
    graph_relationship_pcts: dict[str, float],
    graph_status: str,
    tp_result: TpTaxonomyResult,
) -> list[MetricTableRow]:
    rows = [
        MetricTableRow("Top-1", unique_query_agg.top1, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, STATUS_OK),
        MetricTableRow("Top-3", unique_query_agg.top3, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, STATUS_OK),
        MetricTableRow("Top-5", unique_query_agg.top5, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, STATUS_OK),
        MetricTableRow("MRR", unique_query_agg.mrr, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, STATUS_OK),
        MetricTableRow(
            "Recall@GT",
            unique_query_agg.recall_at_gt,
            None,
            unique_query_agg.recall_at_gt_n,
            EVAL_UNIT_UNIQUE_QUERY,
            STATUS_OK,
        ),
    ]
    for label, key in (
        ("% Same", "Same"),
        ("% More Specific", "More Specific"),
        ("% More General", "More General"),
        ("% Sibling", "Sibling"),
        ("% Unrelated", "Unrelated"),
    ):
        rows.append(
            MetricTableRow(
                label,
                graph_relationship_pcts.get(key),
                None,
                unique_query_agg.n,
                EVAL_UNIT_UNIQUE_QUERY,
                graph_status,
            )
        )

    # Precision/Recall/F1 (Part 16) are fully automatic -- see
    # aggregate_tp_taxonomy -- so tp_result.status is always STATUS_OK and
    # these three values are always numeric whenever graph classification
    # itself is available (which build_metric_table's caller guarantees by
    # construction: EfoGraphIndex.classify() fails loudly rather than
    # silently degrading, so this function is never called with partial
    # graph data).
    rows.append(
        MetricTableRow(
            "Precision", tp_result.precision, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, tp_result.status
        )
    )
    rows.append(
        MetricTableRow(
            "Recall", tp_result.recall, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, tp_result.status
        )
    )
    rows.append(
        MetricTableRow("F1", tp_result.f1, None, unique_query_agg.n, EVAL_UNIT_UNIQUE_QUERY, tp_result.status)
    )
    return rows
