"""
Scenario 1 (OLS-EFO) output writers: predictions.csv, experiment_config.json
+ resume-fingerprint validation, and every report file under Part 20/21/22.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import statistics
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from llm_ontology_mapper.benchmarking.scenario1_dataset import (
    CanonicalQuery,
    DatasetAudit,
    MappingPairRow,
)
from llm_ontology_mapper.benchmarking.scenario1_graph_distance import (
    EFO_URL,
    EFO_VERSION,
    PINNED_COMMIT,
    SOURCE_FILE,
    SOURCE_REPOSITORY,
    EfoGraphIndex,
    GraphDistanceResult,
)
from llm_ontology_mapper.benchmarking.scenario1_metrics import (
    MAX_RANK,
    MetricTableRow,
    PredictionRecord,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    RankSlot,
    SapBertHealth,
    Scenario1RowResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Part 8 -- predictions.csv (one row per canonical/unique query)
# ─────────────────────────────────────────────────────────────────────────────

PREDICTIONS_CSV_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "gold_codes",
    "gold_labels",
    "gold_count",
    "status",
    "mapped_code",
    "mapped_term",
    "mapped_ontology",
    "confidence",
    "rank_1_code",
    "rank_1_label",
    "rank_1_ontology",
    "rank_2_code",
    "rank_2_label",
    "rank_2_ontology",
    "rank_3_code",
    "rank_3_label",
    "rank_3_ontology",
    "rank_4_code",
    "rank_4_label",
    "rank_4_ontology",
    "rank_5_code",
    "rank_5_label",
    "rank_5_ontology",
    "first_gold_rank",
    "top1_hit",
    "top3_hit",
    "top5_hit",
    "reciprocal_rank",
    "recall_at_gt",
    "graph_relationship",
    "graph_matched_gold_code",
    "execution_error",
    "error_type",
    "error_stage",
    "error_message",
    "end_to_end_seconds",
    "query_planner_seconds",
    "retrieval_seconds",
    "reranker_seconds",
    "llm_seconds",
    "total_input_tokens",
    "total_cached_input_tokens",
    "total_output_tokens",
    "total_reasoning_tokens",
    "api_cost_usd",
    "retrieval_request_count",
    "retrieval_retry_count",
    "retrieval_recovered_error_count",
    "retrieval_final_error_count",
    "retrieval_error_sources",
    "retrieval_error_types",
)

_PIPE = "|"


def _join(values: Sequence[str | None]) -> str:
    return _PIPE.join("" if v is None else str(v) for v in values)


def _split(value: str) -> list[str]:
    if value == "":
        return []
    return value.split(_PIPE)


def row_to_csv_dict(
    row: Scenario1RowResult,
    *,
    row_metrics: Any,  # scenario1_metrics.RowMetrics
    graph: GraphDistanceResult | None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "query_id": row.query_id,
        "query": row.query,
        "gold_codes": _join(row.gold_codes),
        "gold_labels": _join([g if g is not None else "" for g in row.gold_labels]),
        "gold_count": row.gold_count,
        "status": row.status,
        "mapped_code": row.mapped_code,
        "mapped_term": row.mapped_term,
        "mapped_ontology": row.mapped_ontology,
        "confidence": row.confidence,
        "first_gold_rank": row_metrics.gold_rank,
        "top1_hit": row_metrics.top1_hit,
        "top3_hit": row_metrics.top3_hit,
        "top5_hit": row_metrics.top5_hit,
        "reciprocal_rank": row_metrics.reciprocal_rank,
        "recall_at_gt": row_metrics.recall_at_gt,
        "graph_relationship": graph.graph_relationship if graph else None,
        "graph_matched_gold_code": graph.graph_matched_gold_code if graph else None,
        "execution_error": row.status == "error",
        "error_type": row.error_type,
        "error_stage": row.error_stage,
        "error_message": row.error_message,
        "end_to_end_seconds": row.end_to_end_seconds,
        "query_planner_seconds": row.query_planner_seconds,
        "retrieval_seconds": row.retrieval_seconds,
        "reranker_seconds": row.reranker_seconds,
        "llm_seconds": row.llm_seconds,
        "total_input_tokens": row.total_input_tokens,
        "total_cached_input_tokens": row.total_cached_input_tokens,
        "total_output_tokens": row.total_output_tokens,
        "total_reasoning_tokens": row.total_reasoning_tokens,
        "api_cost_usd": row.api_cost_usd,
        "retrieval_request_count": row.retrieval_request_count,
        "retrieval_retry_count": row.retrieval_retry_count,
        "retrieval_recovered_error_count": row.retrieval_recovered_error_count,
        "retrieval_final_error_count": row.retrieval_final_error_count,
        "retrieval_error_sources": row.retrieval_error_sources,
        "retrieval_error_types": row.retrieval_error_types,
    }
    for i in range(MAX_RANK):
        slot = row.ranks[i] if i < len(row.ranks) else RankSlot()
        prefix = f"rank_{i + 1}_"
        d[prefix + "code"] = slot.code
        d[prefix + "label"] = slot.label
        d[prefix + "ontology"] = slot.ontology
    return d


def csv_row_to_prediction_record(csv_row: dict[str, str]) -> PredictionRecord:
    """Reload one predictions.csv row back into a PredictionRecord so metrics
    can be recomputed with zero mapper calls (--evaluate-existing)."""
    ranks = tuple(
        (csv_row.get(f"rank_{i + 1}_code") or None) for i in range(MAX_RANK)
    )
    rank_ontologies = tuple(
        (csv_row.get(f"rank_{i + 1}_ontology") or None) for i in range(MAX_RANK)
    )
    return PredictionRecord(
        query_id=int(csv_row["query_id"]),
        query=csv_row["query"],
        gold_codes=tuple(_split(csv_row.get("gold_codes", ""))),
        status=csv_row["status"],
        ranks=ranks,
        rank_ontologies=rank_ontologies,
    )


class IncrementalPredictionsCsvWriter:
    """Writes/flushes each row immediately (crash never loses prior rows).
    Supports append mode for resume (Part 18)."""

    def __init__(self, path: Path, *, append: bool = False) -> None:
        self._path = path
        self._append = append
        self._fh: IO[str] | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> IncrementalPredictionsCsvWriter:
        mode = "a" if self._append and self._path.exists() else "w"
        write_header = mode == "w"
        self._fh = self._path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        if write_header:
            self._writer.writeheader()
            self._fh.flush()
        return self

    def write_row(self, row_dict: dict[str, Any]) -> None:
        assert self._writer is not None and self._fh is not None
        self._writer.writerow(row_dict)
        self._fh.flush()

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            self._fh.close()


def read_existing_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─────────────────────────────────────────────────────────────────────────────
# Resume reliability: quarantine prior error rows so --resume retries them
# instead of skipping them forever (reliability-audit follow-up). See
# quarantine_error_rows_for_resume() below.
# ─────────────────────────────────────────────────────────────────────────────

RETRY_ERROR_HISTORY_CSV_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "previous_status",
    "error_type",
    "error_stage",
    "error_message",
    "resume_timestamp",
    "provider",
    "model",
    "attempt_source",
)


def _append_retry_error_history(
    error_rows: list[dict[str, str]],
    path: Path,
    *,
    resume_timestamp: str,
    provider: str | None,
    model: str | None,
) -> None:
    """Append prior-error rows to retry_error_history.csv for audit -- this
    file is diagnostic only and is never read back as a canonical
    prediction (see csv_row_to_prediction_record / read_existing_predictions,
    neither of which touch this file)."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RETRY_ERROR_HISTORY_CSV_FIELDS))
        if write_header:
            writer.writeheader()
        for row in error_rows:
            writer.writerow(
                {
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "previous_status": row.get("status"),
                    "error_type": row.get("error_type"),
                    "error_stage": row.get("error_stage"),
                    "error_message": row.get("error_message"),
                    "resume_timestamp": resume_timestamp,
                    "provider": provider or "",
                    "model": model or "",
                    "attempt_source": "prior_resume_error",
                }
            )


def quarantine_error_rows_for_resume(
    output_dir: Path,
    *,
    resume_timestamp: str,
    provider: str | None = None,
    model: str | None = None,
) -> set[int]:
    """Prepare an output directory for --resume so status="error" rows are
    retried instead of skipped forever.

    predictions.csv may currently contain three kinds of rows for a query_id
    that has ever been attempted: "mapped", "unmapped" (both terminal
    scientific results) or "error" (an execution failure -- e.g. a SapBERT
    outage -- that produced no scientific result at all). Only mapped/
    unmapped rows should count as "completed" for resume purposes.

    This reads predictions.csv, splits rows by status, appends any error
    rows to retry_error_history.csv (audit trail, never treated as a
    canonical prediction), and -- only when there were error rows to
    remove -- atomically rewrites predictions.csv (temp file + os.replace)
    to contain ONLY the mapped/unmapped rows. iter_predictions() will then
    naturally retry the removed query_ids and append their new result,
    so predictions.csv never ends up with two canonical rows for the same
    query_id.

    Returns the set of query_ids that remain completed (mapped or unmapped)
    and should be skipped on this resume.
    """
    predictions_path = output_dir / "predictions.csv"
    existing_rows = read_existing_predictions(predictions_path)
    if not existing_rows:
        return set()

    canonical_rows = [r for r in existing_rows if r.get("status") in ("mapped", "unmapped")]
    error_rows = [r for r in existing_rows if r.get("status") == "error"]

    if error_rows:
        _append_retry_error_history(
            error_rows,
            output_dir / "retry_error_history.csv",
            resume_timestamp=resume_timestamp,
            provider=provider,
            model=model,
        )
        tmp_path = predictions_path.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
            writer.writeheader()
            for row in canonical_rows:
                writer.writerow({field: row.get(field, "") for field in PREDICTIONS_CSV_FIELDS})
        os.replace(tmp_path, predictions_path)

    return {int(r["query_id"]) for r in canonical_rows}


# ─────────────────────────────────────────────────────────────────────────────
# unique_queries.csv
# ─────────────────────────────────────────────────────────────────────────────


def write_unique_queries_csv(canonical_queries: list[CanonicalQuery], path: Path) -> None:
    fields = ("query_id", "source_query", "gold_codes", "gold_labels", "original_row_indices", "original_mapping_pair_count")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cq in canonical_queries:
            writer.writerow(
                {
                    "query_id": cq.query_id,
                    "source_query": cq.source_query,
                    "gold_codes": _join(cq.gold_codes),
                    "gold_labels": _join([g if g is not None else "" for g in cq.gold_labels]),
                    "original_row_indices": _join([str(i) for i in cq.original_row_indices]),
                    "original_mapping_pair_count": cq.original_mapping_pair_count,
                }
            )


# ─────────────────────────────────────────────────────────────────────────────
# mapping_pair_expanded_predictions.csv (Part 4 SECONDARY denominator)
# ─────────────────────────────────────────────────────────────────────────────


def write_mapping_pair_expanded_csv(
    pairs: list[MappingPairRow],
    predictions_by_query_id: dict[int, dict[str, str]],
    path: Path,
) -> None:
    from llm_ontology_mapper.benchmarking.scenario1_metrics import (
        first_gold_rank,
        reciprocal_rank,
        top_k_hit,
    )

    fields = (
        "query_id",
        "source_query",
        "gold_code",
        "gold_label",
        "raw_row_index",
        "rank_1_code",
        "rank_2_code",
        "rank_3_code",
        "rank_4_code",
        "rank_5_code",
        "first_gold_rank",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "reciprocal_rank",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            pred = predictions_by_query_id.get(pair.query_id)
            ranks = tuple((pred.get(f"rank_{i + 1}_code") or None) for i in range(MAX_RANK)) if pred else (None,) * MAX_RANK
            gold_rank = first_gold_rank(ranks, (pair.gold_code,))
            row = {
                "query_id": pair.query_id,
                "source_query": pair.source_query,
                "gold_code": pair.gold_code,
                "gold_label": pair.gold_label,
                "raw_row_index": pair.raw_row_index,
                "first_gold_rank": gold_rank,
                "top1_hit": top_k_hit(gold_rank, 1),
                "top3_hit": top_k_hit(gold_rank, 3),
                "top5_hit": top_k_hit(gold_rank, 5),
                "reciprocal_rank": reciprocal_rank(gold_rank),
                "status": pred.get("status") if pred else "missing_prediction",
            }
            for i in range(MAX_RANK):
                row[f"rank_{i + 1}_code"] = ranks[i]
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# graph_distance_rows.csv / graph_distance_summary.csv
# ─────────────────────────────────────────────────────────────────────────────


GRAPH_DISTANCE_ROWS_CSV_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "predicted_code",
    "predicted_label",
    "gold_codes",
    "graph_relationship",
    "graph_matched_gold_code",
    "graph_shared_parent_code",
    "graph_prediction_found",
    "graph_gold_found",
    "note",
)


def write_graph_distance_rows_csv(
    rows: list[tuple[int, str, str | None, GraphDistanceResult]], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(GRAPH_DISTANCE_ROWS_CSV_FIELDS))
        writer.writeheader()
        for query_id, query, predicted_label, g in rows:
            writer.writerow(
                {
                    "query_id": query_id,
                    "query": query,
                    "predicted_code": g.predicted_code,
                    "predicted_label": predicted_label,
                    "gold_codes": _join(list(g.gold_codes)),
                    "graph_relationship": g.graph_relationship,
                    "graph_matched_gold_code": g.graph_matched_gold_code,
                    "graph_shared_parent_code": g.graph_shared_parent_code,
                    "graph_prediction_found": g.graph_prediction_found,
                    "graph_gold_found": g.graph_gold_found,
                    "note": g.note,
                }
            )


def write_graph_reference_metadata(index: EfoGraphIndex, path: Path) -> None:
    """Record exactly which reference hierarchy artifacts produced the graph
    classification -- source repository, source file, pinned commit, EFO
    version, and the checksums actually verified at load time (Part 2 of the
    graph-distance task: never claim EFO version without recording how it
    was pinned/verified)."""
    payload = {
        "source_repository": SOURCE_REPOSITORY,
        "source_file": SOURCE_FILE,
        "pinned_commit": PINNED_COMMIT,
        "efo_version": EFO_VERSION,
        "efo_url": EFO_URL,
        "edges_file": str(index.edges_path),
        "edges_sha256": index.edges_sha256,
        "entailed_edges_file": str(index.entailed_path),
        "entailed_edges_sha256": index.entailed_sha256,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_graph_distance_summary_csv(distribution: dict[str, int], denominator: int, path: Path) -> None:
    fields = ("relationship", "count", "percentage", "denominator")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for relationship, count in sorted(distribution.items()):
            writer.writerow(
                {
                    "relationship": relationship,
                    "count": count,
                    "percentage": count / denominator if denominator else 0.0,
                    "denominator": denominator,
                }
            )


# ─────────────────────────────────────────────────────────────────────────────
# namespace_distribution.csv
# ─────────────────────────────────────────────────────────────────────────────


def write_namespace_distribution_csv(distribution: dict[str, int], path: Path) -> None:
    total = sum(distribution.values())
    fields = ("ontology", "count", "percentage")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ontology, count in sorted(distribution.items(), key=lambda kv: -kv[1]):
            writer.writerow({"ontology": ontology, "count": count, "percentage": count / total if total else 0.0})


# ─────────────────────────────────────────────────────────────────────────────
# manual_review_required.csv (Part 15)
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_REVIEW_CSV_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "predicted_code",
    "predicted_label",
    "predicted_ontology",
    "gold_codes",
    "gold_labels",
    "graph_relationship",
    "graph_matched_gold_code",
    "reviewer_decision",
    "reviewer_notes",
)


def write_manual_review_required_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANUAL_REVIEW_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out.setdefault("reviewer_decision", "")
            out.setdefault("reviewer_notes", "")
            writer.writerow(out)


def read_manual_review_decisions(path: Path) -> dict[tuple[int, str], str]:
    """Load a completed manual_review_required.csv into
    {(query_id, predicted_code): reviewer_decision}. Blank/missing decisions
    are omitted -- never fabricated."""
    decisions: dict[tuple[int, str], str] = {}
    if not path.exists():
        return decisions
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            decision = (row.get("reviewer_decision") or "").strip()
            if not decision:
                continue
            code = row.get("predicted_code") or ""
            try:
                query_id = int(row["query_id"])
            except (KeyError, ValueError):
                continue
            decisions[(query_id, code)] = decision
    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# execution_diagnostics.csv / telemetry_summary.csv
# ─────────────────────────────────────────────────────────────────────────────


def write_execution_diagnostics_csv(diag: Any, path: Path) -> None:
    fields = ("total", "mapped_count", "unmapped_count", "error_count", "mapped_rate", "unmapped_rate", "error_rate")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "total": diag.total,
                "mapped_count": diag.mapped_count,
                "unmapped_count": diag.unmapped_count,
                "error_count": diag.error_count,
                "mapped_rate": diag.mapped_rate,
                "unmapped_rate": diag.unmapped_rate,
                "error_rate": diag.error_rate,
            }
        )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    idx = (len(data) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    frac = idx - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def write_telemetry_summary_csv(csv_rows: list[dict[str, str]], path: Path) -> None:
    """Supplementary diagnostics (Part 23) computed directly from the saved
    predictions.csv rows -- never mixed into the core Part 21 metric table."""

    def _floats(key: str) -> list[float]:
        out = []
        for r in csv_rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            with contextlib.suppress(ValueError):
                out.append(float(v))
        return out

    e2e = _floats("end_to_end_seconds")
    planner = _floats("query_planner_seconds")
    retrieval = _floats("retrieval_seconds")
    reranker = _floats("reranker_seconds")
    costs = _floats("api_cost_usd")

    fields = (
        "mean_end_to_end_seconds",
        "median_end_to_end_seconds",
        "mean_query_planner_seconds",
        "median_query_planner_seconds",
        "mean_retrieval_seconds",
        "median_retrieval_seconds",
        "mean_reranker_seconds",
        "median_reranker_seconds",
        "total_api_cost_usd",
        "mean_api_cost_per_query_usd",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "mean_end_to_end_seconds": statistics.fmean(e2e) if e2e else None,
                "median_end_to_end_seconds": statistics.median(e2e) if e2e else None,
                "mean_query_planner_seconds": statistics.fmean(planner) if planner else None,
                "median_query_planner_seconds": statistics.median(planner) if planner else None,
                "mean_retrieval_seconds": statistics.fmean(retrieval) if retrieval else None,
                "median_retrieval_seconds": statistics.median(retrieval) if retrieval else None,
                "mean_reranker_seconds": statistics.fmean(reranker) if reranker else None,
                "median_reranker_seconds": statistics.median(reranker) if reranker else None,
                "total_api_cost_usd": sum(costs) if costs else None,
                "mean_api_cost_per_query_usd": statistics.fmean(costs) if costs else None,
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# scenario1_metrics.csv / .md (Part 21)
# ─────────────────────────────────────────────────────────────────────────────


def write_metric_table_csv(rows: list[MetricTableRow], path: Path) -> None:
    fields = ("metric", "value", "numerator", "denominator", "evaluation_unit", "status")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "metric": r.metric,
                    "value": r.value,
                    "numerator": r.numerator,
                    "denominator": r.denominator,
                    "evaluation_unit": r.evaluation_unit,
                    "status": r.status,
                }
            )


def write_metric_table_md(rows: list[MetricTableRow], path: Path, *, notes: list[str] | None = None) -> None:
    lines = [
        "# Scenario 1 -- OLS-EFO metrics (llm-ontology-mapper, local SapBERT, non-strict EFO)",
        "",
        "| Metric | Value | Denominator (N) | Evaluation unit | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        value = f"{r.value:.4f}" if isinstance(r.value, float) else str(r.value)
        lines.append(f"| {r.metric} | {value} | {r.denominator} | {r.evaluation_unit} | {r.status} |")
    if notes:
        lines.append("")
        lines.extend(f"- {n}" for n in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# published_comparison.csv / .md (Part 22) -- never fabricated
# ─────────────────────────────────────────────────────────────────────────────

PUBLISHED_BASELINES_FIELDS: tuple[str, ...] = (
    "benchmark",
    "tool",
    "metric",
    "value",
    "denominator",
    "source_publication",
    "source_table_or_figure",
    "notes",
)


def read_published_baselines(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_published_comparison(
    baselines: list[dict[str, str]],
    this_study_rows: list[MetricTableRow],
    path_csv: Path,
    path_md: Path,
    *,
    benchmark: str = "OLS-EFO",
) -> None:
    this_study_by_metric = {r.metric: r.value for r in this_study_rows}
    text2term_by_metric = {
        b["metric"]: b["value"]
        for b in baselines
        if b.get("benchmark") == benchmark and b.get("tool", "").lower() == "text2term"
    }
    metaharmonizer_by_metric = {
        b["metric"]: b["value"]
        for b in baselines
        if b.get("benchmark") == benchmark and b.get("tool", "").lower() == "metaharmonizer"
    }
    metrics = sorted({r.metric for r in this_study_rows} | set(text2term_by_metric) | set(metaharmonizer_by_metric))

    fields = ("Metric", "text2term (published)", "MetaHarmonizer (published)", "llm-ontology-mapper (this study)")
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            this_value = this_study_by_metric.get(metric)
            writer.writerow(
                {
                    "Metric": metric,
                    "text2term (published)": text2term_by_metric.get(metric, "unavailable"),
                    "MetaHarmonizer (published)": metaharmonizer_by_metric.get(metric, "unavailable"),
                    "llm-ontology-mapper (this study)": (
                        f"{this_value:.4f}" if isinstance(this_value, float) else this_value
                    ),
                }
            )

    lines = [
        "# Published baseline comparison -- OLS-EFO (Scenario 1)",
        "",
        (
            "text2term and MetaHarmonizer were NOT rerun for this study. Their values "
            "below (if present) come only from `published_baselines.csv`, a manually "
            "curated file citing the exact publication/table/figure -- never fabricated "
            "or estimated. 'unavailable' means no verified value has been supplied yet."
        ),
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for metric in metrics:
        this_value = this_study_by_metric.get(metric)
        this_str = f"{this_value:.4f}" if isinstance(this_value, float) else str(this_value)
        lines.append(
            f"| {metric} | {text2term_by_metric.get(metric, 'unavailable')} | "
            f"{metaharmonizer_by_metric.get(metric, 'unavailable')} | {this_str} |"
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# dataset_validation.json
# ─────────────────────────────────────────────────────────────────────────────


def write_dataset_validation_json(audit: DatasetAudit, canonical_query_count: int, path: Path) -> None:
    payload = audit.to_dict()
    payload["canonical_unique_query_count"] = canonical_query_count
    payload["max_gold_codes_within_rank_capacity"] = audit.max_gold_codes_per_query <= MAX_RANK
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Part 19 -- experiment_config.json + Part 18 -- resume fingerprint
# ─────────────────────────────────────────────────────────────────────────────

RESUME_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "source_dataset_sha256",
    "provider",
    "model",
    "reasoning_effort",
    "temperature_mode",
    "temperature",
    "seed",
    "target_ontology",
    "retrieval_mode",
    "strict_target_ontology",
    "max_alternatives",
    "sapbert_url",
    "llm_ontology_mapper_git_commit",
)


def get_git_commit_hash(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_git_dirty(repo_dir: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def build_experiment_config(
    *,
    source_dataset_path: Path,
    source_dataset_sha256: str,
    raw_row_count: int,
    unique_mapping_pair_count: int,
    unique_query_count: int,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    temperature: float | None,
    temperature_mode: str,
    seed: int,
    target_ontology: str,
    retrieval_mode: str,
    strict_target_ontology: bool,
    max_alternatives: int,
    sapbert_url: str,
    sapbert_health: SapBertHealth,
    repo_dir: Path,
    start_timestamp: str,
) -> dict[str, Any]:
    return {
        "experiment_name": "scenario1_ols_efo",
        "source_dataset_path": str(source_dataset_path),
        "source_dataset_sha256": source_dataset_sha256,
        "raw_row_count": raw_row_count,
        "unique_mapping_pair_count": unique_mapping_pair_count,
        "unique_query_count": unique_query_count,
        "llm_ontology_mapper_git_commit": get_git_commit_hash(repo_dir),
        "working_tree_dirty": get_git_dirty(repo_dir),
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort if reasoning_effort is not None else "N/A",
        "temperature": temperature,
        "temperature_mode": temperature_mode,
        "seed": seed,
        "target_ontology": target_ontology,
        "retrieval_mode": retrieval_mode,
        "strict_target_ontology": strict_target_ontology,
        "max_alternatives": max_alternatives,
        "sapbert_url": sapbert_url,
        "sapbert_health_response": sapbert_health.raw_response,
        "sapbert_model": sapbert_health.model,
        "efo_index_version": "unknown / not exposed by deployed SapBERT service",
        "start_timestamp": start_timestamp,
        "end_timestamp": None,
        "completed": False,
        "rows_completed": 0,
    }


def write_experiment_config(config: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, default=str)


def load_experiment_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class ResumeConfigMismatchError(RuntimeError):
    """Raised when --resume targets an output directory whose locked
    experiment-defining fields differ from the requested run (Part 18)."""


def validate_resume(existing_config: dict[str, Any], new_config: dict[str, Any]) -> None:
    mismatches = []
    for field_name in RESUME_FINGERPRINT_FIELDS:
        old_value = existing_config.get(field_name)
        new_value = new_config.get(field_name)
        if old_value != new_value:
            mismatches.append(f"{field_name}: existing={old_value!r} != requested={new_value!r}")
    if mismatches:
        raise ResumeConfigMismatchError(
            "Refusing to resume: the following experiment-defining fields differ "
            "from the run already in this output directory:\n  " + "\n  ".join(mismatches)
        )
