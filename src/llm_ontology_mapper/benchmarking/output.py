"""
Output writers, summary computation, and progress printing for the model
benchmark: raw per-row CSV, per-run summary CSV, model summary CSV,
reproducibility summary CSV, and benchmark_config.json.

Kept separate from runner.py so execution (network calls) and I/O/reporting
are independently testable.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

from llm_ontology_mapper.benchmarking.pricing import ModelPricing
from llm_ontology_mapper.benchmarking.runner import RowResult, compute_run_metrics
from llm_ontology_mapper.benchmarking.scoring import top1_exact_agreement, top5_set_agreement
from llm_ontology_mapper.models import LogicType

# ─────────────────────────────────────────────────────────────────────────────
# Raw per-row CSV
# ─────────────────────────────────────────────────────────────────────────────

ROW_CSV_FIELDS: tuple[str, ...] = (
    "input_row",
    "source_variable",
    "source_label",
    "source_description",
    "target_ontology",
    "gold_code_raw",
    "gold_codes_normalized",
    "gold_target_term",
    "mapped_status",
    "mapped_code",
    "mapped_code_normalized",
    "mapped_term",
    "mapped_ontology",
    "confidence",
    "logic_type",
    "alternative_1_code",
    "alternative_1_term",
    "alternative_1_ontology",
    "alternative_1_confidence",
    "alternative_2_code",
    "alternative_2_term",
    "alternative_2_ontology",
    "alternative_2_confidence",
    "alternative_3_code",
    "alternative_3_term",
    "alternative_3_ontology",
    "alternative_3_confidence",
    "alternative_4_code",
    "alternative_4_term",
    "alternative_4_ontology",
    "alternative_4_confidence",
    "gold_rank",
    "top1_correct",
    "top5_hit",
    "tp",
    "fp",
    "fn",
    "tn",
    "end_to_end_seconds",
    "query_planner_seconds",
    "retrieval_seconds",
    "reranker_seconds",
    "llm_seconds",
    "planner_input_tokens",
    "planner_cached_input_tokens",
    "planner_output_tokens",
    "planner_reasoning_tokens",
    "reranker_input_tokens",
    "reranker_cached_input_tokens",
    "reranker_output_tokens",
    "reranker_reasoning_tokens",
    "total_input_tokens",
    "total_cached_input_tokens",
    "total_output_tokens",
    "total_reasoning_tokens",
    "api_cost_usd",
    "model",
    "requested_reasoning_effort",
    "temperature",
    "temperature_mode",
    "seed",
    "retrieval_mode",
    "max_alternatives",
    "run_number",
    "system_fingerprint",
    "retrieval_request_count",
    "retrieval_retry_count",
    "retrieval_recovered_error_count",
    "retrieval_final_error_count",
    "retrieval_error_sources",
    "retrieval_error_types",
    "error_type",
    "error_message",
)


def row_to_csv_dict(row: RowResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "input_row": row.input_row,
        "source_variable": row.source_variable,
        "source_label": row.source_label,
        "source_description": row.source_description,
        "target_ontology": row.target_ontology,
        "gold_code_raw": row.gold_code_raw,
        "gold_codes_normalized": "|".join(row.gold_codes_normalized),
        "gold_target_term": row.gold_target_term,
        "mapped_status": row.mapped_status,
        "mapped_code": row.mapped_code,
        "mapped_code_normalized": row.mapped_code_normalized,
        "mapped_term": row.mapped_term,
        "mapped_ontology": row.mapped_ontology,
        "confidence": row.confidence,
        "logic_type": row.logic_type,
        "gold_rank": row.gold_rank,
        "top1_correct": row.top1_correct,
        "top5_hit": row.top5_hit,
        "tp": row.tp,
        "fp": row.fp,
        "fn": row.fn,
        "tn": row.tn,
        "end_to_end_seconds": row.end_to_end_seconds,
        "query_planner_seconds": row.query_planner_seconds,
        "retrieval_seconds": row.retrieval_seconds,
        "reranker_seconds": row.reranker_seconds,
        "llm_seconds": row.llm_seconds,
        "planner_input_tokens": row.planner_input_tokens,
        "planner_cached_input_tokens": row.planner_cached_input_tokens,
        "planner_output_tokens": row.planner_output_tokens,
        "planner_reasoning_tokens": row.planner_reasoning_tokens,
        "reranker_input_tokens": row.reranker_input_tokens,
        "reranker_cached_input_tokens": row.reranker_cached_input_tokens,
        "reranker_output_tokens": row.reranker_output_tokens,
        "reranker_reasoning_tokens": row.reranker_reasoning_tokens,
        "total_input_tokens": row.total_input_tokens,
        "total_cached_input_tokens": row.total_cached_input_tokens,
        "total_output_tokens": row.total_output_tokens,
        "total_reasoning_tokens": row.total_reasoning_tokens,
        "api_cost_usd": row.api_cost_usd,
        "model": row.model,
        "requested_reasoning_effort": row.requested_reasoning_effort,
        "temperature": row.temperature,
        "temperature_mode": row.temperature_mode,
        "seed": row.seed,
        "retrieval_mode": row.retrieval_mode,
        "max_alternatives": row.max_alternatives,
        "run_number": row.run_number,
        "system_fingerprint": row.system_fingerprint,
        "retrieval_request_count": row.retrieval_request_count,
        "retrieval_retry_count": row.retrieval_retry_count,
        "retrieval_recovered_error_count": row.retrieval_recovered_error_count,
        "retrieval_final_error_count": row.retrieval_final_error_count,
        "retrieval_error_sources": row.retrieval_error_sources,
        "retrieval_error_types": row.retrieval_error_types,
        "error_type": row.error_type,
        "error_message": row.error_message,
    }
    alts = row.alternatives + [None] * max(0, 4 - len(row.alternatives))  # type: ignore[list-item]
    for i in range(4):
        slot = alts[i]
        prefix = f"alternative_{i + 1}_"
        if slot is None:
            d[prefix + "code"] = None
            d[prefix + "term"] = None
            d[prefix + "ontology"] = None
            d[prefix + "confidence"] = None
        else:
            d[prefix + "code"] = slot.code
            d[prefix + "term"] = slot.term
            d[prefix + "ontology"] = slot.ontology
            d[prefix + "confidence"] = slot.confidence
    return d


class IncrementalRawCsvWriter:
    """Writes/flushes each row immediately so a crash never loses prior rows."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: IO[str] | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> IncrementalRawCsvWriter:
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(ROW_CSV_FIELDS))
        self._writer.writeheader()
        self._fh.flush()
        return self

    def write_row(self, row: RowResult) -> None:
        assert self._writer is not None and self._fh is not None
        self._writer.writerow(row_to_csv_dict(row))
        self._fh.flush()

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Run / model / reproducibility summaries
# ─────────────────────────────────────────────────────────────────────────────


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    idx = (len(data) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    frac = idx - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def _sum_optional_seq(values: Iterable[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def build_run_summary(
    *,
    model: str,
    run_number: int,
    run_complete: bool,
    rows: Sequence[RowResult],
) -> dict[str, Any]:
    """Assumes the run's total_run_seconds is provided by the caller (measured
    around the whole run loop) and merged into the returned dict afterward via
    `total_run_seconds=` -- kept out of this signature so pure aggregation
    logic (no timer dependency) stays independently testable.
    """
    metrics, error_count = compute_run_metrics(rows)
    evaluated = [r for r in rows if r.mapped_status != "error"]
    mapped_count = sum(1 for r in evaluated if r.mapped_status == "mapped")
    unmapped_count = sum(1 for r in evaluated if r.mapped_status == "unmapped")

    logic_counts: dict[str, int] = {lt.value: 0 for lt in LogicType}
    for r in evaluated:
        if r.logic_type:
            logic_counts[r.logic_type] = logic_counts.get(r.logic_type, 0) + 1

    e2e = [r.end_to_end_seconds for r in rows]
    llm_secs = [r.llm_seconds for r in evaluated if r.llm_seconds is not None]
    retrieval_secs = [r.retrieval_seconds for r in evaluated if r.retrieval_seconds is not None]

    total_input = _sum_optional_seq(r.total_input_tokens for r in evaluated)
    total_cached = _sum_optional_seq(r.total_cached_input_tokens for r in evaluated)
    total_output = _sum_optional_seq(r.total_output_tokens for r in evaluated)
    total_reasoning = _sum_optional_seq(r.total_reasoning_tokens for r in evaluated)
    costs = [r.api_cost_usd for r in evaluated if r.api_cost_usd is not None]
    total_cost = sum(costs) if costs else None

    # Retrieval retry/error telemetry is diagnostic-only (see RowResult
    # retrieval_* fields) and computed over ALL rows, not just evaluated --
    # a row that errors after a billed retrieval retry still had one.
    total_retrieval_retries = _sum_optional_seq(r.retrieval_retry_count for r in rows) or 0
    rows_with_retrieval_retries = sum(1 for r in rows if (r.retrieval_retry_count or 0) > 0)
    rows_with_final_retrieval_errors = sum(
        1 for r in rows if (r.retrieval_final_error_count or 0) > 0
    )

    return {
        "model": model,
        "run_number": run_number,
        "run_complete": run_complete,
        "rows_total": len(rows),
        "rows_evaluated": len(evaluated),
        "error_count": error_count,
        "tp_total": metrics.tp_total,
        "fp_total": metrics.fp_total,
        "fn_total": metrics.fn_total,
        "tn_total": metrics.tn_total,
        "weighted_precision": metrics.weighted_precision,
        "weighted_recall": metrics.weighted_recall,
        "weighted_f1": metrics.weighted_f1,
        "top1_correct_count": metrics.top1_correct_count,
        "top1_accuracy": metrics.top1_accuracy,
        "top5_hit_count": metrics.top5_hit_count,
        "top5_hit_rate": metrics.top5_hit_rate,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "direct_count": logic_counts.get("direct", 0),
        "rag_count": logic_counts.get("rag", 0),
        "hybrid_count": logic_counts.get("hybrid", 0),
        "llm_count": logic_counts.get("llm", 0),
        "total_run_seconds": None,  # filled in by caller with true wall-clock time
        "mean_end_to_end_seconds_per_term": statistics.fmean(e2e) if e2e else 0.0,
        "median_end_to_end_seconds_per_term": statistics.median(e2e) if e2e else 0.0,
        "p95_end_to_end_seconds_per_term": _percentile(e2e, 0.95),
        "total_llm_seconds": sum(llm_secs) if llm_secs else 0.0,
        "mean_llm_seconds_per_term": statistics.fmean(llm_secs) if llm_secs else 0.0,
        "median_llm_seconds_per_term": statistics.median(llm_secs) if llm_secs else 0.0,
        "p95_llm_seconds_per_term": _percentile(llm_secs, 0.95),
        "total_retrieval_seconds": sum(retrieval_secs) if retrieval_secs else 0.0,
        "total_input_tokens": total_input,
        "total_cached_input_tokens": total_cached,
        "total_output_tokens": total_output,
        "total_reasoning_tokens": total_reasoning,
        "total_api_cost_usd": total_cost,
        "mean_api_cost_per_term_usd": (
            total_cost / len(evaluated) if total_cost is not None and evaluated else None
        ),
        "total_retrieval_retries": total_retrieval_retries,
        "rows_with_retrieval_retries": rows_with_retrieval_retries,
        "rows_with_final_retrieval_errors": rows_with_final_retrieval_errors,
    }


RUN_SUMMARY_FIELDS: tuple[str, ...] = (
    "model",
    "run_number",
    "run_complete",
    "rows_total",
    "rows_evaluated",
    "error_count",
    "tp_total",
    "fp_total",
    "fn_total",
    "tn_total",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "top1_correct_count",
    "top1_accuracy",
    "top5_hit_count",
    "top5_hit_rate",
    "mapped_count",
    "unmapped_count",
    "direct_count",
    "rag_count",
    "hybrid_count",
    "llm_count",
    "total_run_seconds",
    "mean_end_to_end_seconds_per_term",
    "median_end_to_end_seconds_per_term",
    "p95_end_to_end_seconds_per_term",
    "total_llm_seconds",
    "mean_llm_seconds_per_term",
    "median_llm_seconds_per_term",
    "p95_llm_seconds_per_term",
    "total_retrieval_seconds",
    "total_input_tokens",
    "total_cached_input_tokens",
    "total_output_tokens",
    "total_reasoning_tokens",
    "total_api_cost_usd",
    "mean_api_cost_per_term_usd",
    "total_retrieval_retries",
    "rows_with_retrieval_retries",
    "rows_with_final_retrieval_errors",
)


def write_single_row_csv(summary: dict[str, Any], path: Path, fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in fields})


def build_reproducibility_summary(
    *,
    model: str,
    rows_run1: Sequence[RowResult],
    rows_run2: Sequence[RowResult],
    run1_summary: dict[str, Any],
    run2_summary: dict[str, Any],
) -> dict[str, Any]:
    by_row1 = {r.input_row: r for r in rows_run1}
    by_row2 = {r.input_row: r for r in rows_run2}
    common_rows = sorted(set(by_row1) & set(by_row2))

    top1_pairs: list[tuple[str | None, str | None]] = []
    top5_pairs: list[tuple[list[str | None], list[str | None]]] = []
    skipped_error_rows = 0
    for input_row in common_rows:
        r1, r2 = by_row1[input_row], by_row2[input_row]
        if r1.mapped_status == "error" or r2.mapped_status == "error":
            skipped_error_rows += 1
            continue
        top1_pairs.append((r1.mapped_code_normalized, r2.mapped_code_normalized))
        r1_codes = [r1.mapped_code_normalized, *[a.code for a in r1.alternatives]]
        r2_codes = [r2.mapped_code_normalized, *[a.code for a in r2.alternatives]]
        top5_pairs.append((r1_codes, r2_codes))

    weighted_f1_diff = abs(run1_summary["weighted_f1"] - run2_summary["weighted_f1"])
    top1_diff = abs(run1_summary["top1_accuracy"] - run2_summary["top1_accuracy"])
    top5_diff = abs(run1_summary["top5_hit_rate"] - run2_summary["top5_hit_rate"])

    return {
        "model": model,
        "rows_compared": len(top1_pairs),
        "rows_skipped_execution_error": skipped_error_rows,
        "top1_exact_agreement": top1_exact_agreement(top1_pairs),
        "top5_set_agreement": top5_set_agreement(top5_pairs),
        "run_1_weighted_f1": run1_summary["weighted_f1"],
        "run_2_weighted_f1": run2_summary["weighted_f1"],
        "weighted_f1_absolute_difference": weighted_f1_diff,
        "run_1_top1_accuracy": run1_summary["top1_accuracy"],
        "run_2_top1_accuracy": run2_summary["top1_accuracy"],
        "top1_absolute_difference": top1_diff,
        "run_1_top5_hit_rate": run1_summary["top5_hit_rate"],
        "run_2_top5_hit_rate": run2_summary["top5_hit_rate"],
        "top5_absolute_difference": top5_diff,
    }


REPRODUCIBILITY_SUMMARY_FIELDS: tuple[str, ...] = (
    "model",
    "rows_compared",
    "rows_skipped_execution_error",
    "top1_exact_agreement",
    "top5_set_agreement",
    "run_1_weighted_f1",
    "run_2_weighted_f1",
    "weighted_f1_absolute_difference",
    "run_1_top1_accuracy",
    "run_2_top1_accuracy",
    "top1_absolute_difference",
    "run_1_top5_hit_rate",
    "run_2_top5_hit_rate",
    "top5_absolute_difference",
)


def build_model_summary(
    *,
    model: str,
    run1_summary: dict[str, Any],
    run2_summary: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    runs = [run1_summary, run2_summary]
    runs_completed = sum(1 for r in runs if r["run_complete"])

    def _mean(key: str) -> float:
        return statistics.fmean(r[key] for r in runs)

    costs = [r["total_api_cost_usd"] for r in runs if r["total_api_cost_usd"] is not None]
    combined_cost = sum(costs) if costs else None
    total_evaluated = sum(r["rows_evaluated"] for r in runs)
    total_retrieval_retries = sum(r.get("total_retrieval_retries", 0) or 0 for r in runs)
    rows_with_retrieval_retries = sum(r.get("rows_with_retrieval_retries", 0) or 0 for r in runs)
    rows_with_final_retrieval_errors = sum(
        r.get("rows_with_final_retrieval_errors", 0) or 0 for r in runs
    )

    return {
        "model": model,
        "runs_completed": runs_completed,
        "mean_weighted_precision": _mean("weighted_precision"),
        "mean_weighted_recall": _mean("weighted_recall"),
        "mean_weighted_f1": _mean("weighted_f1"),
        "mean_top1_accuracy": _mean("top1_accuracy"),
        "mean_top5_hit_rate": _mean("top5_hit_rate"),
        "top1_exact_agreement": reproducibility["top1_exact_agreement"],
        "top5_set_agreement": reproducibility["top5_set_agreement"],
        "mean_total_run_seconds": _mean("total_run_seconds"),
        "mean_end_to_end_seconds_per_term": _mean("mean_end_to_end_seconds_per_term"),
        "mean_llm_seconds_per_term": _mean("mean_llm_seconds_per_term"),
        "combined_api_cost_usd": combined_cost,
        "mean_api_cost_per_term_usd": (
            combined_cost / total_evaluated if combined_cost is not None and total_evaluated else None
        ),
        "run_1_weighted_f1": reproducibility["run_1_weighted_f1"],
        "run_2_weighted_f1": reproducibility["run_2_weighted_f1"],
        "weighted_f1_absolute_difference": reproducibility["weighted_f1_absolute_difference"],
        "run_1_top1_accuracy": reproducibility["run_1_top1_accuracy"],
        "run_2_top1_accuracy": reproducibility["run_2_top1_accuracy"],
        "top1_absolute_difference": reproducibility["top1_absolute_difference"],
        "run_1_top5_hit_rate": reproducibility["run_1_top5_hit_rate"],
        "run_2_top5_hit_rate": reproducibility["run_2_top5_hit_rate"],
        "top5_absolute_difference": reproducibility["top5_absolute_difference"],
        "total_retrieval_retries": total_retrieval_retries,
        "rows_with_retrieval_retries": rows_with_retrieval_retries,
        "rows_with_final_retrieval_errors": rows_with_final_retrieval_errors,
    }


MODEL_SUMMARY_FIELDS: tuple[str, ...] = (
    "model",
    "runs_completed",
    "mean_weighted_precision",
    "mean_weighted_recall",
    "mean_weighted_f1",
    "mean_top1_accuracy",
    "mean_top5_hit_rate",
    "top1_exact_agreement",
    "top5_set_agreement",
    "mean_total_run_seconds",
    "mean_end_to_end_seconds_per_term",
    "mean_llm_seconds_per_term",
    "combined_api_cost_usd",
    "mean_api_cost_per_term_usd",
    "run_1_weighted_f1",
    "run_2_weighted_f1",
    "weighted_f1_absolute_difference",
    "run_1_top1_accuracy",
    "run_2_top1_accuracy",
    "top1_absolute_difference",
    "run_1_top5_hit_rate",
    "run_2_top5_hit_rate",
    "top5_absolute_difference",
    "total_retrieval_retries",
    "rows_with_retrieval_retries",
    "rows_with_final_retrieval_errors",
)


# ─────────────────────────────────────────────────────────────────────────────
# benchmark_config.json
# ─────────────────────────────────────────────────────────────────────────────


def get_git_commit_hash(repo_dir: Path) -> str | None:
    """Best-effort git commit hash; None (never fabricated) when unavailable."""
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


def build_benchmark_config(
    *,
    input_path: Path,
    input_file_hash: str,
    model: str,
    provider: str,
    runs: int,
    temperature: float | None,
    seed: int,
    reasoning_effort: str | None,
    retrieval_mode: str,
    max_alternatives: int,
    pricing: ModelPricing,
    repo_dir: Path,
    start_timestamp: str,
) -> dict[str, Any]:
    return {
        "input_filename": input_path.name,
        "input_file_sha256": input_file_hash,
        "model": model,
        "provider": provider,
        "runs": runs,
        # temperature is null when omitted from the request (provider
        # default) -- temperature_mode makes that explicit so the config
        # never falsely implies a specific value (e.g. 0.0) was requested.
        "temperature": temperature,
        "temperature_mode": "provider_default" if temperature is None else "explicit",
        "seed": seed,
        "reasoning_effort": reasoning_effort if reasoning_effort is not None else "N/A",
        "retrieval_mode": retrieval_mode,
        "max_alternatives": max_alternatives,
        "hard_target_ontology_behavior": (
            "Each row's target_ontology is supplied as a one-item ontologies=[...] "
            "hard constraint to OntologyMapper (one mapper instance per distinct "
            "target ontology in the dataset, sharing one PlannedPipeline/provider)."
        ),
        "pricing_snapshot": asdict(pricing),
        "git_commit_hash": get_git_commit_hash(repo_dir),
        "start_timestamp": start_timestamp,
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Progress printing
# ─────────────────────────────────────────────────────────────────────────────


def format_progress_line(row: RowResult, *, index: int, total: int) -> str:
    status = row.mapped_status
    gold_rank = row.gold_rank if row.gold_rank is not None else "-"
    return (
        f"[{row.model} run={row.run_number} {index}/{total}] "
        f"{row.source_variable!r} -> {row.target_ontology} "
        f"status={status} gold_rank={gold_rank} "
        f"latency={row.end_to_end_seconds:.2f}s"
    )
