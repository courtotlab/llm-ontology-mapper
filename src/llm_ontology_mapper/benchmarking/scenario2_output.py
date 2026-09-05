"""
Scenario 2 (retrieval-mode ablation) output writers: predictions.csv,
experiment_config.json + resume-fingerprint validation, hallucination
validation (re-runnable with zero mapper/LLM calls), and every per-mode
report file (Part 20/24/25).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from llm_ontology_mapper.benchmarking.scenario1_runner import SapBertHealth
from llm_ontology_mapper.benchmarking.scenario2_calibration import (
    EceResult,
    RocAucResult,
    SeparationStats,
)
from llm_ontology_mapper.benchmarking.scenario2_dataset import Scenario2DatasetAudit
from llm_ontology_mapper.benchmarking.scenario2_metrics import (
    MAX_RANK,
    STATUS_MAPPED,
    AbstentionStats,
    ExecutionDiagnostics,
    PredictionRecord,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario2_runner import AlternativeSlot, Scenario2RowResult
from llm_ontology_mapper.benchmarking.scenario2_validation import (
    NOT_APPLICABLE,
    HallucinationSummary,
    SupportsValidateCode,
    ValidationCache,
    read_validation_cache,
    validate_one,
    write_validation_cache,
)

_PIPE = "|"
_BOOL_TRUE = {"true", "1", "yes"}


def _join(values: Sequence[str | None]) -> str:
    return _PIPE.join("" if v is None else str(v) for v in values)


def _split(value: str) -> list[str]:
    return [] if value == "" else value.split(_PIPE)


def _to_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in _BOOL_TRUE


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Part 20 -- predictions.csv (one row per dataset row)
# ─────────────────────────────────────────────────────────────────────────────

PREDICTIONS_CSV_FIELDS: tuple[str, ...] = (
    "row_id",
    "source_variable",
    "source_label",
    "source_description",
    "target_ontology",
    "gold_codes",
    "gold_terms",
    "retrieval_mode",
    "status",
    "mapped_code",
    "mapped_code_normalized",
    "mapped_term",
    "mapped_ontology",
    "confidence",
    "logic_type",
    "rank_1_code",
    "rank_1_term",
    "rank_2_code",
    "rank_2_term",
    "rank_3_code",
    "rank_3_term",
    "rank_4_code",
    "rank_4_term",
    "rank_5_code",
    "rank_5_term",
    "first_gold_rank",
    "top1_hit",
    "top3_hit",
    "top5_hit",
    "reciprocal_rank",
    "recall_at_gt",
    "semantic_correctness",
    "is_grounded",
    "grounding_source",
    "selected_code_was_retrieved",
    "retrieval_skipped",
    "validation_status",
    "validation_source",
    "execution_error",
    "error_stage",
    "error_type",
    "error_message",
    "end_to_end_seconds",
    "planner_seconds",
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

NOT_VALIDATED = "NOT_VALIDATED"  # placeholder written by the live loop, before finalize() validates


def row_result_to_csv_dict(row: Scenario2RowResult) -> dict[str, Any]:
    """Build the persisted row for one Scenario2RowResult straight off the
    live pipeline, including every PURE (network-free) metric -- gold-based
    Top-k/MRR/Recall@GT/semantic_correctness never depend on an external
    service, so they are computed and written immediately. validation_status
    is deliberately left as NOT_VALIDATED here (Part 18): hallucination
    validation is a separate, re-runnable, network-touching pass performed by
    finalize_mode_outputs(), never inline in the mapping loop."""
    ranked_codes = row.rank_codes
    record = PredictionRecord(
        row_id=row.input_row,
        status=row.mapped_status,
        gold_codes=tuple(row.gold_codes_normalized),
        ranks=ranked_codes,
    )
    metrics = score_prediction(record)

    alts = row.alternatives + [None] * max(0, 4 - len(row.alternatives))  # type: ignore[list-item]
    ranks_out: list[AlternativeSlot | None] = [
        AlternativeSlot(
            code=row.mapped_code_normalized,
            term=row.mapped_term,
            ontology=row.mapped_ontology,
            confidence=row.confidence,
        )
        if row.mapped_status == "mapped"
        else None
    ]
    ranks_out.extend(alts[:4])

    d: dict[str, Any] = {
        "row_id": row.input_row,
        "source_variable": row.source_variable,
        "source_label": row.source_label,
        "source_description": row.source_description,
        "target_ontology": row.target_ontology,
        "gold_codes": _join(row.gold_codes_normalized),
        "gold_terms": row.gold_target_term,
        "retrieval_mode": row.retrieval_mode,
        "status": row.mapped_status,
        "mapped_code": row.mapped_code,
        "mapped_code_normalized": row.mapped_code_normalized,
        "mapped_term": row.mapped_term,
        "mapped_ontology": row.mapped_ontology,
        "confidence": row.confidence,
        "logic_type": row.logic_type,
        "first_gold_rank": metrics.gold_rank,
        "top1_hit": metrics.top1_hit,
        "top3_hit": metrics.top3_hit,
        "top5_hit": metrics.top5_hit,
        "reciprocal_rank": metrics.reciprocal_rank,
        "recall_at_gt": metrics.recall_at_gt,
        "semantic_correctness": metrics.semantic_correctness,
        "is_grounded": row.is_grounded,
        "grounding_source": row.grounding_source,
        "selected_code_was_retrieved": row.selected_code_was_retrieved,
        "retrieval_skipped": row.retrieval_skipped,
        "validation_status": NOT_VALIDATED if row.mapped_status == STATUS_MAPPED else NOT_APPLICABLE,
        "validation_source": None,
        "execution_error": row.mapped_status == "error",
        "error_stage": row.error_stage,
        "error_type": row.error_type,
        "error_message": row.error_message,
        "end_to_end_seconds": row.end_to_end_seconds,
        "planner_seconds": row.query_planner_seconds,
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
        slot = ranks_out[i] if i < len(ranks_out) else None
        d[f"rank_{i + 1}_code"] = slot.code if slot else None
        d[f"rank_{i + 1}_term"] = slot.term if slot else None
    return d


class IncrementalPredictionsCsvWriter:
    """Writes/flushes each row immediately so a crash never loses prior rows.
    Supports append mode for resume."""

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


def rewrite_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PREDICTIONS_CSV_FIELDS})
    os.replace(tmp_path, path)


def csv_row_to_prediction_record(csv_row: dict[str, str]) -> PredictionRecord:
    ranks = tuple((csv_row.get(f"rank_{i + 1}_code") or None) for i in range(MAX_RANK))
    return PredictionRecord(
        row_id=int(csv_row["row_id"]),
        status=csv_row["status"],
        gold_codes=tuple(_split(csv_row.get("gold_codes", ""))),
        ranks=ranks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resume: quarantine prior status="error" rows so --resume retries them
# instead of skipping them forever (same reliability pattern as Scenario 1).
# ─────────────────────────────────────────────────────────────────────────────

RETRY_ERROR_HISTORY_CSV_FIELDS: tuple[str, ...] = (
    "row_id",
    "source_variable",
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
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RETRY_ERROR_HISTORY_CSV_FIELDS))
        if write_header:
            writer.writeheader()
        for row in error_rows:
            writer.writerow(
                {
                    "row_id": row.get("row_id"),
                    "source_variable": row.get("source_variable"),
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
    """Prepare an output directory for --resume: status="error" rows are
    quarantined to retry_error_history.csv and stripped from predictions.csv
    so iter_run_rows() naturally retries them; only mapped/unmapped rows
    count as completed. Returns the set of completed row_ids to skip."""
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
        rewrite_predictions_csv(canonical_rows, predictions_path)

    return {int(r["row_id"]) for r in canonical_rows}


# ─────────────────────────────────────────────────────────────────────────────
# experiment_config.json + resume fingerprint
# ─────────────────────────────────────────────────────────────────────────────

RESUME_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "source_dataset_sha256",
    "dataset_row_count",
    "provider",
    "model",
    "reasoning_effort",
    "temperature_mode",
    "temperature",
    "seed",
    "retrieval_mode",
    "strict_target_ontology",
    "max_alternatives",
    "sapbert_url",
    "llm_ontology_mapper_git_commit",
)

# Fields that MUST be identical across public/local/disabled for --compare to
# be scientifically valid; retrieval_mode (and mode-specific metadata such as
# sapbert_url/sapbert health) is the ONLY thing allowed to differ (Part 26).
COMPARE_MUST_MATCH_FIELDS: tuple[str, ...] = (
    "source_dataset_sha256",
    "dataset_row_count",
    "provider",
    "model",
    "reasoning_effort",
    "temperature_mode",
    "temperature",
    "seed",
    "max_alternatives",
    "strict_target_ontology",
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
    dataset_row_count: int,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    temperature: float | None,
    temperature_mode: str,
    seed: int,
    retrieval_mode: str,
    strict_target_ontology: bool,
    max_alternatives: int,
    sapbert_url: str | None,
    sapbert_health: SapBertHealth | None,
    repo_dir: Path,
    start_timestamp: str,
) -> dict[str, Any]:
    return {
        "experiment_name": "scenario2_retrieval_ablation",
        "source_dataset_path": str(source_dataset_path),
        "source_dataset_sha256": source_dataset_sha256,
        "dataset_row_count": dataset_row_count,
        "llm_ontology_mapper_git_commit": get_git_commit_hash(repo_dir),
        "working_tree_dirty": get_git_dirty(repo_dir),
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort if reasoning_effort is not None else "N/A",
        "temperature": temperature,
        "temperature_mode": temperature_mode,
        "seed": seed,
        "retrieval_mode": retrieval_mode,
        "strict_target_ontology": strict_target_ontology,
        "max_alternatives": max_alternatives,
        "sapbert_url": sapbert_url,
        "sapbert_health_response": sapbert_health.raw_response if sapbert_health else None,
        "sapbert_model": sapbert_health.model if sapbert_health else None,
        "start_timestamp": start_timestamp,
        "end_timestamp": None,
        "completed": False,
        "rows_completed": 0,
        "limit": None,
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
    experiment-defining fields differ from the requested run."""


class CompareConfigMismatchError(RuntimeError):
    """Raised when --compare is given three run directories whose
    experiment-defining fields differ by anything other than retrieval_mode
    (and mode-specific retrieval-service metadata)."""


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


def validate_compare_configs(configs: dict[str, dict[str, Any]]) -> None:
    """configs: {"public": cfg, "local": cfg, "disabled": cfg}. Requires
    every COMPARE_MUST_MATCH_FIELDS value to be identical across all three,
    and each config's own retrieval_mode to equal its dict key (Part 26)."""
    mismatches: list[str] = []
    for field_name in COMPARE_MUST_MATCH_FIELDS:
        values = {mode: cfg.get(field_name) for mode, cfg in configs.items()}
        if len(set(values.values())) > 1:
            mismatches.append(f"{field_name}: {values!r}")
    for mode, cfg in configs.items():
        if cfg.get("retrieval_mode") != mode:
            mismatches.append(
                f"retrieval_mode in the {mode!r} directory's experiment_config.json is "
                f"{cfg.get('retrieval_mode')!r}, expected {mode!r}"
            )
    if mismatches:
        raise CompareConfigMismatchError(
            "Refusing to compare: the following fields are not identical across "
            "public/local/disabled (only retrieval_mode and mode-specific retrieval "
            "metadata may differ):\n  " + "\n  ".join(mismatches)
        )


# ─────────────────────────────────────────────────────────────────────────────
# dataset_validation.json
# ─────────────────────────────────────────────────────────────────────────────


def write_dataset_validation_json(audit: Scenario2DatasetAudit, sha256: str, path: Path) -> None:
    payload = audit.to_dict()
    payload["source_dataset_sha256"] = sha256
    payload["calibration_scientifically_evaluable"] = True
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination validation pass (Part 17/18) -- re-runnable, zero LLM calls
# ─────────────────────────────────────────────────────────────────────────────


def run_validation_pass(
    csv_rows: list[dict[str, str]],
    *,
    validator: SupportsValidateCode,
    cache: ValidationCache,
) -> list[dict[str, str]]:
    """Fill validation_status/validation_source on every row (mapped rows hit
    OntologyValidator, cached by normalized CURIE; non-mapped rows are
    NOT_APPLICABLE and are never validated). Mutates `cache` in place and
    returns updated row dicts; never mutates `csv_rows` in place."""
    updated: list[dict[str, str]] = []
    for row in csv_rows:
        mapped_code = row.get("mapped_code_normalized") or row.get("mapped_code")
        status, source = validate_one(
            status=row.get("status", ""),
            mapped_code=mapped_code,
            validator=validator,
            cache=cache,
        )
        new_row = dict(row)
        new_row["validation_status"] = status
        new_row["validation_source"] = source or ""
        updated.append(new_row)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Part 25 -- per-mode summary + Part 14/15/16 calibration reports
# ─────────────────────────────────────────────────────────────────────────────


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for r in rows:
        v = _to_float(r.get(key))
        if v is not None:
            out.append(v)
    return out


def write_calibration_bins_csv(mode: str, ece: EceResult, path: Path) -> None:
    fields = ("mode", "bin_lower", "bin_upper", "count", "mean_confidence", "empirical_accuracy", "calibration_gap")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for b in ece.bins:
            writer.writerow(
                {
                    "mode": mode,
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "count": b.count,
                    "mean_confidence": b.mean_confidence,
                    "empirical_accuracy": b.empirical_accuracy,
                    "calibration_gap": b.calibration_gap,
                }
            )


def write_calibration_statistics_csv(
    mode: str,
    *,
    calibration_n: int,
    n_correct: int,
    n_incorrect: int,
    auc: RocAucResult,
    brier: float | None,
    ece: EceResult,
    separation: SeparationStats,
    path: Path,
) -> None:
    fields = (
        "mode",
        "calibration_n",
        "n_correct",
        "n_incorrect",
        "roc_auc",
        "roc_auc_status",
        "brier_score",
        "ece",
        "rank_sum_test_name",
        "rank_sum_statistic",
        "rank_sum_p_value",
        "mean_confidence_correct",
        "mean_confidence_incorrect",
        "sd_confidence_correct",
        "sd_confidence_incorrect",
        "cohens_d",
        "separation_status",
        "separation_note",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "mode": mode,
                "calibration_n": calibration_n,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "roc_auc": auc.value,
                "roc_auc_status": auc.status,
                "brier_score": brier,
                "ece": ece.ece,
                "rank_sum_test_name": separation.test_name,
                "rank_sum_statistic": separation.statistic,
                "rank_sum_p_value": separation.p_value,
                "mean_confidence_correct": separation.mean_confidence_correct,
                "mean_confidence_incorrect": separation.mean_confidence_incorrect,
                "sd_confidence_correct": separation.sd_confidence_correct,
                "sd_confidence_incorrect": separation.sd_confidence_incorrect,
                "cohens_d": separation.cohens_d,
                "separation_status": separation.status,
                "separation_note": separation.note,
            }
        )


def write_execution_diagnostics_csv(diag: ExecutionDiagnostics, path: Path) -> None:
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


def write_retrieval_diagnostics_csv(csv_rows: list[dict[str, str]], path: Path) -> None:
    def _sum_int(key: str) -> int:
        values: list[str] = [v for r in csv_rows if (v := r.get(key))]
        return sum(int(v) for v in values)

    total_retries = _sum_int("retrieval_retry_count")
    rows_with_retries = sum(1 for r in csv_rows if (r.get("retrieval_retry_count") or "0") not in ("", "0"))
    rows_with_final_errors = sum(
        1 for r in csv_rows if (r.get("retrieval_final_error_count") or "0") not in ("", "0")
    )
    total_requests = _sum_int("retrieval_request_count")

    fields = (
        "total_retrieval_requests",
        "total_retrieval_retries",
        "rows_with_retrieval_retries",
        "rows_with_final_retrieval_errors",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "total_retrieval_requests": total_requests,
                "total_retrieval_retries": total_retries,
                "rows_with_retrieval_retries": rows_with_retries,
                "rows_with_final_retrieval_errors": rows_with_final_errors,
            }
        )


def write_telemetry_summary_csv(csv_rows: list[dict[str, str]], path: Path) -> None:
    e2e = _floats(csv_rows, "end_to_end_seconds")
    llm = _floats(csv_rows, "llm_seconds")
    costs = _floats(csv_rows, "api_cost_usd")

    def _median(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    fields = (
        "mean_end_to_end_seconds",
        "median_end_to_end_seconds",
        "mean_llm_seconds",
        "median_llm_seconds",
        "total_api_cost_usd",
        "mean_api_cost_per_row_usd",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "mean_end_to_end_seconds": statistics.fmean(e2e) if e2e else None,
                "median_end_to_end_seconds": _median(e2e),
                "mean_llm_seconds": statistics.fmean(llm) if llm else None,
                "median_llm_seconds": _median(llm),
                "total_api_cost_usd": sum(costs) if costs else None,
                "mean_api_cost_per_row_usd": statistics.fmean(costs) if costs else None,
            }
        )


def write_mode_summary(
    mode: str,
    *,
    agg: Any,  # scenario2_metrics.AggregateMetrics
    abstention: AbstentionStats,
    grounding_rate: float | None,
    hallucination: HallucinationSummary,
    auc: RocAucResult,
    brier: float | None,
    ece: EceResult,
    separation: SeparationStats,
    execution: ExecutionDiagnostics,
    csv_rows: list[dict[str, str]],
    path_csv: Path,
    path_md: Path,
) -> None:
    e2e = _floats(csv_rows, "end_to_end_seconds")
    llm = _floats(csv_rows, "llm_seconds")
    costs = _floats(csv_rows, "api_cost_usd")
    rows: list[tuple[str, Any]] = [
        ("mode", mode),
        ("n", agg.n),
        ("top1_accuracy", agg.top1),
        ("top3_accuracy", agg.top3),
        ("top5_accuracy", agg.top5),
        ("mrr", agg.mrr),
        ("recall_at_gt", agg.recall_at_gt),
        ("recall_at_gt_n", agg.recall_at_gt_n),
        ("abstention_count", abstention.abstention_count),
        ("abstention_total", abstention.total),
        ("abstention_rate", abstention.abstention_rate),
        ("grounding_rate", grounding_rate),
        ("hallucination_rate", hallucination.hallucination_rate),
        ("validation_coverage", hallucination.validation_coverage),
        ("unresolved_validation_count", hallucination.unresolved_validation_count),
        ("unresolved_validation_rate", hallucination.unresolved_validation_rate),
        ("valid_code_count", hallucination.valid_count),
        ("invalid_code_count", hallucination.invalid_count),
        ("roc_auc", auc.value),
        ("roc_auc_status", auc.status),
        ("brier_score", brier),
        ("ece", ece.ece),
        ("rank_sum_test_name", separation.test_name),
        ("rank_sum_statistic", separation.statistic),
        ("rank_sum_p_value", separation.p_value),
        ("cohens_d", separation.cohens_d),
        ("execution_error_count", execution.error_count),
        ("execution_error_rate", execution.error_rate),
        ("mapped_count", execution.mapped_count),
        ("unmapped_count", execution.unmapped_count),
        ("mean_end_to_end_seconds", statistics.fmean(e2e) if e2e else None),
        ("median_end_to_end_seconds", statistics.median(e2e) if e2e else None),
        ("mean_llm_seconds", statistics.fmean(llm) if llm else None),
        ("median_llm_seconds", statistics.median(llm) if llm else None),
        ("total_api_cost_usd", sum(costs) if costs else None),
        ("mean_api_cost_per_row_usd", statistics.fmean(costs) if costs else None),
    ]

    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("metric", "value"))
        writer.writeheader()
        for key, value in rows:
            writer.writerow({"metric": key, "value": value})

    lines = [f"# Scenario 2 -- retrieval-mode ablation -- mode={mode}", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in rows:
        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {value_str} |")
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "COMPARE_MUST_MATCH_FIELDS",
    "NOT_VALIDATED",
    "PREDICTIONS_CSV_FIELDS",
    "RESUME_FINGERPRINT_FIELDS",
    "CompareConfigMismatchError",
    "IncrementalPredictionsCsvWriter",
    "ResumeConfigMismatchError",
    "build_experiment_config",
    "csv_row_to_prediction_record",
    "get_git_commit_hash",
    "get_git_dirty",
    "load_experiment_config",
    "quarantine_error_rows_for_resume",
    "read_existing_predictions",
    "read_validation_cache",
    "rewrite_predictions_csv",
    "row_result_to_csv_dict",
    "run_validation_pass",
    "validate_compare_configs",
    "validate_resume",
    "write_calibration_bins_csv",
    "write_calibration_statistics_csv",
    "write_dataset_validation_json",
    "write_execution_diagnostics_csv",
    "write_experiment_config",
    "write_mode_summary",
    "write_retrieval_diagnostics_csv",
    "write_telemetry_summary_csv",
    "write_validation_cache",
]
