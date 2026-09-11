"""
Scenario 2 (retrieval-mode ablation) cross-mode comparison (Part 26/27).

Reads three ALREADY-COMPLETED run directories (public/local/disabled) and
produces paired_predictions.csv + scenario2_comparison.csv/.md. Makes ZERO
mapping/LLM calls -- every input here is a file already on disk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_ontology_mapper.benchmarking.scenario2_output import (
    CompareConfigMismatchError,
    _to_bool,
    _to_float,
    load_experiment_config,
    read_existing_predictions,
    validate_compare_configs,
)

MODES: tuple[str, ...] = ("public", "local", "disabled")


class CompareDatasetMismatchError(RuntimeError):
    """Raised when the three run directories do not share identical dataset
    content (row IDs, source fields, golds, target ontologies)."""


@dataclass(frozen=True)
class LoadedRun:
    mode: str
    output_dir: Path
    config: dict[str, Any]
    rows_by_id: dict[int, dict[str, str]]


def load_run(mode: str, output_dir: Path) -> LoadedRun:
    config_path = output_dir / "experiment_config.json"
    config = load_experiment_config(config_path)
    if config is None:
        raise CompareDatasetMismatchError(f"No experiment_config.json found in {output_dir}")
    rows = read_existing_predictions(output_dir / "predictions.csv")
    if not rows:
        raise CompareDatasetMismatchError(f"No predictions.csv rows found in {output_dir}")
    rows_by_id = {int(r["row_id"]): r for r in rows}
    return LoadedRun(mode=mode, output_dir=output_dir, config=config, rows_by_id=rows_by_id)


def validate_dataset_compatibility(runs: dict[str, LoadedRun]) -> None:
    """Requires identical dataset SHA/N (via experiment_config, checked by
    validate_compare_configs) AND identical row IDs, source fields, golds,
    and target ontologies -- checked directly against each run's own
    predictions.csv content, never assumed from a matching SHA alone."""
    reference_mode = MODES[0]
    reference = runs[reference_mode]
    reference_ids = set(reference.rows_by_id)

    for mode in MODES[1:]:
        other = runs[mode]
        other_ids = set(other.rows_by_id)
        if other_ids != reference_ids:
            raise CompareDatasetMismatchError(
                f"row_id sets differ between {reference_mode!r} and {mode!r} directories: "
                f"only-in-{reference_mode}={sorted(reference_ids - other_ids)[:10]} "
                f"only-in-{mode}={sorted(other_ids - reference_ids)[:10]}"
            )

    for row_id in sorted(reference_ids):
        ref_row = reference.rows_by_id[row_id]
        for mode in MODES[1:]:
            other_row = runs[mode].rows_by_id[row_id]
            for field_name in ("source_variable", "source_label", "source_description", "target_ontology", "gold_codes"):
                if ref_row.get(field_name, "") != other_row.get(field_name, ""):
                    raise CompareDatasetMismatchError(
                        f"row_id={row_id} field {field_name!r} differs between "
                        f"{reference_mode!r} ({ref_row.get(field_name)!r}) and "
                        f"{mode!r} ({other_row.get(field_name)!r})"
                    )


def load_and_validate_runs(
    *, public_dir: Path, local_dir: Path, disabled_dir: Path
) -> dict[str, LoadedRun]:
    runs = {
        "public": load_run("public", public_dir),
        "local": load_run("local", local_dir),
        "disabled": load_run("disabled", disabled_dir),
    }
    validate_compare_configs({mode: run.config for mode, run in runs.items()})
    validate_dataset_compatibility(runs)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# paired_predictions.csv (Part 26)
# ─────────────────────────────────────────────────────────────────────────────

PAIRED_PREDICTIONS_CSV_FIELDS: tuple[str, ...] = (
    "row_id",
    "source_variable",
    "target_ontology",
    "gold_codes",
    "public_code",
    "public_correct",
    "public_confidence",
    "public_status",
    "public_grounded",
    "public_valid",
    "local_code",
    "local_correct",
    "local_confidence",
    "local_status",
    "local_grounded",
    "local_valid",
    "disabled_code",
    "disabled_correct",
    "disabled_confidence",
    "disabled_status",
    "disabled_grounded",
    "disabled_valid",
    "public_local_same_code",
    "public_disabled_same_code",
    "local_disabled_same_code",
)


def _mode_fields(row: dict[str, str]) -> dict[str, Any]:
    return {
        "code": row.get("mapped_code_normalized") or None,
        "correct": _to_bool(row.get("semantic_correctness")),
        "confidence": _to_float(row.get("confidence")),
        "status": row.get("status"),
        "grounded": _to_bool(row.get("is_grounded")),
        "valid": row.get("validation_status") == "VALID",
    }


def build_paired_predictions(runs: dict[str, LoadedRun]) -> list[dict[str, Any]]:
    reference = runs["public"]
    paired: list[dict[str, Any]] = []
    for row_id in sorted(reference.rows_by_id):
        base = reference.rows_by_id[row_id]
        per_mode = {mode: _mode_fields(runs[mode].rows_by_id[row_id]) for mode in MODES}
        row: dict[str, Any] = {
            "row_id": row_id,
            "source_variable": base.get("source_variable"),
            "target_ontology": base.get("target_ontology"),
            "gold_codes": base.get("gold_codes"),
        }
        for mode in MODES:
            fields = per_mode[mode]
            row[f"{mode}_code"] = fields["code"]
            row[f"{mode}_correct"] = fields["correct"]
            row[f"{mode}_confidence"] = fields["confidence"]
            row[f"{mode}_status"] = fields["status"]
            row[f"{mode}_grounded"] = fields["grounded"]
            row[f"{mode}_valid"] = fields["valid"]
        row["public_local_same_code"] = per_mode["public"]["code"] == per_mode["local"]["code"]
        row["public_disabled_same_code"] = per_mode["public"]["code"] == per_mode["disabled"]["code"]
        row["local_disabled_same_code"] = per_mode["local"]["code"] == per_mode["disabled"]["code"]
        paired.append(row)
    return paired


def write_paired_predictions_csv(paired: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PAIRED_PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in paired:
            writer.writerow({field: row.get(field, "") for field in PAIRED_PREDICTIONS_CSV_FIELDS})


# ─────────────────────────────────────────────────────────────────────────────
# Paired exact-correctness transition counts (Part 26, optional diagnostic)
# ─────────────────────────────────────────────────────────────────────────────


def transition_counts(paired: list[dict[str, Any]]) -> dict[str, int]:
    def _count(a_mode: str, b_mode: str, *, a_correct: bool, b_correct: bool) -> int:
        return sum(
            1
            for row in paired
            if row[f"{a_mode}_correct"] is a_correct and row[f"{b_mode}_correct"] is b_correct
        )

    return {
        "correct_in_public_wrong_in_local": _count("public", "local", a_correct=True, b_correct=False),
        "correct_in_local_wrong_in_public": _count("local", "public", a_correct=True, b_correct=False),
        "correct_in_public_wrong_in_disabled": _count("public", "disabled", a_correct=True, b_correct=False),
        "correct_in_disabled_wrong_in_public": _count("disabled", "public", a_correct=True, b_correct=False),
        "correct_in_local_wrong_in_disabled": _count("local", "disabled", a_correct=True, b_correct=False),
        "correct_in_disabled_wrong_in_local": _count("disabled", "local", a_correct=True, b_correct=False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# scenario2_comparison.csv / .md (Part 27)
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_METRICS: tuple[str, ...] = (
    "top1_accuracy",
    "top3_accuracy",
    "top5_accuracy",
    "mrr",
    "recall_at_gt",
    "abstention_rate",
    "hallucination_rate",
    "validation_coverage",
    "grounding_rate",
    "roc_auc",
    "brier_score",
    "ece",
    "cohens_d",
    "execution_error_rate",
    "mean_end_to_end_seconds",
    "mean_llm_seconds",
    "mean_api_cost_per_row_usd",
    "total_api_cost_usd",
)

_METRIC_LABELS: dict[str, str] = {
    "top1_accuracy": "Top-1",
    "top3_accuracy": "Top-3",
    "top5_accuracy": "Top-5",
    "mrr": "MRR",
    "recall_at_gt": "Recall@GT",
    "abstention_rate": "Abstention",
    "hallucination_rate": "Hallucination",
    "validation_coverage": "Validation coverage",
    "grounding_rate": "Grounding",
    "roc_auc": "AUC",
    "brier_score": "Brier",
    "ece": "ECE",
    "cohens_d": "Cohen's d",
    "execution_error_rate": "Execution error rate",
    "mean_end_to_end_seconds": "Mean E2E latency",
    "mean_llm_seconds": "Mean LLM latency",
    "mean_api_cost_per_row_usd": "Cost / row",
    "total_api_cost_usd": "Total cost",
}


def read_mode_summary_values(mode_summary_csv: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with mode_summary_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            values[row["metric"]] = row["value"]
    return values


def build_comparison_table(mode_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in COMPARISON_METRICS:
        row: dict[str, Any] = {"metric": _METRIC_LABELS.get(metric, metric)}
        for mode in MODES:
            row[mode] = mode_summaries[mode].get(metric)
        rows.append(row)
    return rows


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ("metric", "public", "local", "disabled")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_comparison_md(
    rows: list[dict[str, Any]],
    transitions: dict[str, int],
    path: Path,
) -> None:
    lines = [
        "# Scenario 2 -- retrieval-mode ablation -- cross-mode comparison",
        "",
        "| Metric | Public | Local | Disabled |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:

        def _fmt(v: Any) -> str:
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v)

        lines.append(f"| {row['metric']} | {_fmt(row['public'])} | {_fmt(row['local'])} | {_fmt(row['disabled'])} |")

    lines += ["", "## Paired exact-correctness transitions", "", "| Transition | Count |", "| --- | --- |"]
    for key, value in transitions.items():
        lines.append(f"| {key} | {value} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "MODES",
    "PAIRED_PREDICTIONS_CSV_FIELDS",
    "CompareConfigMismatchError",
    "CompareDatasetMismatchError",
    "LoadedRun",
    "build_comparison_table",
    "build_paired_predictions",
    "load_and_validate_runs",
    "load_run",
    "read_mode_summary_values",
    "transition_counts",
    "validate_compare_configs",
    "validate_dataset_compatibility",
    "write_comparison_csv",
    "write_comparison_md",
    "write_paired_predictions_csv",
]
