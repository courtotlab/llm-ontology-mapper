"""
Scenario 1 EFO targeted UNMAPPED rerun-and-patch workflow.

Reruns ONLY the rows that were `status == "unmapped"` in one of the three
already-completed Scenario 1 EFO benchmark runs (OLS-EFO, Biomappings-EFO,
UKBB-EFO), under the pipeline's current UNMAPPED-retains-alternatives
behavior, then patches those rows into a *copy* of the original full
predictions.csv -- the original run directories are never opened for
writing.

This module intentionally reuses the existing Scenario 1 building blocks
(scenario1_dataset, scenario1_runner, scenario1_output, scenario1_metrics,
scenario1_graph_distance) rather than reimplementing dataset loading, mapper
construction, per-row execution, checkpointing, or scoring. The only new
logic here is: selecting the originally-unmapped query_ids, restricting
execution to exactly that set, and patching them into a full-size copy.

Column-group contract (see PREDICTIONS_CSV_FIELDS in scenario1_output.py)
───────────────────────────────────────────────────────────────────────────
    IMMUTABLE_FIELDS        -- source/gold columns, never touched
    MAPPER_OUTPUT_FIELDS    -- replaced wholesale for targeted query_ids
    DERIVED_SCORING_FIELDS  -- recomputed for EVERY row from the (possibly
                               patched) mapper-output fields, never copied
                               verbatim from the rerun subset

Configuration pin
──────────────────
The original three runs used max_results_per_query=10, max_candidates=10,
max_alternatives=4 (confirmed against the committed Scenario1RunConfig at
each run's recorded commit -- see the read-only audit). Scenario1RunConfig's
own field defaults have since drifted to 15/20 in the working tree.
build_pinned_run_config() below pins the rerun to 10/10/4 explicitly,
independent of whatever Scenario1RunConfig's own defaults currently are --
Scenario1RunConfig itself is never edited by this module.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_ontology_mapper.benchmarking.dataset import file_sha256
from llm_ontology_mapper.benchmarking.model_registry import BenchmarkModelConfig, get_model_config
from llm_ontology_mapper.benchmarking.pricing import ModelPricing, get_pricing
from llm_ontology_mapper.benchmarking.scenario1_dataset import (
    CanonicalQuery,
    audit_dataset,
    build_canonical_queries,
    load_raw_dataset,
)
from llm_ontology_mapper.benchmarking.scenario1_graph_distance import (
    EfoGraphIndex,
    GraphDistanceResult,
    get_graph_index,
)
from llm_ontology_mapper.benchmarking.scenario1_metrics import (
    MAX_RANK,
    PredictionRecord,
    RowMetrics,
    first_gold_rank,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario1_output import (
    PREDICTIONS_CSV_FIELDS,
    IncrementalPredictionsCsvWriter,
    ResumeConfigMismatchError,
    build_experiment_config,
    load_experiment_config,
    quarantine_error_rows_for_resume,
    read_existing_predictions,
    row_to_csv_dict,
    validate_resume,
    write_experiment_config,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    ERROR_STAGE_LOCAL_RETRIEVAL,
    RETRIEVAL_MODE,
    STRICT_TARGET_ONTOLOGY,
    TARGET_ONTOLOGY,
    Scenario1RowResult,
    Scenario1RunConfig,
    build_mapper,
    build_provider,
    check_sapbert_health,
    iter_predictions,
    run_preflight,
)

REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SAPBERT_URL = "http://localhost:8765"
DEFAULT_PUBLISHED_BASELINES = REPO_DIR / "published_baselines.csv"
DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS = 3
_PIPE = "|"


class Scenario1PatchError(RuntimeError):
    """Raised for any targeted-rerun/patch workflow validation failure."""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry -- the three canonical runs this task applies to, and only
# these. Not auto-discovered: hardcoded to the exact paths confirmed by the
# read-only audit, so a future/unrelated run directory under the same
# scenario folder can never be silently picked up instead.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario1DatasetSpec:
    key: str
    label: str
    original_run_dir: Path
    dataset_path: Path
    rerun_output_root: Path
    patched_output_root: Path
    stability_output_root: Path
    gold_corrected_output_root: Path


DATASET_SPECS: dict[str, Scenario1DatasetSpec] = {
    "ols-efo": Scenario1DatasetSpec(
        key="ols-efo",
        label="OLS-EFO",
        original_run_dir=REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo" / "2026-08-26T15-04-18Z",
        dataset_path=REPO_DIR / "OLS-EFO_full.csv",
        rerun_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo_rerun_unmapped",
        patched_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo_patched",
        stability_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo_stability_sample",
        gold_corrected_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo_gold_corrected",
    ),
    "biomappings-efo": Scenario1DatasetSpec(
        key="biomappings-efo",
        label="Biomappings-EFO",
        original_run_dir=(
            REPO_DIR / "outputs" / "evaluation" / "scenario1_biomappings_efo" / "2026-08-31T16-10-24Z"
        ),
        dataset_path=REPO_DIR / "Biomappings-EFO.csv",
        rerun_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_biomappings_efo_rerun_unmapped",
        patched_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_biomappings_efo_patched",
        stability_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_biomappings_efo_stability_sample",
        gold_corrected_output_root=(
            REPO_DIR / "outputs" / "evaluation" / "scenario1_biomappings_efo_gold_corrected"
        ),
    ),
    "ukbb-efo": Scenario1DatasetSpec(
        key="ukbb-efo",
        label="UKBB-EFO",
        original_run_dir=REPO_DIR / "outputs" / "evaluation" / "scenario1_ukbb_efo" / "2026-08-31T13-54-53Z",
        dataset_path=REPO_DIR / "UKBB-EFO.csv",
        rerun_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ukbb_efo_rerun_unmapped",
        patched_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ukbb_efo_patched",
        stability_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ukbb_efo_stability_sample",
        gold_corrected_output_root=REPO_DIR / "outputs" / "evaluation" / "scenario1_ukbb_efo_gold_corrected",
    ),
}

# Audit-confirmed originally-unmapped counts, used as a fail-loud guard in
# extract_unmapped_subset() -- see allow_count_mismatch to override.
EXPECTED_UNMAPPED_COUNTS: dict[str, int] = {"ols-efo": 115, "biomappings-efo": 14, "ukbb-efo": 4}
EXPECTED_TOTAL_ROW_COUNTS: dict[str, int] = {"ols-efo": 7377, "biomappings-efo": 795, "ukbb-efo": 888}

# Audit-recommended mapped-row stability sample sizes (see item 15 of the audit).
STABILITY_SAMPLE_SIZES: dict[str, int] = {"ols-efo": 120, "biomappings-efo": 20, "ukbb-efo": 10}
STABILITY_SAMPLE_SEED = 20260904


def dataset_keys(value: str) -> list[str]:
    if value == "all":
        return list(DATASET_SPECS)
    if value not in DATASET_SPECS:
        raise Scenario1PatchError(f"Unknown dataset key {value!r}. Valid keys: {sorted(DATASET_SPECS)} or 'all'.")
    return [value]


# ─────────────────────────────────────────────────────────────────────────────
# Column-group contract, derived from the actual predictions.csv schema
# (PREDICTIONS_CSV_FIELDS) rather than a separately hand-maintained list --
# any field added to PREDICTIONS_CSV_FIELDS in the future without being
# classified into one of the two explicit groups below trips the assertion
# at import time instead of silently landing in the wrong group.
# ─────────────────────────────────────────────────────────────────────────────

IMMUTABLE_FIELDS: tuple[str, ...] = ("query_id", "query", "gold_codes", "gold_labels", "gold_count")

DERIVED_SCORING_FIELDS: tuple[str, ...] = (
    "first_gold_rank",
    "top1_hit",
    "top3_hit",
    "top5_hit",
    "reciprocal_rank",
    "recall_at_gt",
    "graph_relationship",
    "graph_matched_gold_code",
)

MAPPER_OUTPUT_FIELDS: tuple[str, ...] = tuple(
    f for f in PREDICTIONS_CSV_FIELDS if f not in IMMUTABLE_FIELDS and f not in DERIVED_SCORING_FIELDS
)

_covered = set(IMMUTABLE_FIELDS) | set(DERIVED_SCORING_FIELDS) | set(MAPPER_OUTPUT_FIELDS)
assert _covered == set(PREDICTIONS_CSV_FIELDS), (
    "scenario1_patch column-group partition does not cover every "
    f"PREDICTIONS_CSV_FIELDS column: missing={set(PREDICTIONS_CSV_FIELDS) - _covered}"
)
assert len(_covered) == len(IMMUTABLE_FIELDS) + len(DERIVED_SCORING_FIELDS) + len(MAPPER_OUTPUT_FIELDS), (
    "scenario1_patch column groups overlap -- a field is classified into more than one group"
)

UNMAPPED_SUBSET_CSV_FIELDS: tuple[str, ...] = IMMUTABLE_FIELDS


def _split_pipe(value: str) -> tuple[str, ...]:
    """Mirror scenario1_output._split()'s whitespace-stripping contract --
    both read the same pipe-joined `gold_codes` cell, so both must tolerate
    a pre-fix compound cell like "EFO:0009679 | EFO:0009684" without leaving
    stray leading/trailing spaces on the split tokens."""
    if not value:
        return ()
    return tuple(p.strip() for p in value.split(_PIPE) if p.strip())


def _join_pipe(values: list[Any]) -> str:
    return _PIPE.join("" if v is None else str(v) for v in values)


# Gold metadata is the subset of IMMUTABLE_FIELDS that the gold-parsing-bug
# correction is allowed to rebuild (query_id/query never change either way).
GOLD_METADATA_FIELDS: tuple[str, ...] = ("gold_codes", "gold_labels", "gold_count")


# ─────────────────────────────────────────────────────────────────────────────
# Pinned original benchmark configuration (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

PINNED_PROVIDER = "openai"
PINNED_MODEL = "gpt-5.6-luna"
PINNED_REASONING_EFFORT = "low"
PINNED_TEMPERATURE: float | None = None
PINNED_SEED = 42
PINNED_MAX_RESULTS_PER_QUERY = 10
PINNED_MAX_CANDIDATES = 10
PINNED_MAX_ALTERNATIVES = 4


def build_pinned_run_config(
    *,
    model_config: BenchmarkModelConfig,
    sapbert_url: str,
    temperature: float | None = PINNED_TEMPERATURE,
    seed: int = PINNED_SEED,
) -> Scenario1RunConfig:
    """Explicitly pin the targeted rerun to the ORIGINAL Scenario 1 benchmark
    settings (max_results_per_query=10, max_candidates=10, max_alternatives=4)
    -- independent of whatever Scenario1RunConfig's own dataclass field
    defaults currently are (they have drifted to 15/20 in the working tree).
    Never achieve this by editing Scenario1RunConfig's defaults; every value
    that matters is passed explicitly here."""
    return Scenario1RunConfig(
        model_config=model_config,
        sapbert_url=sapbert_url,
        temperature=temperature,
        seed=seed,
        max_alternatives=PINNED_MAX_ALTERNATIVES,
        max_results_per_query=PINNED_MAX_RESULTS_PER_QUERY,
        max_candidates=PINNED_MAX_CANDIDATES,
    )


def pinned_model_config() -> BenchmarkModelConfig:
    model_config = get_model_config(PINNED_MODEL)
    if PINNED_REASONING_EFFORT is not None and model_config.reasoning_effort != PINNED_REASONING_EFFORT:
        model_config = dataclasses.replace(model_config, reasoning_effort=PINNED_REASONING_EFFORT)
    return model_config


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 -- extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionResult:
    dataset_key: str
    original_run_dir: Path
    subset_path: Path
    subset_count: int
    query_ids: tuple[int, ...]


def select_unmapped_rows(original_run_dir: Path) -> list[dict[str, str]]:
    """Select status=="unmapped" rows straight from the original
    predictions.csv -- the ONLY source of truth for the rerun set (never
    recomputed dynamically from a later/different result)."""
    predictions_path = original_run_dir / "predictions.csv"
    rows = read_existing_predictions(predictions_path)
    if not rows:
        raise Scenario1PatchError(f"No predictions found at {predictions_path}")

    unmapped = [r for r in rows if r.get("status") == "unmapped"]

    seen_ids: set[int] = set()
    for row in unmapped:
        raw_id = row.get("query_id")
        if raw_id is None or raw_id == "":
            raise Scenario1PatchError(f"Row with empty/missing query_id found among unmapped rows in {predictions_path}")
        qid = int(raw_id)
        if qid in seen_ids:
            raise Scenario1PatchError(f"Duplicate query_id={qid} found among unmapped rows in {predictions_path}")
        seen_ids.add(qid)

    return unmapped


def extract_unmapped_subset(
    spec: Scenario1DatasetSpec,
    *,
    expected_count: int | None = None,
    allow_count_mismatch: bool = False,
) -> ExtractionResult:
    """Phase 1: write <original_run_dir>/unmapped_subset.csv. Never opens
    predictions.csv for writing -- only reads it."""
    unmapped_rows = select_unmapped_rows(spec.original_run_dir)
    count = len(unmapped_rows)

    if expected_count is not None and count != expected_count and not allow_count_mismatch:
        raise Scenario1PatchError(
            f"{spec.label}: expected {expected_count} originally-unmapped row(s) (per the completed "
            f"read-only audit) but found {count} in {spec.original_run_dir / 'predictions.csv'}. "
            "The original run directory may have changed since the audit -- refusing to proceed. "
            "Pass allow_count_mismatch=True (CLI: --allow-count-mismatch) only if this is expected."
        )

    subset_path = spec.original_run_dir / "unmapped_subset.csv"
    with subset_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(UNMAPPED_SUBSET_CSV_FIELDS))
        writer.writeheader()
        for row in unmapped_rows:
            writer.writerow({f: row.get(f, "") for f in UNMAPPED_SUBSET_CSV_FIELDS})

    query_ids = tuple(sorted(int(r["query_id"]) for r in unmapped_rows))
    return ExtractionResult(
        dataset_key=spec.key,
        original_run_dir=spec.original_run_dir,
        subset_path=subset_path,
        subset_count=count,
        query_ids=query_ids,
    )


def read_subset_query_ids(subset_path: Path) -> set[int]:
    if not subset_path.exists():
        raise Scenario1PatchError(f"{subset_path} not found -- run the extraction phase first.")
    with subset_path.open(newline="", encoding="utf-8") as fh:
        return {int(row["query_id"]) for row in csv.DictReader(fh)}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 -- targeted rerun (reuses scenario1_runner/scenario1_output as-is)
# ─────────────────────────────────────────────────────────────────────────────


def load_canonical_queries(spec: Scenario1DatasetSpec) -> list[CanonicalQuery]:
    df = load_raw_dataset(spec.dataset_path)
    return build_canonical_queries(df)


@dataclass(frozen=True)
class RerunOutcome:
    output_dir: Path
    targeted_query_ids: frozenset[int]
    total_targeted: int
    rows_completed_this_call: int
    rows_completed_total: int
    completed: bool
    stop_reason: str | None


def execute_targeted_rerun(
    *,
    label: str,
    mapper: Any,
    canonical_queries: list[CanonicalQuery],
    targeted_query_ids: set[int],
    pricing: ModelPricing | None,
    graph_index: EfoGraphIndex,
    output_dir: Path,
    append: bool,
    already_completed_query_ids: set[int] = frozenset(),
    max_consecutive_local_retrieval_errors: int = DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS,
) -> RerunOutcome:
    """The injectable, network-free loop: given an already-built `mapper`,
    execute exactly `targeted_query_ids` (minus whatever's already completed
    on --resume) and nothing else. Reuses iter_predictions/score_prediction/
    row_to_csv_dict/IncrementalPredictionsCsvWriter verbatim -- see
    run_rerun_for_query_ids() for the real-network wrapper that builds
    `mapper` and calls this."""
    all_query_ids = {cq.query_id for cq in canonical_queries}
    unknown = targeted_query_ids - all_query_ids
    if unknown:
        raise Scenario1PatchError(
            f"{label}: targeted query_id(s) not present in the current dataset file: {sorted(unknown)}"
        )

    skip_query_ids = (all_query_ids - targeted_query_ids) | set(already_completed_query_ids)
    rows_completed_this_call = 0
    consecutive_local_errors = 0
    stop_reason: str | None = None

    with IncrementalPredictionsCsvWriter(output_dir / "predictions.csv", append=append) as writer:
        for row in iter_predictions(
            mapper=mapper, canonical_queries=canonical_queries, pricing=pricing, skip_query_ids=skip_query_ids
        ):
            if row.query_id not in targeted_query_ids:
                # Must be unreachable given skip_query_ids above -- fail loudly
                # rather than silently writing a row outside the targeted set.
                raise Scenario1PatchError(
                    f"{label}: iter_predictions yielded query_id={row.query_id}, which is outside the "
                    "originally-unmapped targeted set. Refusing to write it."
                )

            row_metrics = score_prediction(
                PredictionRecord(
                    query_id=row.query_id,
                    query=row.query,
                    gold_codes=tuple(row.gold_codes),
                    status=row.status,
                    ranks=row.rank_codes,
                    rank_ontologies=row.rank_ontologies,
                )
            )
            graph = graph_index.classify(row.rank_codes[0], row.gold_codes) if row.status == "mapped" else None
            writer.write_row(row_to_csv_dict(row, row_metrics=row_metrics, graph=graph))
            rows_completed_this_call += 1

            # Same consecutive-local-SapBERT-failure guard family as the main
            # benchmark runner's loop (without the immediate re-check-on-
            # first-failure refinement -- see module docstring/limitations).
            if row.status in ("mapped", "unmapped"):
                consecutive_local_errors = 0
            elif row.status == "error" and row.error_stage == ERROR_STAGE_LOCAL_RETRIEVAL:
                consecutive_local_errors += 1
                if consecutive_local_errors >= max_consecutive_local_retrieval_errors:
                    stop_reason = "consecutive_local_retrieval_errors"
                    break

    rows_completed_total = len(already_completed_query_ids) + rows_completed_this_call
    return RerunOutcome(
        output_dir=output_dir,
        targeted_query_ids=frozenset(targeted_query_ids),
        total_targeted=len(targeted_query_ids),
        rows_completed_this_call=rows_completed_this_call,
        rows_completed_total=rows_completed_total,
        completed=stop_reason is None and rows_completed_total == len(targeted_query_ids),
        stop_reason=stop_reason,
    )


def _build_rerun_experiment_config(
    *,
    spec: Scenario1DatasetSpec,
    model_config: BenchmarkModelConfig,
    sapbert_url: str,
    temperature: float | None,
    seed: int,
    sapbert_health: Any,
    canonical_queries: list[CanonicalQuery],
    targeted_query_ids: set[int],
    source_label: str,
    original_status_filter: str,
    source_dir: Path,
    start_timestamp: str,
) -> dict[str, Any]:
    sha256 = file_sha256(spec.dataset_path)
    df = load_raw_dataset(spec.dataset_path)
    audit = audit_dataset(df)

    config = build_experiment_config(
        source_dataset_path=spec.dataset_path,
        source_dataset_sha256=sha256,
        raw_row_count=audit.raw_row_count,
        unique_mapping_pair_count=audit.unique_mapping_pair_count,
        unique_query_count=len(canonical_queries),
        provider=PINNED_PROVIDER,
        model=model_config.model,
        reasoning_effort=model_config.reasoning_effort,
        temperature=temperature,
        temperature_mode="provider_default" if temperature is None else "explicit",
        seed=seed,
        target_ontology=TARGET_ONTOLOGY,
        retrieval_mode=RETRIEVAL_MODE,
        strict_target_ontology=STRICT_TARGET_ONTOLOGY,
        max_alternatives=PINNED_MAX_ALTERNATIVES,
        sapbert_url=sapbert_url,
        sapbert_health=sapbert_health,
        repo_dir=REPO_DIR,
        start_timestamp=start_timestamp,
    )
    # Explicit fix for the audit-confirmed experiment_config.json omission:
    # the original three runs never recorded max_results_per_query/
    # max_candidates at all. This rerun's config records them.
    config["targeted_rerun"] = True
    config["source_run"] = str(source_dir)
    config["source_run_label"] = source_label
    config["original_status_filter"] = original_status_filter
    config["max_results_per_query"] = PINNED_MAX_RESULTS_PER_QUERY
    config["max_candidates"] = PINNED_MAX_CANDIDATES
    config["targeted_query_id_count"] = len(targeted_query_ids)
    config["targeted_query_ids"] = sorted(targeted_query_ids)
    return config


def run_rerun_for_query_ids(
    spec: Scenario1DatasetSpec,
    targeted_query_ids: set[int],
    *,
    output_root: Path,
    source_dir: Path,
    source_label: str,
    original_status_filter: str,
    sapbert_url: str = DEFAULT_SAPBERT_URL,
    resume_dir: Path | None = None,
    max_consecutive_local_retrieval_errors: int = DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS,
) -> RerunOutcome:
    """Real-network wrapper: builds provider/mapper/pricing/graph-index for
    real, then delegates to execute_targeted_rerun(). Used by both the
    UNMAPPED targeted rerun (Phase 2) and the optional mapped-row stability
    sample -- both are "rerun exactly this query_id set" operations."""
    model_config = pinned_model_config()
    run_config = build_pinned_run_config(model_config=model_config, sapbert_url=sapbert_url)

    sapbert_health = check_sapbert_health(sapbert_url)
    graph_index = get_graph_index()

    provider = build_provider(model_config.model)
    run_preflight(provider, run_config.to_llm_call_config())

    try:
        pricing = get_pricing(model_config.model)
    except KeyError:
        pricing = None

    mapper = build_mapper(provider=provider, run_config=run_config)
    canonical_queries = load_canonical_queries(spec)

    already_completed: set[int] = set()
    if resume_dir is not None:
        output_dir = resume_dir
        existing_config = load_experiment_config(output_dir / "experiment_config.json")
        if existing_config is None:
            raise Scenario1PatchError(f"No experiment_config.json found in {output_dir} -- cannot resume.")
        start_timestamp = existing_config.get("start_timestamp", datetime.now(timezone.utc).isoformat())
        new_config = _build_rerun_experiment_config(
            spec=spec,
            model_config=model_config,
            sapbert_url=sapbert_url,
            temperature=PINNED_TEMPERATURE,
            seed=PINNED_SEED,
            sapbert_health=sapbert_health,
            canonical_queries=canonical_queries,
            targeted_query_ids=targeted_query_ids,
            source_label=source_label,
            original_status_filter=original_status_filter,
            source_dir=source_dir,
            start_timestamp=start_timestamp,
        )
        try:
            validate_resume(existing_config, new_config)
        except ResumeConfigMismatchError as exc:
            raise Scenario1PatchError(str(exc)) from exc

        resume_query_ids = quarantine_error_rows_for_resume(
            output_dir,
            resume_timestamp=datetime.now(timezone.utc).isoformat(),
            provider=PINNED_PROVIDER,
            model=model_config.model,
        )
        already_completed = resume_query_ids & targeted_query_ids
        append = True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        output_dir = output_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)
        start_timestamp = datetime.now(timezone.utc).isoformat()
        new_config = _build_rerun_experiment_config(
            spec=spec,
            model_config=model_config,
            sapbert_url=sapbert_url,
            temperature=PINNED_TEMPERATURE,
            seed=PINNED_SEED,
            sapbert_health=sapbert_health,
            canonical_queries=canonical_queries,
            targeted_query_ids=targeted_query_ids,
            source_label=source_label,
            original_status_filter=original_status_filter,
            source_dir=source_dir,
            start_timestamp=start_timestamp,
        )
        append = False

    write_experiment_config(new_config, output_dir / "experiment_config.json")

    outcome = execute_targeted_rerun(
        label=spec.label,
        mapper=mapper,
        canonical_queries=canonical_queries,
        targeted_query_ids=targeted_query_ids,
        pricing=pricing,
        graph_index=graph_index,
        output_dir=output_dir,
        append=append,
        already_completed_query_ids=already_completed,
        max_consecutive_local_retrieval_errors=max_consecutive_local_retrieval_errors,
    )

    new_config["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    new_config["completed"] = outcome.completed
    new_config["rows_completed"] = outcome.rows_completed_total
    if outcome.stop_reason is not None:
        new_config["stop_reason"] = outcome.stop_reason
    write_experiment_config(new_config, output_dir / "experiment_config.json")

    return outcome


def run_unmapped_rerun(
    spec: Scenario1DatasetSpec,
    *,
    sapbert_url: str = DEFAULT_SAPBERT_URL,
    resume_dir: Path | None = None,
    max_consecutive_local_retrieval_errors: int = DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS,
) -> RerunOutcome:
    """Phase 2 entry point: rerun exactly the query_ids recorded in
    <original_run_dir>/unmapped_subset.csv (Phase 1's output)."""
    subset_path = spec.original_run_dir / "unmapped_subset.csv"
    targeted_query_ids = read_subset_query_ids(subset_path)
    return run_rerun_for_query_ids(
        spec,
        targeted_query_ids,
        output_root=spec.rerun_output_root,
        source_dir=spec.original_run_dir,
        source_label=spec.label,
        original_status_filter="unmapped",
        sapbert_url=sapbert_url,
        resume_dir=resume_dir,
        max_consecutive_local_retrieval_errors=max_consecutive_local_retrieval_errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 -- rerun validation (re-read from disk, standalone from Phase 2)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerunValidationResult:
    dataset_key: str
    subset_count: int
    rerun_count: int
    missing_query_ids: tuple[int, ...]
    unexpected_query_ids: tuple[int, ...]
    duplicate_query_ids: tuple[int, ...]
    passed: bool


def validate_rerun_against_subset(spec: Scenario1DatasetSpec, rerun_dir: Path) -> RerunValidationResult:
    subset_path = spec.original_run_dir / "unmapped_subset.csv"
    subset_ids = sorted(read_subset_query_ids(subset_path))

    rerun_rows = read_existing_predictions(rerun_dir / "predictions.csv")
    rerun_ids = [int(r["query_id"]) for r in rerun_rows]

    subset_set, rerun_set = set(subset_ids), set(rerun_ids)
    missing = tuple(sorted(subset_set - rerun_set))
    unexpected = tuple(sorted(rerun_set - subset_set))
    dup_counts = Counter(rerun_ids)
    duplicates = tuple(sorted(qid for qid, n in dup_counts.items() if n > 1))

    passed = (
        not missing
        and not unexpected
        and not duplicates
        and len(subset_ids) == len(subset_set)
    )
    return RerunValidationResult(
        dataset_key=spec.key,
        subset_count=len(subset_ids),
        rerun_count=len(rerun_ids),
        missing_query_ids=missing,
        unexpected_query_ids=unexpected,
        duplicate_query_ids=duplicates,
        passed=passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 -- patch (immutable/mapper-output/derived-scoring column contract)
# ─────────────────────────────────────────────────────────────────────────────


def _row_metrics_and_graph(row: dict[str, Any], *, graph_index: EfoGraphIndex) -> tuple[RowMetrics, GraphDistanceResult | None]:
    ranks = tuple((row.get(f"rank_{i + 1}_code") or None) for i in range(MAX_RANK))
    rank_ontologies = tuple((row.get(f"rank_{i + 1}_ontology") or None) for i in range(MAX_RANK))
    gold_codes = _split_pipe(row.get("gold_codes", ""))
    record = PredictionRecord(
        query_id=int(row["query_id"]),
        query=row["query"],
        gold_codes=gold_codes,
        status=row["status"],
        ranks=ranks,
        rank_ontologies=rank_ontologies,
    )
    row_metrics = score_prediction(record)
    graph = graph_index.classify(ranks[0], list(gold_codes)) if row["status"] == "mapped" else None
    return row_metrics, graph


def _apply_derived_scoring(row: dict[str, Any], row_metrics: RowMetrics, graph: GraphDistanceResult | None) -> None:
    row["first_gold_rank"] = row_metrics.gold_rank
    row["top1_hit"] = row_metrics.top1_hit
    row["top3_hit"] = row_metrics.top3_hit
    row["top5_hit"] = row_metrics.top5_hit
    row["reciprocal_rank"] = row_metrics.reciprocal_rank
    row["recall_at_gt"] = row_metrics.recall_at_gt
    row["graph_relationship"] = graph.graph_relationship if graph is not None else ""
    row["graph_matched_gold_code"] = graph.graph_matched_gold_code if graph is not None else ""


@dataclass(frozen=True)
class PatchResult:
    dataset_key: str
    patched_dir: Path
    original_selection_run: Path
    baseline_run: Path
    original_row_count: int
    targeted_row_count: int
    rerun_row_count: int
    patched_row_count: int
    replaced_query_ids_count: int
    missing_query_ids: tuple[int, ...]
    unexpected_query_ids: tuple[int, ...]
    duplicate_query_ids: tuple[int, ...]
    immutable_column_mismatches: tuple[dict[str, Any], ...]
    non_target_row_mismatches: tuple[dict[str, Any], ...]
    passed: bool


def build_patched_predictions(
    spec: Scenario1DatasetSpec,
    rerun_dir: Path,
    patched_dir: Path,
    *,
    original_selection_run: Path | None = None,
    baseline_run: Path | None = None,
) -> PatchResult:
    """Phase 4 + row-level Derived Scoring recompute.

    Three distinct, independently-overridable sources (see the UKBB
    gold-parsing-bug fix, which is exactly why this is three parameters and
    not one):

        original_selection_run  -- determines WHICH query_ids were originally
                                    UNMAPPED (status=="unmapped"). Defaults to
                                    spec.original_run_dir. This is the ONLY
                                    thing that decides the targeted set --
                                    never re-derived from a corrected baseline.
        baseline_run             -- supplies immutable source/gold metadata
                                    and mapper-output fields for every row
                                    NOT in the targeted set, and immutable
                                    fields for rows that ARE. Defaults to
                                    original_selection_run (the common case:
                                    no gold correction needed). Pass a
                                    gold-corrected baseline directory here to
                                    patch on top of corrected gold instead of
                                    the buggy original.
        rerun_dir                -- supplies fresh mapper-output fields for
                                    the targeted query_ids.

    baseline_run must cover exactly the same query_id set as
    original_selection_run (a gold-metadata correction never adds/removes
    rows) -- checked explicitly below.
    """
    original_selection_run = original_selection_run or spec.original_run_dir
    baseline_run = baseline_run or original_selection_run

    selection_rows = read_existing_predictions(original_selection_run / "predictions.csv")
    if not selection_rows:
        raise Scenario1PatchError(f"{spec.label}: original_selection_run predictions.csv is empty or missing.")
    targeted_ids = {int(r["query_id"]) for r in selection_rows if r.get("status") == "unmapped"}
    selection_by_id = {int(r["query_id"]): r for r in selection_rows}

    base_rows = read_existing_predictions(baseline_run / "predictions.csv")
    if not base_rows:
        raise Scenario1PatchError(f"{spec.label}: baseline_run predictions.csv is empty or missing.")
    base_by_id = {int(r["query_id"]): r for r in base_rows}

    if set(selection_by_id) != set(base_by_id):
        raise Scenario1PatchError(
            f"{spec.label}: baseline_run {baseline_run} does not cover the same query_id set as "
            f"original_selection_run {original_selection_run} -- refusing to patch."
        )

    rerun_rows = read_existing_predictions(rerun_dir / "predictions.csv")

    rerun_by_id: dict[int, dict[str, str]] = {}
    duplicate_ids: list[int] = []
    for r in rerun_rows:
        qid = int(r["query_id"])
        if qid in rerun_by_id:
            duplicate_ids.append(qid)
        rerun_by_id[qid] = r
    rerun_ids = set(rerun_by_id)

    missing = tuple(sorted(targeted_ids - rerun_ids))
    unexpected = tuple(sorted(rerun_ids - targeted_ids))
    duplicate_query_ids = tuple(sorted(set(duplicate_ids)))

    if missing or unexpected or duplicate_query_ids:
        raise Scenario1PatchError(
            f"{spec.label}: refusing to patch -- rerun does not exactly match the originally-unmapped "
            f"set. missing={list(missing)} unexpected={list(unexpected)} duplicates={list(duplicate_query_ids)}"
        )

    graph_index = get_graph_index()
    immutable_mismatches: list[dict[str, Any]] = []
    patched_rows: list[dict[str, Any]] = []
    replaced_count = 0

    for row in base_rows:
        qid = int(row["query_id"])
        new_row = dict(row)  # immutable + (pre-replacement) mapper-output from the BASELINE, never the rerun
        if qid in targeted_ids:
            rerun_row = rerun_by_id[qid]
            selection_row = selection_by_id.get(qid, {})
            # Sanity check against the ORIGINAL SELECTION run's immutable
            # fields (what the rerun actually targeted), never against the
            # baseline -- a corrected baseline is EXPECTED to differ from a
            # rerun that executed before the correction existed (exactly the
            # UKBB gold-parsing-bug scenario: query 872's rerun still carries
            # the old compound gold_codes text).
            for f in IMMUTABLE_FIELDS:
                if selection_row.get(f, "") != rerun_row.get(f, ""):
                    immutable_mismatches.append(
                        {
                            "query_id": qid,
                            "field": f,
                            "original_selection": selection_row.get(f, ""),
                            "rerun": rerun_row.get(f, ""),
                        }
                    )
            for f in MAPPER_OUTPUT_FIELDS:
                new_row[f] = rerun_row.get(f, "")
            replaced_count += 1
        row_metrics, graph = _row_metrics_and_graph(new_row, graph_index=graph_index)
        _apply_derived_scoring(new_row, row_metrics, graph)
        patched_rows.append(new_row)

    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_predictions_path = patched_dir / "predictions.csv"
    with patched_predictions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in patched_rows:
            writer.writerow({f: row.get(f, "") for f in PREDICTIONS_CSV_FIELDS})

    # Re-read from disk (not the in-memory rows) for the validation below --
    # verifies what was actually written, not what we intended to write.
    reread = read_existing_predictions(patched_predictions_path)
    non_target_mismatches: list[dict[str, Any]] = []
    for row in reread:
        qid = int(row["query_id"])
        if qid in targeted_ids:
            continue
        base = base_by_id.get(qid, {})
        for f in MAPPER_OUTPUT_FIELDS:
            if row.get(f, "") != base.get(f, ""):
                non_target_mismatches.append(
                    {"query_id": qid, "field": f, "baseline": base.get(f, ""), "patched": row.get(f, "")}
                )

    patched_ids = [int(r["query_id"]) for r in reread]
    passed = (
        not missing
        and not unexpected
        and not duplicate_query_ids
        and not immutable_mismatches
        and not non_target_mismatches
        and len(reread) == len(base_rows)
        and len(patched_ids) == len(set(patched_ids))
        and replaced_count == len(targeted_ids)
    )

    return PatchResult(
        dataset_key=spec.key,
        patched_dir=patched_dir,
        original_selection_run=original_selection_run,
        baseline_run=baseline_run,
        original_row_count=len(base_rows),
        targeted_row_count=len(targeted_ids),
        rerun_row_count=len(rerun_rows),
        patched_row_count=len(reread),
        replaced_query_ids_count=replaced_count,
        missing_query_ids=missing,
        unexpected_query_ids=unexpected,
        duplicate_query_ids=duplicate_query_ids,
        immutable_column_mismatches=tuple(immutable_mismatches),
        non_target_row_mismatches=tuple(non_target_mismatches),
        passed=passed,
    )


def write_patched_experiment_config(
    spec: Scenario1DatasetSpec,
    patched_dir: Path,
    *,
    rerun_dir: Path,
    baseline_run: Path | None = None,
) -> Path:
    baseline_run = baseline_run or spec.original_run_dir
    baseline_config = load_experiment_config(baseline_run / "experiment_config.json")
    if baseline_config is None:
        raise Scenario1PatchError(f"{spec.label}: no experiment_config.json found in {baseline_run}")
    config = dict(baseline_config)
    # Absolute path so --evaluate-existing resolves correctly regardless of
    # the invoking process's current working directory -- this is a brand
    # new file, so making it absolute does not touch the historical record.
    config["source_dataset_path"] = str(spec.dataset_path)
    config["patched"] = True
    config["patch_original_selection_run"] = str(spec.original_run_dir)
    config["patch_baseline_run"] = str(baseline_run)
    config["patch_rerun_run"] = str(rerun_dir)
    config["patch_timestamp"] = datetime.now(timezone.utc).isoformat()
    path = patched_dir / "experiment_config.json"
    write_experiment_config(config, path)
    return path


def write_patch_validation_json(result: PatchResult, spec: Scenario1DatasetSpec, rerun_dir: Path) -> Path:
    payload = {
        "original_selection_run": str(result.original_selection_run),
        "baseline_run": str(result.baseline_run),
        "rerun_run": str(rerun_dir),
        "patched_run": str(result.patched_dir),
        "original_row_count": result.original_row_count,
        "targeted_row_count": result.targeted_row_count,
        "rerun_row_count": result.rerun_row_count,
        "patched_row_count": result.patched_row_count,
        "replaced_query_ids_count": result.replaced_query_ids_count,
        "missing_query_ids": list(result.missing_query_ids),
        "unexpected_query_ids": list(result.unexpected_query_ids),
        "duplicate_query_ids": list(result.duplicate_query_ids),
        "immutable_column_mismatches": list(result.immutable_column_mismatches),
        "non_target_row_mismatches": list(result.non_target_row_mismatches),
        "validation_passed": result.passed,
    }
    path = result.patched_dir / "patch_validation.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Gold-metadata correction (UKBB-EFO multi-gold parsing-bug fix)
#
# A one-time, zero-mapper-call remediation: rebuild ONLY gold_codes/
# gold_labels/gold_count for every row of a full copy of an original
# predictions.csv, using the corrected scenario1_dataset.build_canonical_
# queries() parsing, then recompute derived scoring for the corrected file.
# The original predictions.csv is opened for reading only. This is
# logically prior to, and independent of, the UNMAPPED rerun-and-patch
# workflow above -- see build_patched_predictions()'s baseline_run
# parameter for how the two compose.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoldCorrectionResult:
    dataset_key: str
    corrected_dir: Path
    source_original_run: Path
    original_row_count: int
    corrected_row_count: int
    affected_query_ids: tuple[int, ...]
    gold_metadata_changes: tuple[dict[str, Any], ...]
    mapper_output_mismatch_count: int
    missing_query_ids: tuple[int, ...]
    unexpected_query_ids: tuple[int, ...]
    duplicate_query_ids: tuple[int, ...]
    original_predictions_sha256: str
    passed: bool


def build_gold_corrected_predictions(spec: Scenario1DatasetSpec, corrected_dir: Path) -> GoldCorrectionResult:
    """Read spec.original_run_dir/predictions.csv (read-only), rebuild
    GOLD_METADATA_FIELDS for every row from freshly-parsed canonical queries
    (the fixed build_canonical_queries()), recompute DERIVED_SCORING_FIELDS
    for every row, and write the result to corrected_dir/predictions.csv.
    Every MAPPER_OUTPUT_FIELDS value is verified byte-identical to the
    original (re-read from disk) -- this function must never change mapper
    output, and calls no mapper/provider/retrieval function of any kind."""
    original_path = spec.original_run_dir / "predictions.csv"
    original_rows = read_existing_predictions(original_path)
    if not original_rows:
        raise Scenario1PatchError(f"{spec.label}: original predictions.csv is empty or missing.")
    original_sha256 = file_sha256(original_path)

    canonical_queries = load_canonical_queries(spec)
    canonical_by_id = {cq.query_id: cq for cq in canonical_queries}

    graph_index = get_graph_index()
    corrected_rows: list[dict[str, Any]] = []
    affected_ids: list[int] = []
    gold_changes: list[dict[str, Any]] = []

    seen_ids: set[int] = set()
    duplicate_ids: list[int] = []
    for row in original_rows:
        qid = int(row["query_id"])
        if qid in seen_ids:
            duplicate_ids.append(qid)
        seen_ids.add(qid)

        cq = canonical_by_id.get(qid)
        if cq is None:
            raise Scenario1PatchError(
                f"{spec.label}: query_id={qid} from the original predictions.csv was not found among "
                f"the canonical queries freshly built from {spec.dataset_path} -- the dataset file may "
                "have changed since the original run. Refusing to guess; investigate before correcting."
            )

        new_row = dict(row)
        new_gold_codes = _join_pipe(cq.gold_codes)
        new_gold_labels = _join_pipe(cq.gold_labels)
        new_gold_count = str(cq.gold_count)

        if (
            new_gold_codes != row.get("gold_codes", "")
            or new_gold_labels != row.get("gold_labels", "")
            or new_gold_count != row.get("gold_count", "")
        ):
            affected_ids.append(qid)
            gold_changes.append(
                {
                    "query_id": qid,
                    "old_gold_codes": row.get("gold_codes", ""),
                    "new_gold_codes": new_gold_codes,
                    "old_gold_labels": row.get("gold_labels", ""),
                    "new_gold_labels": new_gold_labels,
                    "old_gold_count": row.get("gold_count", ""),
                    "new_gold_count": new_gold_count,
                }
            )

        new_row["gold_codes"] = new_gold_codes
        new_row["gold_labels"] = new_gold_labels
        new_row["gold_count"] = new_gold_count

        row_metrics, graph = _row_metrics_and_graph(new_row, graph_index=graph_index)
        _apply_derived_scoring(new_row, row_metrics, graph)
        corrected_rows.append(new_row)

    corrected_dir.mkdir(parents=True, exist_ok=True)
    corrected_predictions_path = corrected_dir / "predictions.csv"
    with corrected_predictions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in corrected_rows:
            writer.writerow({f: row.get(f, "") for f in PREDICTIONS_CSV_FIELDS})

    # Re-read from disk and verify mapper-output fields are byte-identical to
    # the original for EVERY row -- this function must never change them.
    reread = read_existing_predictions(corrected_predictions_path)
    original_by_id = {int(r["query_id"]): r for r in original_rows}
    mapper_output_mismatches = 0
    for row in reread:
        qid = int(row["query_id"])
        orig = original_by_id.get(qid, {})
        for f in MAPPER_OUTPUT_FIELDS:
            if row.get(f, "") != orig.get(f, ""):
                mapper_output_mismatches += 1

    corrected_ids = {int(r["query_id"]) for r in reread}
    original_ids = set(original_by_id)
    missing = tuple(sorted(original_ids - corrected_ids))
    unexpected = tuple(sorted(corrected_ids - original_ids))
    duplicate_query_ids = tuple(sorted(set(duplicate_ids)))

    passed = (
        not missing
        and not unexpected
        and not duplicate_query_ids
        and mapper_output_mismatches == 0
        and len(reread) == len(original_rows)
    )

    return GoldCorrectionResult(
        dataset_key=spec.key,
        corrected_dir=corrected_dir,
        source_original_run=spec.original_run_dir,
        original_row_count=len(original_rows),
        corrected_row_count=len(reread),
        affected_query_ids=tuple(sorted(affected_ids)),
        gold_metadata_changes=tuple(gold_changes),
        mapper_output_mismatch_count=mapper_output_mismatches,
        missing_query_ids=missing,
        unexpected_query_ids=unexpected,
        duplicate_query_ids=duplicate_query_ids,
        original_predictions_sha256=original_sha256,
        passed=passed,
    )


def write_gold_corrected_experiment_config(spec: Scenario1DatasetSpec, corrected_dir: Path) -> Path:
    original_config = load_experiment_config(spec.original_run_dir / "experiment_config.json")
    if original_config is None:
        raise Scenario1PatchError(f"{spec.label}: no experiment_config.json found in {spec.original_run_dir}")
    config = dict(original_config)
    config["source_dataset_path"] = str(spec.dataset_path)
    config["gold_corrected"] = True
    config["gold_correction_source_run"] = str(spec.original_run_dir)
    config["gold_correction_timestamp"] = datetime.now(timezone.utc).isoformat()
    path = corrected_dir / "experiment_config.json"
    write_experiment_config(config, path)
    return path


def write_gold_correction_validation_json(result: GoldCorrectionResult, spec: Scenario1DatasetSpec) -> Path:
    payload = {
        "source_original_run": str(result.source_original_run),
        "source_dataset": str(spec.dataset_path),
        "original_row_count": result.original_row_count,
        "corrected_row_count": result.corrected_row_count,
        "affected_query_ids": list(result.affected_query_ids),
        "affected_row_count": len(result.affected_query_ids),
        "gold_metadata_changes": list(result.gold_metadata_changes),
        "mapper_output_mismatch_count": result.mapper_output_mismatch_count,
        "missing_query_ids": list(result.missing_query_ids),
        "unexpected_query_ids": list(result.unexpected_query_ids),
        "duplicate_query_ids": list(result.duplicate_query_ids),
        "original_predictions_sha256": result.original_predictions_sha256,
        "validation_passed": result.passed,
    }
    path = result.corrected_dir / "gold_correction_validation.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 -- regenerate companion outputs (zero mapper calls, existing script)
# ─────────────────────────────────────────────────────────────────────────────


def regenerate_patched_companions(patched_dir: Path, *, published_baselines: Path | None = None) -> None:
    """Reuses scripts/run_scenario1_ols_efo.py --evaluate-existing verbatim
    (subprocess, not reimplemented) to regenerate scenario1_metrics.csv/.md,
    graph_distance_rows.csv/summary.csv, namespace_distribution.csv,
    manual_review_required.csv, execution_diagnostics.csv,
    telemetry_summary.csv, published_comparison.csv/.md from the patched
    predictions.csv. Never invokes the mapper."""
    script = REPO_DIR / "scripts" / "run_scenario1_ols_efo.py"
    cmd = [sys.executable, str(script), "--evaluate-existing", str(patched_dir)]
    if published_baselines is not None:
        cmd += ["--published-baselines", str(published_baselines)]
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 -- summary comparison (reporting only, never gates patching)
# ─────────────────────────────────────────────────────────────────────────────

TRACKED_METRICS: tuple[str, ...] = ("Top-1", "Top-3", "Top-5", "MRR", "Recall@GT", "Precision", "Recall", "F1")


def read_metric_table(run_dir: Path) -> dict[str, str]:
    path = run_dir / "scenario1_metrics.csv"
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            metrics[row["metric"]] = row["value"]
    return metrics


def summarize_patch(spec: Scenario1DatasetSpec, rerun_dir: Path, patched_dir: Path) -> dict[str, Any]:
    original_metrics = read_metric_table(spec.original_run_dir)
    patched_metrics = read_metric_table(patched_dir)
    metric_comparison = {
        m: {"original": original_metrics.get(m), "patched": patched_metrics.get(m)} for m in TRACKED_METRICS
    }

    rerun_rows = read_existing_predictions(rerun_dir / "predictions.csv")
    still_unmapped = sum(1 for r in rerun_rows if r["status"] == "unmapped")
    now_mapped = sum(1 for r in rerun_rows if r["status"] == "mapped")
    now_error = sum(1 for r in rerun_rows if r["status"] == "error")

    gold_rank_counts: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "absent": 0}
    for row in rerun_rows:
        ranks = tuple((row.get(f"rank_{i + 1}_code") or None) for i in range(MAX_RANK))
        gold_codes = _split_pipe(row.get("gold_codes", ""))
        rank = first_gold_rank(ranks, gold_codes)
        if rank is None:
            gold_rank_counts["absent"] += 1
        else:
            gold_rank_counts[str(rank)] += 1

    return {
        "dataset": spec.key,
        "metric_comparison": metric_comparison,
        "transitions": {
            "originally_unmapped_still_unmapped": still_unmapped,
            "originally_unmapped_now_mapped": now_mapped,
            "originally_unmapped_now_error": now_error,
            "total_rerun": len(rerun_rows),
        },
        "gold_rank_among_rerun_rows": gold_rank_counts,
    }


def write_summary_json(summary: dict[str, Any], patched_dir: Path) -> Path:
    path = patched_dir / "patch_summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Optional -- mapped-row stability sample (never mixed into patched outputs)
# ─────────────────────────────────────────────────────────────────────────────


def sample_mapped_query_ids(spec: Scenario1DatasetSpec, *, n: int, seed: int = STABILITY_SAMPLE_SEED) -> list[int]:
    rows = read_existing_predictions(spec.original_run_dir / "predictions.csv")
    mapped_ids = sorted(int(r["query_id"]) for r in rows if r.get("status") == "mapped")
    if n >= len(mapped_ids):
        return mapped_ids
    rng = random.Random(seed)
    return sorted(rng.sample(mapped_ids, n))


def run_stability_sample_rerun(
    spec: Scenario1DatasetSpec,
    *,
    n: int | None = None,
    sapbert_url: str = DEFAULT_SAPBERT_URL,
    resume_dir: Path | None = None,
    max_consecutive_local_retrieval_errors: int = DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS,
) -> RerunOutcome:
    sample_size = n if n is not None else STABILITY_SAMPLE_SIZES[spec.key]
    targeted_query_ids = set(sample_mapped_query_ids(spec, n=sample_size))
    return run_rerun_for_query_ids(
        spec,
        targeted_query_ids,
        output_root=spec.stability_output_root,
        source_dir=spec.original_run_dir,
        source_label=spec.label,
        original_status_filter="mapped_stability_sample",
        sapbert_url=sapbert_url,
        resume_dir=resume_dir,
        max_consecutive_local_retrieval_errors=max_consecutive_local_retrieval_errors,
    )


def summarize_stability_sample(spec: Scenario1DatasetSpec, stability_dir: Path) -> dict[str, Any]:
    """Diagnostic-only comparison -- these rows must never be written into
    the patched benchmark."""
    original_rows = {
        int(r["query_id"]): r for r in read_existing_predictions(spec.original_run_dir / "predictions.csv")
    }
    rerun_rows = read_existing_predictions(stability_dir / "predictions.csv")

    n = len(rerun_rows)
    mapped_code_agree = 0
    top1_agree = 0
    top5_agree = 0
    mapped_to_unmapped = 0

    for row in rerun_rows:
        qid = int(row["query_id"])
        orig = original_rows.get(qid)
        if orig is None:
            continue
        if row.get("mapped_code", "") == orig.get("mapped_code", ""):
            mapped_code_agree += 1
        if row.get("top1_hit", "") == orig.get("top1_hit", ""):
            top1_agree += 1
        if row.get("top5_hit", "") == orig.get("top5_hit", ""):
            top5_agree += 1
        if orig.get("status") == "mapped" and row.get("status") == "unmapped":
            mapped_to_unmapped += 1

    return {
        "dataset": spec.key,
        "sample_size": n,
        "mapped_code_exact_agreement_rate": (mapped_code_agree / n) if n else None,
        "top1_agreement_rate": (top1_agree / n) if n else None,
        "top5_agreement_rate": (top5_agree / n) if n else None,
        "mapped_to_unmapped_transition_count": mapped_to_unmapped,
        "mapped_to_unmapped_transition_rate": (mapped_to_unmapped / n) if n else None,
    }


__all__ = [
    "DATASET_SPECS",
    "DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS",
    "DEFAULT_PUBLISHED_BASELINES",
    "DEFAULT_SAPBERT_URL",
    "DERIVED_SCORING_FIELDS",
    "EXPECTED_TOTAL_ROW_COUNTS",
    "EXPECTED_UNMAPPED_COUNTS",
    "GOLD_METADATA_FIELDS",
    "IMMUTABLE_FIELDS",
    "MAPPER_OUTPUT_FIELDS",
    "PINNED_MAX_ALTERNATIVES",
    "PINNED_MAX_CANDIDATES",
    "PINNED_MAX_RESULTS_PER_QUERY",
    "PINNED_MODEL",
    "PINNED_PROVIDER",
    "PINNED_REASONING_EFFORT",
    "PINNED_SEED",
    "PINNED_TEMPERATURE",
    "STABILITY_SAMPLE_SEED",
    "STABILITY_SAMPLE_SIZES",
    "TRACKED_METRICS",
    "UNMAPPED_SUBSET_CSV_FIELDS",
    "ExtractionResult",
    "GoldCorrectionResult",
    "PatchResult",
    "RerunOutcome",
    "RerunValidationResult",
    "Scenario1DatasetSpec",
    "Scenario1PatchError",
    "build_gold_corrected_predictions",
    "build_patched_predictions",
    "build_pinned_run_config",
    "dataset_keys",
    "execute_targeted_rerun",
    "extract_unmapped_subset",
    "load_canonical_queries",
    "pinned_model_config",
    "read_metric_table",
    "read_subset_query_ids",
    "regenerate_patched_companions",
    "run_rerun_for_query_ids",
    "run_stability_sample_rerun",
    "run_unmapped_rerun",
    "sample_mapped_query_ids",
    "select_unmapped_rows",
    "summarize_patch",
    "summarize_stability_sample",
    "validate_rerun_against_subset",
    "write_gold_correction_validation_json",
    "write_gold_corrected_experiment_config",
    "write_patch_validation_json",
    "write_patched_experiment_config",
    "write_summary_json",
]
