#!/usr/bin/env python
"""
Targeted Scenario 1 EFO UNMAPPED rerun-and-patch orchestration.

Applies ONLY to the three canonical Scenario 1 EFO benchmark runs identified
by the completed read-only audit:

    ols-efo           outputs/evaluation/scenario1_ols_efo/2026-08-26T15-04-18Z/
    biomappings-efo    outputs/evaluation/scenario1_biomappings_efo/2026-08-31T16-10-24Z/
    ukbb-efo            outputs/evaluation/scenario1_ukbb_efo/2026-08-31T13-54-53Z/

For each selected dataset, reruns ONLY the rows that were `status ==
"unmapped"` in that original predictions.csv, pinned to the ORIGINAL
benchmark configuration (max_results_per_query=10, max_candidates=10,
max_alternatives=4 -- see scenario1_patch.py), then patches exactly those
rows into a full-size COPY of the original benchmark. The original run
directories are never opened for writing.

All actual pipeline/scoring/mapper logic is reused from the existing
benchmarking modules (scenario1_dataset, scenario1_runner, scenario1_output,
scenario1_metrics, scenario1_graph_distance) via scenario1_patch.py -- this
script only sequences the phases and prints a human-readable report.

Usage
─────
Phase 1 (extraction, no network calls):
    uv run python scripts/patch_scenario1_efo_unmapped.py --extract --dataset all

Phase 2 (targeted rerun -- REAL mapper/LLM calls, requires OPENAI_API_KEY +
a running local SapBERT service):
    uv run python scripts/patch_scenario1_efo_unmapped.py --rerun --dataset ols-efo

Resume an interrupted targeted rerun:
    uv run python scripts/patch_scenario1_efo_unmapped.py --rerun --dataset ols-efo \\
        --resume-dir outputs/evaluation/scenario1_ols_efo_rerun_unmapped/<timestamp>

Phase 3 (validate the rerun against the extracted subset, no network calls):
    uv run python scripts/patch_scenario1_efo_unmapped.py --validate --dataset ols-efo

Phases 4-7 (patch + recompute scoring + regenerate companions + summary,
zero mapper calls):
    uv run python scripts/patch_scenario1_efo_unmapped.py --patch --dataset ols-efo

Full pipeline for one dataset (extract, then patch/validate against the most
recent rerun dir -- does NOT run Phase 2 automatically, since that costs
money/time; run --rerun explicitly first):
    uv run python scripts/patch_scenario1_efo_unmapped.py --extract --validate --patch --dataset ols-efo

Optional mapped-row stability sample (diagnostic only -- never mixed into
patched outputs):
    uv run python scripts/patch_scenario1_efo_unmapped.py --stability-sample --dataset all
    uv run python scripts/patch_scenario1_efo_unmapped.py --stability-summary --dataset all

This script never invokes the mapper/LLM except during --rerun and
--stability-sample. Every other phase is pure file I/O + the existing
Scenario 1 scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_ontology_mapper.benchmarking.scenario1_patch import (  # noqa: E402
    DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS,
    DEFAULT_PUBLISHED_BASELINES,
    DEFAULT_SAPBERT_URL,
    EXPECTED_UNMAPPED_COUNTS,
    STABILITY_SAMPLE_SIZES,
    DATASET_SPECS,
    Scenario1PatchError,
    build_gold_corrected_predictions,
    build_patched_predictions,
    dataset_keys,
    extract_unmapped_subset,
    regenerate_patched_companions,
    run_stability_sample_rerun,
    run_unmapped_rerun,
    summarize_patch,
    summarize_stability_sample,
    validate_rerun_against_subset,
    write_gold_correction_validation_json,
    write_gold_corrected_experiment_config,
    write_patch_validation_json,
    write_patched_experiment_config,
    write_summary_json,
)


def _latest_subdir(root: Path) -> Path:
    if not root.exists():
        raise Scenario1PatchError(f"{root} does not exist -- run --rerun first.")
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not candidates:
        raise Scenario1PatchError(f"No run directories found under {root} -- run --rerun first.")
    return candidates[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted Scenario 1 EFO UNMAPPED rerun-and-patch orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_SPECS.keys(), "all"],
        default="all",
        help="Which of the three canonical benchmarks to operate on (default: all)",
    )
    parser.add_argument("--extract", action="store_true", help="Phase 1: write unmapped_subset.csv")
    parser.add_argument("--rerun", action="store_true", help="Phase 2: rerun the targeted subset (REAL mapper calls)")
    parser.add_argument("--validate", action="store_true", help="Phase 3: validate rerun vs. extracted subset")
    parser.add_argument("--patch", action="store_true", help="Phases 4-7: patch, recompute scoring, regenerate companions, summarize")
    parser.add_argument(
        "--gold-correct",
        action="store_true",
        help=(
            "One-time, zero-mapper-call gold-metadata remediation (UKBB multi-gold parsing-bug fix): "
            "rebuild gold_codes/gold_labels/gold_count for every row in a fresh COPY of the original "
            "predictions.csv, recompute derived scoring, and regenerate companion outputs. Never "
            "modifies the original run directory or invokes the mapper. Logically prior to --patch; "
            "pass the resulting directory to --patch via --baseline-dir."
        ),
    )
    parser.add_argument(
        "--all", action="store_true", help="Run --validate --patch (does NOT run --extract or --rerun -- see module docstring)"
    )
    parser.add_argument("--stability-sample", action="store_true", help="Optional: rerun a mapped-row stability sample (REAL mapper calls)")
    parser.add_argument("--stability-summary", action="store_true", help="Optional: summarize the most recent stability-sample rerun")
    parser.add_argument("--stability-sample-size", type=int, default=None, help="Override the per-dataset stability sample size")

    parser.add_argument("--rerun-dir", default=None, help="Explicit rerun output directory (--validate/--patch). Default: most recent under the dataset's rerun_output_root")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help=(
            "Explicit baseline directory for --patch (immutable/gold metadata + non-target mapper "
            "output). Default: the dataset's original_run_dir (i.e. no gold correction). Pass a "
            "--gold-correct output directory here to patch on top of corrected gold. The targeted "
            "(originally-unmapped) query_id set is ALWAYS determined from original_run_dir, never "
            "from this directory."
        ),
    )
    parser.add_argument("--resume-dir", default=None, help="Existing targeted-rerun directory to resume (--rerun)")
    parser.add_argument("--sapbert-url", default=DEFAULT_SAPBERT_URL)
    parser.add_argument("--published-baselines", default=str(DEFAULT_PUBLISHED_BASELINES))
    parser.add_argument("--max-consecutive-local-retrieval-errors", type=int, default=DEFAULT_MAX_CONSECUTIVE_LOCAL_RETRIEVAL_ERRORS)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow --extract to proceed even if the unmapped-row count differs from the audit-confirmed count",
    )
    return parser.parse_args(argv)


def cmd_extract(key: str, args: argparse.Namespace) -> int:
    spec = DATASET_SPECS[key]
    result = extract_unmapped_subset(
        spec,
        expected_count=EXPECTED_UNMAPPED_COUNTS.get(key),
        allow_count_mismatch=args.allow_count_mismatch,
    )
    print(f"[{spec.label}] extracted {result.subset_count} unmapped row(s) -> {result.subset_path}")
    return 0


def cmd_rerun(key: str, args: argparse.Namespace) -> int:
    spec = DATASET_SPECS[key]
    resume_dir = Path(args.resume_dir) if args.resume_dir else None
    outcome = run_unmapped_rerun(
        spec,
        sapbert_url=args.sapbert_url,
        resume_dir=resume_dir,
        max_consecutive_local_retrieval_errors=args.max_consecutive_local_retrieval_errors,
    )
    print(
        f"[{spec.label}] targeted rerun: {outcome.rows_completed_total}/{outcome.total_targeted} rows "
        f"completed={outcome.completed} stop_reason={outcome.stop_reason} -> {outcome.output_dir}"
    )
    return 0 if outcome.completed else 1


def cmd_gold_correct(key: str, args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    spec = DATASET_SPECS[key]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    corrected_dir = spec.gold_corrected_output_root / timestamp

    result = build_gold_corrected_predictions(spec, corrected_dir)
    write_gold_corrected_experiment_config(spec, corrected_dir)
    validation_path = write_gold_correction_validation_json(result, spec)

    print(
        f"[{spec.label}] gold correction built: original={result.original_row_count} "
        f"corrected={result.corrected_row_count} affected={len(result.affected_query_ids)} "
        f"affected_query_ids={list(result.affected_query_ids)} "
        f"mapper_output_mismatch_count={result.mapper_output_mismatch_count} "
        f"passed={result.passed} -> {corrected_dir}"
    )
    if not result.passed:
        print(f"[{spec.label}] Gold correction validation FAILED -- see {validation_path}. Companion outputs NOT regenerated.")
        return 1

    regenerate_patched_companions(corrected_dir, published_baselines=Path(args.published_baselines))
    print(f"[{spec.label}] companions regenerated -> {corrected_dir}")
    return 0


def cmd_validate(key: str, args: argparse.Namespace) -> int:
    spec = DATASET_SPECS[key]
    rerun_dir = Path(args.rerun_dir) if args.rerun_dir else _latest_subdir(spec.rerun_output_root)
    result = validate_rerun_against_subset(spec, rerun_dir)
    status = "PASSED" if result.passed else "FAILED"
    print(
        f"[{spec.label}] rerun validation {status}: subset={result.subset_count} rerun={result.rerun_count} "
        f"missing={list(result.missing_query_ids)} unexpected={list(result.unexpected_query_ids)} "
        f"duplicates={list(result.duplicate_query_ids)}"
    )
    return 0 if result.passed else 1


def cmd_patch(key: str, args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    spec = DATASET_SPECS[key]
    rerun_dir = Path(args.rerun_dir) if args.rerun_dir else _latest_subdir(spec.rerun_output_root)

    pre_check = validate_rerun_against_subset(spec, rerun_dir)
    if not pre_check.passed:
        print(f"[{spec.label}] ABORTING patch: rerun validation failed against {rerun_dir} -- run --validate for details.")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    patched_dir = spec.patched_output_root / timestamp
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    result = build_patched_predictions(spec, rerun_dir, patched_dir, baseline_run=baseline_dir)
    write_patched_experiment_config(spec, patched_dir, rerun_dir=rerun_dir, baseline_run=baseline_dir)
    validation_path = write_patch_validation_json(result, spec, rerun_dir)

    print(
        f"[{spec.label}] patch built: original={result.original_row_count} targeted={result.targeted_row_count} "
        f"rerun={result.rerun_row_count} patched={result.patched_row_count} replaced={result.replaced_query_ids_count} "
        f"passed={result.passed} -> {patched_dir}"
    )
    if not result.passed:
        print(f"[{spec.label}] Patch validation FAILED -- see {validation_path}. Companion outputs NOT regenerated.")
        return 1

    regenerate_patched_companions(patched_dir, published_baselines=Path(args.published_baselines))
    summary = summarize_patch(spec, rerun_dir, patched_dir)
    summary_path = write_summary_json(summary, patched_dir)
    print(f"[{spec.label}] companions regenerated and summary written -> {summary_path}")
    for metric, values in summary["metric_comparison"].items():
        print(f"    {metric:12s} original={values['original']!s:>10} patched={values['patched']!s:>10}")
    print(f"    transitions: {summary['transitions']}")
    print(f"    gold rank among rerun rows: {summary['gold_rank_among_rerun_rows']}")
    return 0


def cmd_stability_sample(key: str, args: argparse.Namespace) -> int:
    spec = DATASET_SPECS[key]
    outcome = run_stability_sample_rerun(
        spec,
        n=args.stability_sample_size,
        sapbert_url=args.sapbert_url,
        max_consecutive_local_retrieval_errors=args.max_consecutive_local_retrieval_errors,
    )
    print(
        f"[{spec.label}] stability sample rerun: {outcome.rows_completed_total}/{outcome.total_targeted} rows "
        f"completed={outcome.completed} -> {outcome.output_dir}"
    )
    return 0 if outcome.completed else 1


def cmd_stability_summary(key: str, args: argparse.Namespace) -> int:
    spec = DATASET_SPECS[key]
    stability_dir = _latest_subdir(spec.stability_output_root)
    summary = summarize_stability_sample(spec, stability_dir)
    print(f"[{spec.label}] stability sample summary ({stability_dir}):")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all:
        args.validate = True
        args.patch = True

    if not any(
        [
            args.extract,
            args.rerun,
            args.validate,
            args.patch,
            args.gold_correct,
            args.stability_sample,
            args.stability_summary,
        ]
    ):
        print(
            "ERROR: specify at least one phase flag (--extract/--rerun/--validate/--patch/--gold-correct/"
            "--all/--stability-sample/--stability-summary)"
        )
        return 1

    keys = dataset_keys(args.dataset)
    exit_code = 0
    for key in keys:
        try:
            if args.extract:
                exit_code |= cmd_extract(key, args)
            if args.rerun:
                exit_code |= cmd_rerun(key, args)
            if args.gold_correct:
                exit_code |= cmd_gold_correct(key, args)
            if args.validate:
                exit_code |= cmd_validate(key, args)
            if args.patch:
                exit_code |= cmd_patch(key, args)
            if args.stability_sample:
                exit_code |= cmd_stability_sample(key, args)
            if args.stability_summary:
                exit_code |= cmd_stability_summary(key, args)
        except Scenario1PatchError as exc:
            print(f"ERROR [{key}]: {exc}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
