#!/usr/bin/env python
"""
Scenario 2 -- retrieval-mode ablation -- llm-ontology-mapper, dict_mapped_all.xlsx.

Measures how retrieval_mode in {public, local, disabled} affects mapping
performance, grounding, hallucination, confidence calibration, abstention,
latency, and cost while holding everything else fixed (model=gpt-5.6-luna,
reasoning_effort=low, temperature=provider_default, seed=42,
max_alternatives=4, strict_target_ontology=False -- all locked constants,
never CLI flags, so the three runs cannot silently drift from each other).

ONE INVOCATION RUNS EXACTLY ONE MODE. There is no "run all three modes"
option -- see --mode below. Cross-mode comparison is a SEPARATE, zero-LLM-call
step (--compare) that only reads already-completed run directories.

Usage
─────
Validate the dataset only (no LLM calls):
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --validate-only

10-row PUBLIC smoke:
    export OPENAI_API_KEY="..."
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --mode public --limit 10

Full PUBLIC run:
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --mode public

Full LOCAL run (SapBERT must be reachable at --sapbert-url):
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --mode local --sapbert-url http://localhost:8765

Full DISABLED run:
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --mode disabled

Resume an interrupted run:
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --input dict_mapped_all.xlsx --mode public \\
        --resume outputs/evaluation/scenario2_retrieval_ablation/public/<timestamp>

Recompute hallucination validation + every report from saved predictions --
no mapper/LLM calls:
    uv run python scripts/run_scenario2_retrieval_ablation.py \\
        --evaluate-existing outputs/evaluation/scenario2_retrieval_ablation/public/<timestamp>

Compare three completed runs (zero mapping/LLM calls):
    uv run python scripts/run_scenario2_retrieval_ablation.py --compare \\
        --public-dir outputs/evaluation/scenario2_retrieval_ablation/public/<ts> \\
        --local-dir outputs/evaluation/scenario2_retrieval_ablation/local/<ts> \\
        --disabled-dir outputs/evaluation/scenario2_retrieval_ablation/disabled/<ts>
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_ontology_mapper.benchmarking.dataset import (  # noqa: E402
    BenchmarkDatasetError,
    file_sha256,
    load_dataset,
)
from llm_ontology_mapper.benchmarking.model_registry import get_model_config  # noqa: E402
from llm_ontology_mapper.benchmarking.pricing import get_pricing  # noqa: E402
from llm_ontology_mapper.benchmarking.scenario2_calibration import (  # noqa: E402
    brier_score,
    build_calibration_pairs,
    confidence_separation_stats,
    expected_calibration_error,
    roc_auc,
)
from llm_ontology_mapper.benchmarking.scenario2_compare import (  # noqa: E402
    build_comparison_table,
    build_paired_predictions,
    load_and_validate_runs,
    read_mode_summary_values,
    transition_counts,
    write_comparison_csv,
    write_comparison_md,
    write_paired_predictions_csv,
)
from llm_ontology_mapper.benchmarking.scenario2_dataset import audit_dataset  # noqa: E402
from llm_ontology_mapper.benchmarking.scenario2_metrics import (  # noqa: E402
    STATUS_MAPPED,
    abstention_stats,
    aggregate,
    execution_diagnostics,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario2_output import (  # noqa: E402
    IncrementalPredictionsCsvWriter,
    ResumeConfigMismatchError,
    build_experiment_config,
    csv_row_to_prediction_record,
    load_experiment_config,
    quarantine_error_rows_for_resume,
    read_existing_predictions,
    read_validation_cache,
    rewrite_predictions_csv,
    row_result_to_csv_dict,
    run_validation_pass,
    validate_resume,
    write_calibration_bins_csv,
    write_calibration_statistics_csv,
    write_dataset_validation_json,
    write_execution_diagnostics_csv,
    write_experiment_config,
    write_mode_summary,
    write_retrieval_diagnostics_csv,
    write_telemetry_summary_csv,
    write_validation_cache,
)
from llm_ontology_mapper.benchmarking.scenario2_reliability_plot import (
    plot_reliability_diagram,  # noqa: E402
)
from llm_ontology_mapper.benchmarking.scenario2_runner import (  # noqa: E402
    RETRIEVAL_MODES,
    STRICT_TARGET_ONTOLOGY,
    PreflightError,
    SapBertHealthError,
    Scenario2RunConfig,
    build_pipeline_and_mappers,
    build_provider,
    check_sapbert_health,
    describe_temperature,
    iter_run_rows,
    run_preflight,
)
from llm_ontology_mapper.benchmarking.scenario2_validation import (  # noqa: E402
    NOT_APPLICABLE,
    summarize_hallucination,
)
from llm_ontology_mapper.ontology_identity import canonical_ontology  # noqa: E402
from llm_ontology_mapper.validator import OntologyValidator  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_DIR / "outputs" / "evaluation" / "scenario2_retrieval_ablation"
DEFAULT_SAPBERT_URL = "http://localhost:8765"

# ── Locked Scenario 2 configuration (Part 3/4) -- never CLI flags ────────────
LOCKED_PROVIDER = "openai"
LOCKED_MODEL = "gpt-5.6-luna"
LOCKED_REASONING_EFFORT = "low"
LOCKED_TEMPERATURE = None  # provider_default
LOCKED_SEED = 42
LOCKED_MAX_ALTERNATIVES = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 2 -- retrieval-mode ablation")
    parser.add_argument("--input", help="Path to dict_mapped_all.xlsx (required unless --evaluate-existing/--compare)")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, help="Exactly ONE retrieval mode for this invocation")
    parser.add_argument("--sapbert-url", default=DEFAULT_SAPBERT_URL, help=f"Default: {DEFAULT_SAPBERT_URL}")
    parser.add_argument("--output-root", default=None, help=f"Default: {DEFAULT_OUTPUT_ROOT}/<mode>")
    parser.add_argument("--resume", default=None, metavar="OUTPUT_DIR")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-existing", default=None, metavar="OUTPUT_DIR")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N dataset rows")
    parser.add_argument(
        "--max-consecutive-local-retrieval-errors",
        type=int,
        default=3,
        help="Local mode only: abort after this many consecutive local SapBERT retrieval infra failures.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Public/disabled mode: abort after this many consecutive execution errors of any kind "
        "(fail fast rather than burning through hundreds of rows on a public-API outage).",
    )

    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--public-dir", default=None, metavar="DIR")
    parser.add_argument("--local-dir", default=None, metavar="DIR")
    parser.add_argument("--disabled-dir", default=None, metavar="DIR")
    return parser.parse_args(argv)


def _load_and_audit(dataset_path: Path):  # type: ignore[no-untyped-def]
    rows = load_dataset(dataset_path)
    audit = audit_dataset(rows)
    return rows, audit


def _print_config_block(
    *,
    dataset_path: Path,
    sha256: str,
    n: int,
    mode: str | None,
    sapbert_url: str | None,
) -> None:
    print("=" * 78)
    print("Scenario 2 -- retrieval-mode ablation -- llm-ontology-mapper")
    print("=" * 78)
    print(f"Dataset:                  {dataset_path}")
    print(f"Dataset SHA256:           {sha256}")
    print(f"N:                        {n}")
    print(f"Provider:                 {LOCKED_PROVIDER}")
    print(f"Model:                    {LOCKED_MODEL}")
    print(f"Reasoning effort:         {LOCKED_REASONING_EFFORT}")
    print(f"Temperature:              {describe_temperature(LOCKED_TEMPERATURE)}")
    print(f"Seed:                     {LOCKED_SEED}")
    print(f"Max alternatives:         {LOCKED_MAX_ALTERNATIVES}")
    print(f"Strict target ontology:   {STRICT_TARGET_ONTOLOGY}")
    print(f"Retrieval mode:           {mode or 'N/A (dataset-only validation)'}")
    if mode == "local":
        print(f"SapBERT URL:              {sapbert_url}")
    print("=" * 78)


def _print_dataset_audit(audit) -> None:  # type: ignore[no-untyped-def]
    print("Dataset audit (Part 1, derived from the actual file -- nothing forced):")
    for key, value in audit.to_dict().items():
        if key == "namespace_violations":
            continue
        print(f"  {key}: {value}")
    if not audit.namespaces_consistent:
        print(f"  WARNING: {len(audit.namespace_violations)} gold-code namespace violation(s) found.")


# ─────────────────────────────────────────────────────────────────────────────
# finalize: hallucination validation (network, no LLM) + every report file.
# Re-runnable via --evaluate-existing with ZERO mapper/LLM calls (Part 18).
# ─────────────────────────────────────────────────────────────────────────────


def _finalize_mode_outputs(output_dir: Path, *, mode: str) -> None:
    predictions_path = output_dir / "predictions.csv"
    csv_rows = read_existing_predictions(predictions_path)

    validator = OntologyValidator()
    cache = read_validation_cache(output_dir / "validation_cache.csv")
    validated_rows = run_validation_pass(csv_rows, validator=validator, cache=cache)
    write_validation_cache(cache, output_dir / "validation_cache.csv")
    rewrite_predictions_csv(validated_rows, predictions_path)

    records = [csv_row_to_prediction_record(r) for r in validated_rows]
    row_metrics = [score_prediction(r) for r in records]
    agg = aggregate(row_metrics)

    mapped_codes = [r.get("mapped_code") or None for r in validated_rows]
    abstention = abstention_stats(records, mapped_codes)
    execution = execution_diagnostics(records)

    grounded_flags = [
        (r.get("selected_code_was_retrieved") or "").strip().lower() in {"true", "1", "yes"}
        for r in validated_rows
        if r.get("status") == STATUS_MAPPED
    ]
    grounding_rate_value = (
        sum(grounded_flags) / len(grounded_flags) if grounded_flags else None
    )

    validation_statuses = [r.get("validation_status", NOT_APPLICABLE) for r in validated_rows]
    hallucination = summarize_hallucination(mapped_count=execution.mapped_count, validation_statuses=validation_statuses)

    y_true, y_score = build_calibration_pairs(validated_rows)

    auc = roc_auc(y_true, y_score)
    brier = brier_score(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score)
    correct_scores = [s for y, s in zip(y_true, y_score, strict=True) if y == 1]
    incorrect_scores = [s for y, s in zip(y_true, y_score, strict=True) if y == 0]
    separation = confidence_separation_stats(correct_scores, incorrect_scores)

    write_calibration_bins_csv(mode, ece, output_dir / "calibration_bins.csv")
    write_calibration_statistics_csv(
        mode,
        calibration_n=len(y_true),
        n_correct=len(correct_scores),
        n_incorrect=len(incorrect_scores),
        auc=auc,
        brier=brier,
        ece=ece,
        separation=separation,
        path=output_dir / "calibration_statistics.csv",
    )
    write_execution_diagnostics_csv(execution, output_dir / "execution_diagnostics.csv")
    write_retrieval_diagnostics_csv(validated_rows, output_dir / "retrieval_diagnostics.csv")
    write_telemetry_summary_csv(validated_rows, output_dir / "telemetry_summary.csv")
    write_mode_summary(
        mode,
        agg=agg,
        abstention=abstention,
        grounding_rate=grounding_rate_value,
        hallucination=hallucination,
        auc=auc,
        brier=brier,
        ece=ece,
        separation=separation,
        execution=execution,
        csv_rows=validated_rows,
        path_csv=output_dir / "mode_summary.csv",
        path_md=output_dir / "mode_summary.md",
    )

    print(f"\nReports written to: {output_dir}")
    print(f"  N={agg.n}  Top-1={agg.top1:.4f}  Top-3={agg.top3:.4f}  Top-5={agg.top5:.4f}  MRR={agg.mrr:.4f}")
    print(f"  Abstention={abstention.abstention_rate:.4f}  Hallucination={hallucination.hallucination_rate}  "
          f"Grounding={grounding_rate_value}")
    print(f"  AUC={auc.value} ({auc.status})  Brier={brier}  ECE={ece.ece}")


# ─────────────────────────────────────────────────────────────────────────────
# --compare
# ─────────────────────────────────────────────────────────────────────────────


def _run_compare(args: argparse.Namespace) -> int:
    if not (args.public_dir and args.local_dir and args.disabled_dir):
        print("ERROR: --compare requires --public-dir, --local-dir, and --disabled-dir")
        return 1

    runs = load_and_validate_runs(
        public_dir=Path(args.public_dir),
        local_dir=Path(args.local_dir),
        disabled_dir=Path(args.disabled_dir),
    )

    paired = build_paired_predictions(runs)
    out_dir = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT / "comparison"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    write_paired_predictions_csv(paired, out_dir / "paired_predictions.csv")
    transitions = transition_counts(paired)

    mode_summaries = {
        mode: read_mode_summary_values(runs[mode].output_dir / "mode_summary.csv") for mode in runs
    }
    comparison_rows = build_comparison_table(mode_summaries)
    write_comparison_csv(comparison_rows, out_dir / "scenario2_comparison.csv")
    write_comparison_md(comparison_rows, transitions, out_dir / "scenario2_comparison.md")

    bins_by_mode = {}
    for mode, run in runs.items():
        bins_path = run.output_dir / "calibration_bins.csv"
        if bins_path.exists():
            import csv as _csv

            with bins_path.open(newline="", encoding="utf-8") as fh:
                bins_by_mode[mode] = list(_csv.DictReader(fh))
    if len(bins_by_mode) == 3:
        plot_reliability_diagram(
            bins_by_mode,
            output_png=out_dir / "reliability_diagram.png",
            output_svg=out_dir / "reliability_diagram.svg",
            output_pdf=out_dir / "reliability_diagram.pdf",
        )
        print(f"Reliability diagram written to: {out_dir}")

    print(f"\nComparison written to: {out_dir}")
    for row in comparison_rows:
        print(f"  {row['metric']:22s} public={row['public']}  local={row['local']}  disabled={row['disabled']}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.compare:
        return _run_compare(args)

    if args.evaluate_existing:
        output_dir = Path(args.evaluate_existing)
        if not output_dir.exists():
            print(f"ERROR: --evaluate-existing directory does not exist: {output_dir}")
            return 1
        config = load_experiment_config(output_dir / "experiment_config.json")
        if config is None:
            print(f"ERROR: no experiment_config.json found in {output_dir}")
            return 1
        _finalize_mode_outputs(output_dir, mode=config["retrieval_mode"])
        return 0

    if not args.input:
        print("ERROR: --input is required (unless --evaluate-existing/--compare)")
        return 1

    dataset_path = Path(args.input)
    try:
        rows, audit = _load_and_audit(dataset_path)
    except BenchmarkDatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    sha256 = file_sha256(dataset_path)
    _print_dataset_audit(audit)
    print(f"\ncanonical row count: {len(rows)}")

    if not audit.namespaces_consistent:
        print(
            f"\nERROR: {len(audit.namespace_violations)} gold-code namespace violation(s) found -- "
            "refusing to launch. See namespace_violations above."
        )
        return 1

    if args.limit is not None:
        rows = rows[: args.limit]

    if args.validate_only:
        _print_config_block(dataset_path=dataset_path, sha256=sha256, n=len(rows), mode=args.mode, sapbert_url=args.sapbert_url)
        if args.mode == "local":
            try:
                health = check_sapbert_health(args.sapbert_url)
                print(f"SapBERT health OK: model={health.model!r} loaded_indexes={health.loaded_indexes}")
            except SapBertHealthError as exc:
                print(f"ERROR: {exc}")
                return 1
        if args.mode in (None, "public"):
            import os

            if not os.environ.get("OPENAI_API_KEY"):
                print("WARNING: OPENAI_API_KEY is not set -- required for a real public/local/disabled run.")
            else:
                print("OPENAI_API_KEY: present")
        out_dir = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_validation_json(audit, sha256, out_dir / "dataset_validation.json")
        print(f"\n--validate-only OK. No LLM calls made. Wrote: {out_dir / 'dataset_validation.json'}")
        return 0

    if not args.mode:
        print("ERROR: --mode {public,local,disabled} is required for a real run (see --validate-only for a dry run)")
        return 1

    model_cfg = get_model_config(LOCKED_MODEL)
    if model_cfg.reasoning_effort != LOCKED_REASONING_EFFORT:
        import dataclasses

        model_cfg = dataclasses.replace(model_cfg, reasoning_effort=LOCKED_REASONING_EFFORT)

    try:
        pricing = get_pricing(model_cfg.model)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 1

    run_config = Scenario2RunConfig(
        model_config=model_cfg,
        retrieval_mode=args.mode,
        temperature=LOCKED_TEMPERATURE,
        seed=LOCKED_SEED,
        max_alternatives=LOCKED_MAX_ALTERNATIVES,
        sapbert_url=args.sapbert_url if args.mode == "local" else None,
    )

    _print_config_block(dataset_path=dataset_path, sha256=sha256, n=len(rows), mode=args.mode, sapbert_url=args.sapbert_url)

    sapbert_health = None
    if args.mode == "local":
        try:
            sapbert_health = check_sapbert_health(args.sapbert_url)
        except SapBertHealthError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"SapBERT health OK: model={sapbert_health.model!r} loaded_indexes={sapbert_health.loaded_indexes}")

    try:
        provider = build_provider(model_cfg.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Preflight: validating temperature/seed/reasoning_effort against the live model ...")
    try:
        run_preflight(provider, run_config.to_llm_call_config())
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Preflight OK.\n")

    distinct_ontologies = sorted({canonical_ontology(r.target_ontology) or r.target_ontology.upper() for r in rows})
    _pipeline, mappers = build_pipeline_and_mappers(provider=provider, run_config=run_config, target_ontologies=distinct_ontologies)

    start_timestamp = datetime.now(timezone.utc).isoformat()
    new_config = build_experiment_config(
        source_dataset_path=dataset_path,
        source_dataset_sha256=sha256,
        dataset_row_count=len(rows),
        provider=LOCKED_PROVIDER,
        model=model_cfg.model,
        reasoning_effort=model_cfg.reasoning_effort,
        temperature=run_config.temperature,
        temperature_mode=run_config.temperature_mode,
        seed=run_config.seed,
        retrieval_mode=args.mode,
        strict_target_ontology=STRICT_TARGET_ONTOLOGY,
        max_alternatives=run_config.max_alternatives,
        sapbert_url=run_config.sapbert_url,
        sapbert_health=sapbert_health,
        repo_dir=REPO_DIR,
        start_timestamp=start_timestamp,
    )
    new_config["limit"] = args.limit

    resume_row_ids: set[int] = set()
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"ERROR: --resume directory does not exist: {output_dir}")
            return 1
        existing_config = load_experiment_config(output_dir / "experiment_config.json")
        if existing_config is None:
            print(f"ERROR: no experiment_config.json found in {output_dir} -- cannot resume")
            return 1
        try:
            validate_resume(existing_config, new_config)
        except ResumeConfigMismatchError as exc:
            print(f"ERROR: {exc}")
            return 1
        prior_row_count = len(read_existing_predictions(output_dir / "predictions.csv"))
        resume_row_ids = quarantine_error_rows_for_resume(
            output_dir,
            resume_timestamp=datetime.now(timezone.utc).isoformat(),
            provider=LOCKED_PROVIDER,
            model=model_cfg.model,
        )
        quarantined = prior_row_count - len(resume_row_ids)
        print(f"Resuming: {len(resume_row_ids)} row(s) already completed in {output_dir}")
        if quarantined > 0:
            print(f"Resuming: {quarantined} prior error row(s) moved to retry_error_history.csv and will be retried.")
    else:
        output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT / args.mode
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        output_dir = output_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)

    write_experiment_config(new_config, output_dir / "experiment_config.json")
    write_dataset_validation_json(audit, sha256, output_dir / "dataset_validation.json")
    print(f"Output directory: {output_dir}\n")

    total = len(rows)
    remaining = total - len(resume_row_ids)
    print(f"Mapping {remaining}/{total} remaining rows (retrieval_mode={args.mode}) ...")

    rows_completed = len(resume_row_ids)
    consecutive_local_errors = 0
    consecutive_any_errors = 0
    stop_reason: str | None = None
    run_start = time.perf_counter()
    with IncrementalPredictionsCsvWriter(output_dir / "predictions.csv", append=bool(args.resume)) as writer:
        for row_result in iter_run_rows(
            mappers=mappers, dataset=rows, run_config=run_config, pricing=pricing, skip_input_rows=resume_row_ids
        ):
            csv_dict = row_result_to_csv_dict(row_result)
            writer.write_row(csv_dict)
            rows_completed += 1
            # gold_rank reuses the SAME first_gold_rank value already written to
            # predictions.csv (computed once, inside row_result_to_csv_dict via
            # scenario2_metrics.score_prediction) -- never recalculated here, so
            # terminal logging can never diverge from Top-k/MRR/predictions.csv.
            print(
                f"[{rows_completed}/{total}] row={row_result.input_row} {row_result.source_variable!r} "
                f"status={row_result.mapped_status} gold_rank={csv_dict['first_gold_rank']} "
                f"latency={row_result.end_to_end_seconds:.2f}s"
                + (f" error_stage={row_result.error_stage}" if row_result.mapped_status == "error" else "")
            )

            if row_result.mapped_status in ("mapped", "unmapped"):
                consecutive_local_errors = 0
                consecutive_any_errors = 0
            elif row_result.mapped_status == "error":
                consecutive_any_errors += 1
                if args.mode == "local" and row_result.error_stage == "local_retrieval":
                    consecutive_local_errors += 1
                    if consecutive_local_errors == 1:
                        try:
                            check_sapbert_health(args.sapbert_url)
                        except SapBertHealthError as health_exc:
                            print(f"\nSapBERT health check failed immediately after a local retrieval error: {health_exc}")
                            stop_reason = "sapbert_health_recheck_failed"
                            break
                    if consecutive_local_errors >= args.max_consecutive_local_retrieval_errors:
                        print(f"\nAborting after {consecutive_local_errors} consecutive local SapBERT retrieval failures.")
                        stop_reason = "consecutive_local_retrieval_errors"
                        break
                elif consecutive_any_errors >= args.max_consecutive_errors:
                    print(f"\nAborting after {consecutive_any_errors} consecutive execution errors ({args.mode} mode).")
                    stop_reason = "consecutive_execution_errors"
                    break

    total_seconds = time.perf_counter() - run_start

    if stop_reason is not None:
        pending = sum(1 for r in read_existing_predictions(output_dir / "predictions.csv") if r.get("status") == "error")
        new_config["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        new_config["completed"] = False
        new_config["stop_reason"] = stop_reason
        new_config["rows_completed"] = rows_completed
        new_config["error_rows_pending_retry"] = pending
        new_config["total_run_seconds"] = total_seconds
        write_experiment_config(new_config, output_dir / "experiment_config.json")
        print(
            f"\n=== Scenario 2 ({args.mode}) run ABORTED (PARTIAL): {rows_completed}/{total} rows attempted, "
            f"{pending} pending retry -- {output_dir} ===\n"
            f"Resume with: uv run python scripts/run_scenario2_retrieval_ablation.py --input {args.input} "
            f"--mode {args.mode} --sapbert-url {args.sapbert_url} --resume {output_dir}"
        )
        return 1

    new_config["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    new_config["completed"] = True
    new_config["rows_completed"] = rows_completed
    new_config["total_run_seconds"] = total_seconds
    write_experiment_config(new_config, output_dir / "experiment_config.json")

    _finalize_mode_outputs(output_dir, mode=args.mode)
    print(f"\n=== Scenario 2 ({args.mode}) run complete: {rows_completed}/{total} rows -- {output_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
