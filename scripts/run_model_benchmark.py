#!/usr/bin/env python
"""
Reproducible per-model benchmark runner for the planned public pipeline.

One invocation runs exactly ONE model across the entire dict_mapped_all.xlsx
dataset, TWICE, using identical configuration for both runs, then writes raw
+ summary outputs and exits. It never loops over other models -- see
llm_ontology_mapper.benchmarking.model_registry.ALLOWED_MODELS for the
explicit, reviewed model list.

Usage:
    export OPENAI_API_KEY="..."
    export LOINC_USERNAME="..."
    export LOINC_PASSWORD="..."

    uv run python scripts/run_model_benchmark.py \\
        --input dict_mapped_all.xlsx \\
        --model gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import dataclasses
import re
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
from llm_ontology_mapper.benchmarking.model_registry import (  # noqa: E402
    ALLOWED_MODELS,
    get_model_config,
)
from llm_ontology_mapper.benchmarking.output import (  # noqa: E402
    MODEL_SUMMARY_FIELDS,
    REPRODUCIBILITY_SUMMARY_FIELDS,
    RUN_SUMMARY_FIELDS,
    IncrementalRawCsvWriter,
    build_benchmark_config,
    build_model_summary,
    build_reproducibility_summary,
    build_run_summary,
    format_progress_line,
    write_json,
    write_single_row_csv,
)
from llm_ontology_mapper.benchmarking.pricing import get_pricing  # noqa: E402
from llm_ontology_mapper.benchmarking.runner import (  # noqa: E402
    BenchmarkRunConfig,
    PreflightError,
    RowResult,
    build_pipeline_and_mappers,
    build_provider,
    describe_temperature,
    iter_run_rows,
    run_preflight,
)
from llm_ontology_mapper.ontology_identity import canonical_ontology  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = REPO_DIR / "outputs" / "benchmarks"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible ontology-mapper benchmark for exactly one "
            "model, twice, over the full dataset. Does not loop over other "
            "models -- run the command again with a different --model."
        )
    )
    parser.add_argument("--input", required=True, help="Path to dict_mapped_all.xlsx")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(ALLOWED_MODELS),
        help="Exactly one reviewed benchmark model (see model_registry.ALLOWED_MODELS)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Locked default: provider default (omitted from the request). "
            "Pass an explicit value to override; an unsupported explicit "
            "value still fails preflight rather than being silently dropped."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Locked default: 42")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Override reasoning effort for reasoning-capable models only "
            "(locked default: low). Rejected for gpt-4.1-mini."
        ),
    )
    return parser.parse_args(argv)


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def resolve_output_dir(base: str | None, model: str) -> Path:
    base_path = Path(base) if base else DEFAULT_OUTPUT_BASE
    model_dir = base_path / _slugify(model)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    candidate = model_dir / timestamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = model_dir / f"{timestamp}-{suffix}"
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model_cfg = get_model_config(args.model)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.reasoning_effort is not None:
        if not model_cfg.is_reasoning:
            print(
                f"ERROR: --reasoning-effort was given but {model_cfg.model!r} is not "
                "a reasoning model; refusing to pretend it supports reasoning_effort."
            )
            return 1
        model_cfg = dataclasses.replace(model_cfg, reasoning_effort=args.reasoning_effort)

    try:
        dataset = load_dataset(args.input)
    except BenchmarkDatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        pricing = get_pricing(model_cfg.model)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 1

    run_config = BenchmarkRunConfig(
        model_config=model_cfg,
        temperature=args.temperature,
        seed=args.seed,
        retrieval_mode="public",
        max_alternatives=4,
    )

    try:
        provider = build_provider(model_cfg.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    llm_call_config = run_config.to_llm_call_config()
    print(
        f"Preflight: validating temperature={describe_temperature(run_config.temperature)}, "
        f"seed={run_config.seed}, reasoning_effort={model_cfg.reasoning_effort!r} "
        f"against model={model_cfg.model!r} ..."
    )
    try:
        run_preflight(provider, llm_call_config)
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Preflight OK -- launching benchmark.\n")

    distinct_ontologies = sorted(
        {canonical_ontology(r.target_ontology) or r.target_ontology.upper() for r in dataset}
    )
    _pipeline, mappers = build_pipeline_and_mappers(
        provider=provider,
        run_config=run_config,
        target_ontologies=distinct_ontologies,
    )

    output_dir = resolve_output_dir(args.output_dir, model_cfg.model)
    output_dir.mkdir(parents=True, exist_ok=False)

    start_timestamp = datetime.now(timezone.utc).isoformat()
    config = build_benchmark_config(
        input_path=Path(args.input),
        input_file_hash=file_sha256(args.input),
        model=model_cfg.model,
        provider="openai",
        runs=2,
        temperature=run_config.temperature,
        seed=run_config.seed,
        reasoning_effort=model_cfg.reasoning_effort,
        retrieval_mode=run_config.retrieval_mode,
        max_alternatives=run_config.max_alternatives,
        pricing=pricing,
        repo_dir=REPO_DIR,
        start_timestamp=start_timestamp,
    )
    write_json(config, output_dir / "benchmark_config.json")
    print(f"Output directory: {output_dir}\n")

    run_summaries: list[dict] = []
    all_run_rows: list[list[RowResult]] = []

    for run_number in (1, 2):
        print(f"=== Run {run_number}/2 : {model_cfg.model} : {len(dataset)} rows ===")
        run_start = time.perf_counter()
        rows: list[RowResult] = []
        raw_path = output_dir / f"run_{run_number}_raw.csv"
        with IncrementalRawCsvWriter(raw_path) as writer:
            for i, row_result in enumerate(
                iter_run_rows(
                    mappers=mappers,
                    dataset=dataset,
                    run_number=run_number,
                    run_config=run_config,
                    pricing=pricing,
                ),
                start=1,
            ):
                writer.write_row(row_result)
                rows.append(row_result)
                print(format_progress_line(row_result, index=i, total=len(dataset)))
        total_run_seconds = time.perf_counter() - run_start

        error_count = sum(1 for r in rows if r.mapped_status == "error")
        run_complete = error_count == 0
        summary = build_run_summary(
            model=model_cfg.model,
            run_number=run_number,
            run_complete=run_complete,
            rows=rows,
        )
        summary["total_run_seconds"] = total_run_seconds
        write_single_row_csv(summary, output_dir / f"run_{run_number}_summary.csv", RUN_SUMMARY_FIELDS)

        print(f"\n--- Run {run_number} summary ({'COMPLETE' if run_complete else 'INCOMPLETE'}) ---")
        for key in (
            "rows_evaluated",
            "error_count",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "top1_accuracy",
            "top5_hit_rate",
            "total_run_seconds",
            "total_api_cost_usd",
        ):
            print(f"  {key}: {summary[key]}")
        if not run_complete:
            print(
                f"  WARNING: run_complete=False -- {error_count} execution error(s). "
                "Metrics above are PROVISIONAL (evaluated rows only)."
            )
        print()

        run_summaries.append(summary)
        all_run_rows.append(rows)

    reproducibility = build_reproducibility_summary(
        model=model_cfg.model,
        rows_run1=all_run_rows[0],
        rows_run2=all_run_rows[1],
        run1_summary=run_summaries[0],
        run2_summary=run_summaries[1],
    )
    write_single_row_csv(
        reproducibility, output_dir / "reproducibility_summary.csv", REPRODUCIBILITY_SUMMARY_FIELDS
    )

    model_summary = build_model_summary(
        model=model_cfg.model,
        run1_summary=run_summaries[0],
        run2_summary=run_summaries[1],
        reproducibility=reproducibility,
    )
    write_single_row_csv(model_summary, output_dir / "model_summary.csv", MODEL_SUMMARY_FIELDS)

    print(f"=== Model benchmark complete: {model_cfg.model} ===")
    for key in MODEL_SUMMARY_FIELDS:
        print(f"  {key}: {model_summary[key]}")
    print(f"\nOutput directory: {output_dir}")

    if not (run_summaries[0]["run_complete"] and run_summaries[1]["run_complete"]):
        print(
            "\nWARNING: one or more runs had execution errors (run_complete=False). "
            "This is not a fully comparable final benchmark."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
