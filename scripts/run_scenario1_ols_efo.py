#!/usr/bin/env python
"""
Scenario 1 (OLS-EFO) benchmark runner: llm-ontology-mapper, local SapBERT
retrieval, non-strict EFO target ontology.

Usage
─────
Validate only (no LLM calls, no paid mapping):
    uv run python scripts/run_scenario1_ols_efo.py \\
        --dataset OLS-EFO_full.csv --validate-only

Full run:
    export OPENAI_API_KEY="..."
    uv run python scripts/run_scenario1_ols_efo.py \\
        --dataset OLS-EFO_full.csv \\
        --provider openai --model gpt-5.6-luna --reasoning-effort low \\
        --sapbert-url http://localhost:8765

Small smoke test (first N unique queries only):
    uv run python scripts/run_scenario1_ols_efo.py \\
        --dataset OLS-EFO_full.csv --provider openai --model gpt-5.6-luna \\
        --reasoning-effort low --limit 10

Resume an interrupted run (output_dir must be an existing scenario1 run dir):
    uv run python scripts/run_scenario1_ols_efo.py \\
        --dataset OLS-EFO_full.csv --provider openai --model gpt-5.6-luna \\
        --reasoning-effort low --resume outputs/evaluation/scenario1_ols_efo/<timestamp>

Recompute every report from saved predictions -- no mapper calls:
    uv run python scripts/run_scenario1_ols_efo.py \\
        --evaluate-existing outputs/evaluation/scenario1_ols_efo/<timestamp>

TP-taxonomy Precision/Recall/F1 (Part 16) are fully automatic -- no manual
review is required or consulted (see scenario1_metrics.classify_tp_taxonomy_row).
manual_review_required.csv, when written, is an optional diagnostic list of
every graph-related (TP-Related) row and never gates any metric.

NON-STRICT MODE IS THE SCENARIO 1 DEFAULT AND IS NOT CONFIGURABLE HERE.
target_ontology=EFO, retrieval_mode=local, strict_target_ontology=False,
max_alternatives=4 are locked module constants -- see
llm_ontology_mapper.benchmarking.scenario1_runner.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_ontology_mapper.benchmarking.dataset import file_sha256  # noqa: E402
from llm_ontology_mapper.benchmarking.model_registry import (  # noqa: E402
    ALLOWED_MODELS,
    get_model_config,
)
from llm_ontology_mapper.benchmarking.pricing import ModelPricing, get_pricing  # noqa: E402
from llm_ontology_mapper.benchmarking.scenario1_dataset import (  # noqa: E402
    Scenario1DatasetError,
    audit_dataset,
    build_canonical_queries,
    expand_to_mapping_pairs,
    load_raw_dataset,
)
from llm_ontology_mapper.benchmarking.scenario1_graph_distance import (  # noqa: E402
    EFO_VERSION,
    PINNED_COMMIT,
    SOURCE_FILE,
    SOURCE_REPOSITORY,
    EfoGraphDataError,
    get_graph_index,
)
from llm_ontology_mapper.benchmarking.scenario1_metrics import (  # noqa: E402
    STATUS_OK,
    PredictionRecord,
    aggregate,
    aggregate_tp_taxonomy,
    build_metric_table,
    classify_tp_taxonomy_row,
    exact_only_diagnostic,
    execution_diagnostics,
    graph_relationship_distribution,
    graph_relationship_percentages,
    namespace_distribution,
    score_prediction,
)
from llm_ontology_mapper.benchmarking.scenario1_output import (  # noqa: E402
    IncrementalPredictionsCsvWriter,
    ResumeConfigMismatchError,
    build_experiment_config,
    csv_row_to_prediction_record,
    load_experiment_config,
    quarantine_error_rows_for_resume,
    read_existing_predictions,
    read_published_baselines,
    row_to_csv_dict,
    validate_resume,
    write_dataset_validation_json,
    write_execution_diagnostics_csv,
    write_experiment_config,
    write_graph_distance_rows_csv,
    write_graph_distance_summary_csv,
    write_graph_reference_metadata,
    write_manual_review_required_csv,
    write_mapping_pair_expanded_csv,
    write_metric_table_csv,
    write_metric_table_md,
    write_namespace_distribution_csv,
    write_published_comparison,
    write_telemetry_summary_csv,
    write_unique_queries_csv,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (  # noqa: E402
    ERROR_STAGE_LOCAL_RETRIEVAL,
    RETRIEVAL_MODE,
    STRICT_TARGET_ONTOLOGY,
    TARGET_ONTOLOGY,
    PreflightError,
    SapBertHealthError,
    Scenario1RunConfig,
    build_mapper,
    build_provider,
    check_sapbert_health,
    describe_temperature,
    iter_predictions,
    run_preflight,
)

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo"
DEFAULT_PUBLISHED_BASELINES = REPO_DIR / "published_baselines.csv"
_MAX_RANK = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 1 (OLS-EFO) llm-ontology-mapper benchmark")
    parser.add_argument("--dataset", help="Path to OLS-EFO_full.csv (required unless --evaluate-existing)")
    parser.add_argument("--provider", default="openai", choices=["openai"])
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), help="Reviewed model (see model_registry)")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sapbert-url", default="http://localhost:8765")
    parser.add_argument("--output-root", default=None, help=f"Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--resume", default=None, metavar="OUTPUT_DIR", help="Resume an existing run directory")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-existing", default=None, metavar="OUTPUT_DIR")
    parser.add_argument("--published-baselines", default=str(DEFAULT_PUBLISHED_BASELINES))
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N unique queries")
    parser.add_argument(
        "--max-consecutive-local-retrieval-errors",
        type=int,
        default=3,
        help=(
            "Abort the run after this many consecutive local SapBERT "
            "retrieval infrastructure errors (checkpointing the last error "
            "row first). Existing mapped/unmapped rows stay valid; error "
            "rows are retried automatically on --resume. Must be >= 1."
        ),
    )
    return parser.parse_args(argv)


def _print_config_summary(
    *,
    dataset_path: Path,
    sha256: str,
    unique_query_count: int,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    temperature: float | None,
    seed: int,
    sapbert_url: str,
) -> None:
    print("=" * 78)
    print("Scenario 1 -- OLS-EFO -- llm-ontology-mapper")
    print("=" * 78)
    print(f"Dataset:                  {dataset_path}")
    print(f"Dataset SHA256:           {sha256}")
    print(f"Unique queries:           {unique_query_count}")
    print(f"Target ontology:          {TARGET_ONTOLOGY}")
    print(f"Retrieval mode:           {RETRIEVAL_MODE}")
    print(f"Strict target ontology:   {STRICT_TARGET_ONTOLOGY}")
    print("Max alternatives:         4")
    print(f"Provider:                 {provider}")
    print(f"Model:                    {model}")
    print(f"Reasoning effort:         {reasoning_effort or 'N/A'}")
    print(f"Temperature:              {describe_temperature(temperature)}")
    print(f"Seed:                     {seed}")
    print(f"SapBERT URL:              {sapbert_url}")
    print("=" * 78)


def _load_and_audit(dataset_path: Path):  # type: ignore[no-untyped-def]
    df = load_raw_dataset(dataset_path)
    audit = audit_dataset(df)
    canonical_queries = build_canonical_queries(df)
    mapping_pairs = expand_to_mapping_pairs(canonical_queries)
    return df, audit, canonical_queries, mapping_pairs


def _finalize_outputs(
    *,
    output_dir: Path,
    canonical_queries: list,
    mapping_pairs: list,
    published_baselines_path: Path,
) -> None:
    """Recompute every report file from predictions.csv already on disk.
    Zero mapper/LLM calls. Shared by the end of a full run and
    --evaluate-existing so both paths stay in sync."""
    predictions_path = output_dir / "predictions.csv"
    csv_rows = read_existing_predictions(predictions_path)
    csv_rows_by_query_id = {int(r["query_id"]): r for r in csv_rows}

    records: list[PredictionRecord] = [csv_row_to_prediction_record(r) for r in csv_rows]
    row_metrics_list = [score_prediction(r) for r in records]
    agg = aggregate(row_metrics_list)

    # Fails loudly (EfoGraphDataError propagates to main()) if the EFO
    # hierarchy reference data is missing or checksum-mismatched -- graph
    # metrics are never silently degraded to NOT_EVALUATED.
    graph_index = get_graph_index()
    write_graph_reference_metadata(graph_index, output_dir / "graph_reference_metadata.json")

    query_text_by_id = {cq.query_id: cq.source_query for cq in canonical_queries}
    graph_rows = []
    graph_relationships: list[str] = []
    tp_taxonomy_rows = []
    manual_review_rows = []
    for rec in records:
        g = graph_index.classify(rec.rank1_code, list(rec.gold_codes))
        src_row = csv_rows_by_query_id.get(rec.query_id, {})
        graph_rows.append((rec.query_id, query_text_by_id.get(rec.query_id, rec.query), src_row.get("mapped_term"), g))
        graph_relationships.append(g.graph_relationship)

        tp_row = classify_tp_taxonomy_row(
            query_id=rec.query_id,
            status=rec.status,
            rank1_code=rec.rank1_code,
            gold_codes=rec.gold_codes,
            graph_relationship=g.graph_relationship,
        )
        tp_taxonomy_rows.append(tp_row)

        # manual_review_required.csv is diagnostic-only (Part 15 update): it
        # is generated for optional human spot-checking of every graph-related
        # (TP-Related) row, but nothing reads it back -- it does not gate
        # classify_tp_taxonomy_row or any metric below.
        if g.graph_relationship in {"More Specific", "More General", "Sibling"}:
            manual_review_rows.append(
                {
                    "query_id": rec.query_id,
                    "query": query_text_by_id.get(rec.query_id, rec.query),
                    "predicted_code": rec.rank1_code,
                    "predicted_label": src_row.get("mapped_term"),
                    "predicted_ontology": src_row.get("mapped_ontology"),
                    "gold_codes": "|".join(rec.gold_codes),
                    "gold_labels": src_row.get("gold_labels"),
                    "graph_relationship": g.graph_relationship,
                    "graph_matched_gold_code": g.graph_matched_gold_code,
                }
            )

    # Graph-distance classification (% Same/More Specific/More General/Sibling/
    # Unrelated) and the TP-taxonomy Precision/Recall/F1 derived from it are
    # BOTH fully automatic (Part 12/15/16) -- neither depends on manual review.
    tp_result = aggregate_tp_taxonomy(tp_taxonomy_rows)
    metric_rows = build_metric_table(
        unique_query_agg=agg,
        graph_relationship_pcts=graph_relationship_percentages(graph_relationships, denominator=len(records)),
        graph_status=STATUS_OK,
        tp_result=tp_result,
    )
    diagnostic = exact_only_diagnostic(tp_taxonomy_rows)

    notes = [
        f"Denominator: {agg.n} unique queries (see Part 4/21 -- unique-query is the PRIMARY denominator).",
        f"Recall@GT computed over {agg.recall_at_gt_n}/{agg.n} rows with a defined gold set.",
        (
            f"Graph-distance classification: EFO v{EFO_VERSION} hierarchy "
            f"({SOURCE_REPOSITORY} @ {PINNED_COMMIT[:12]}, {SOURCE_FILE}); fully automatic, "
            "computed for every row -- no manual review required for these percentages."
        ),
        (
            "TP-taxonomy Precision/Recall/F1 (Part 16) are fully automatic and require no "
            "manual review: Same -> TP-Identical; More Specific/More General/Sibling -> "
            "TP-Related; Unrelated -> FP-Error; unmapped or execution-error rows with a "
            "gold mapping present -> FN (execution errors get zero TP-taxonomy credit, "
            "same as a genuine unmapped row -- see execution_diagnostics.csv for the "
            "separate mapped/unmapped/error rate accounting). manual_review_required.csv "
            "below, if non-empty, is an optional diagnostic list of every TP-Related row "
            "for human spot-checking -- it is never consulted by this computation."
        ),
    ]
    notes.append(
        "Exact-only diagnostic (NEVER the official TP-taxonomy result): treats every "
        f"TP-Related row as FP-Error instead -- precision={diagnostic['precision']:.4f}, "
        f"recall={diagnostic['recall']:.4f}, f1={diagnostic['f1']:.4f}"
    )

    write_metric_table_csv(metric_rows, output_dir / "scenario1_metrics.csv")
    write_metric_table_md(metric_rows, output_dir / "scenario1_metrics.md", notes=notes)

    write_graph_distance_rows_csv(graph_rows, output_dir / "graph_distance_rows.csv")
    write_graph_distance_summary_csv(
        graph_relationship_distribution(graph_relationships), len(records), output_dir / "graph_distance_summary.csv"
    )

    write_namespace_distribution_csv(namespace_distribution(records), output_dir / "namespace_distribution.csv")
    write_manual_review_required_csv(manual_review_rows, output_dir / "manual_review_required.csv")
    write_execution_diagnostics_csv(execution_diagnostics(records), output_dir / "execution_diagnostics.csv")
    write_telemetry_summary_csv(csv_rows, output_dir / "telemetry_summary.csv")

    used_query_ids = {r.query_id for r in records}
    used_pairs = [p for p in mapping_pairs if p.query_id in used_query_ids]
    write_mapping_pair_expanded_csv(used_pairs, csv_rows_by_query_id, output_dir / "mapping_pair_expanded_predictions.csv")

    baselines = read_published_baselines(published_baselines_path)
    write_published_comparison(
        baselines, metric_rows, output_dir / "published_comparison.csv", output_dir / "published_comparison.md"
    )

    print(f"\nReports written to: {output_dir}")
    for row in metric_rows:
        value = f"{row.value:.4f}" if isinstance(row.value, float) else row.value
        print(f"  {row.metric:20s} {value}  (N={row.denominator}, unit={row.evaluation_unit}, status={row.status})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.max_consecutive_local_retrieval_errors < 1:
        print(
            "ERROR: --max-consecutive-local-retrieval-errors must be >= 1 "
            f"(got {args.max_consecutive_local_retrieval_errors}). The guard "
            "cannot be disabled via 0 -- pass a large value instead if you "
            "explicitly want to tolerate many consecutive SapBERT failures."
        )
        return 1

    if args.evaluate_existing:
        output_dir = Path(args.evaluate_existing)
        if not output_dir.exists():
            print(f"ERROR: --evaluate-existing directory does not exist: {output_dir}")
            return 1
        config = load_experiment_config(output_dir / "experiment_config.json")
        if config is None:
            print(f"ERROR: no experiment_config.json found in {output_dir}")
            return 1
        dataset_path = Path(config["source_dataset_path"])
        _df, _audit, canonical_queries, mapping_pairs = _load_and_audit(dataset_path)

        try:
            _finalize_outputs(
                output_dir=output_dir,
                canonical_queries=canonical_queries,
                mapping_pairs=mapping_pairs,
                published_baselines_path=Path(args.published_baselines),
            )
        except EfoGraphDataError as exc:
            print(f"ERROR: {exc}")
            return 1
        return 0

    if not args.dataset:
        print("ERROR: --dataset is required (unless using --evaluate-existing)")
        return 1

    dataset_path = Path(args.dataset)
    try:
        df, audit, canonical_queries, mapping_pairs = _load_and_audit(dataset_path)
    except Scenario1DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    sha256 = file_sha256(dataset_path)

    print("Dataset audit (Part 2, derived from the actual file -- nothing forced):")
    for key, value in audit.to_dict().items():
        print(f"  {key}: {value}")
    print(f"  canonical unique-query count (Part 3): {len(canonical_queries)}")

    if audit.max_gold_codes_per_query > _MAX_RANK:
        affected = sum(1 for cq in canonical_queries if cq.gold_count > _MAX_RANK)
        print(
            f"\nERROR: {affected} quer(ies) have more than {_MAX_RANK} gold codes "
            f"(max={audit.max_gold_codes_per_query}). Recall@GT cannot be computed exactly "
            "against a 5-slot ranked prediction without silently truncating (Part 12). "
            "Refusing to launch. Increase max_alternatives/ranked capacity or adjust the "
            "Recall@GT definition before proceeding."
        )
        return 1

    if args.limit is not None:
        canonical_queries = canonical_queries[: args.limit]
        used_ids = {cq.query_id for cq in canonical_queries}
        mapping_pairs = [p for p in mapping_pairs if p.query_id in used_ids]

    dataset_validation_path_note = "dataset_validation.json will be written to the output directory."

    try:
        sapbert_health = check_sapbert_health(args.sapbert_url)
    except SapBertHealthError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"\nSapBERT health OK: model={sapbert_health.model!r} loaded_indexes={sapbert_health.loaded_indexes}")

    try:
        graph_index = get_graph_index()
    except EfoGraphDataError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"EFO graph reference OK: {SOURCE_REPOSITORY} @ {PINNED_COMMIT[:12]} "
        f"(EFO v{EFO_VERSION}); edges_sha256={graph_index.edges_sha256[:12]}... "
        f"entailed_sha256={graph_index.entailed_sha256[:12]}..."
    )

    if args.validate_only:
        _print_config_summary(
            dataset_path=dataset_path,
            sha256=sha256,
            unique_query_count=len(canonical_queries),
            provider=args.provider,
            model=args.model or "N/A",
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            seed=args.seed,
            sapbert_url=args.sapbert_url,
        )
        out_dir = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_validation_json(audit, len(canonical_queries), out_dir / "dataset_validation.json")
        write_graph_reference_metadata(graph_index, out_dir / "graph_reference_metadata.json")
        print(f"\n--validate-only OK. No LLM calls made. {dataset_validation_path_note}")
        print(f"Wrote: {out_dir / 'dataset_validation.json'}")
        return 0

    if not args.model:
        print("ERROR: --model is required for a real run (see --validate-only for a dry run)")
        return 1

    try:
        model_cfg = get_model_config(args.model)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.reasoning_effort is not None:
        if not model_cfg.is_reasoning:
            print(f"ERROR: --reasoning-effort given but {model_cfg.model!r} is not a reasoning model")
            return 1
        model_cfg = dataclasses.replace(model_cfg, reasoning_effort=args.reasoning_effort)

    pricing: ModelPricing | None
    try:
        pricing = get_pricing(model_cfg.model)
    except KeyError as exc:
        print(f"WARNING: {exc}\nProceeding without cost telemetry (api_cost_usd will be null).")
        pricing = None

    run_config = Scenario1RunConfig(
        model_config=model_cfg,
        sapbert_url=args.sapbert_url,
        temperature=args.temperature,
        seed=args.seed,
    )

    _print_config_summary(
        dataset_path=dataset_path,
        sha256=sha256,
        unique_query_count=len(canonical_queries),
        provider=args.provider,
        model=model_cfg.model,
        reasoning_effort=model_cfg.reasoning_effort,
        temperature=run_config.temperature,
        seed=run_config.seed,
        sapbert_url=args.sapbert_url,
    )

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

    start_timestamp = datetime.now(timezone.utc).isoformat()
    new_config = build_experiment_config(
        source_dataset_path=dataset_path,
        source_dataset_sha256=sha256,
        raw_row_count=audit.raw_row_count,
        unique_mapping_pair_count=audit.unique_mapping_pair_count,
        unique_query_count=len(canonical_queries),
        provider=args.provider,
        model=model_cfg.model,
        reasoning_effort=model_cfg.reasoning_effort,
        temperature=run_config.temperature,
        temperature_mode=run_config.temperature_mode,
        seed=run_config.seed,
        target_ontology=TARGET_ONTOLOGY,
        retrieval_mode=RETRIEVAL_MODE,
        strict_target_ontology=STRICT_TARGET_ONTOLOGY,
        max_alternatives=run_config.max_alternatives,
        sapbert_url=args.sapbert_url,
        sapbert_health=sapbert_health,
        repo_dir=REPO_DIR,
        start_timestamp=start_timestamp,
    )
    new_config["limit"] = args.limit
    new_config["max_consecutive_local_retrieval_errors"] = args.max_consecutive_local_retrieval_errors

    resume_query_ids: set[int] = set()
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
        # Quarantine prior status="error" rows (e.g. from a SapBERT outage
        # that tripped the consecutive-failure guard below): they are moved
        # to retry_error_history.csv and stripped from the canonical
        # predictions.csv so they are retried below rather than skipped
        # forever. Only status="mapped"/"unmapped" rows count as completed.
        prior_row_count = len(read_existing_predictions(output_dir / "predictions.csv"))
        resume_query_ids = quarantine_error_rows_for_resume(
            output_dir,
            resume_timestamp=datetime.now(timezone.utc).isoformat(),
            provider=args.provider,
            model=model_cfg.model,
        )
        quarantined_count = prior_row_count - len(resume_query_ids)
        print(f"Resuming: {len(resume_query_ids)} query/queries already completed in {output_dir}")
        if quarantined_count > 0:
            print(
                f"Resuming: {quarantined_count} prior error row(s) moved to "
                f"retry_error_history.csv and will be retried this run."
            )
    else:
        output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        output_dir = output_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=False)

    write_experiment_config(new_config, output_dir / "experiment_config.json")
    write_dataset_validation_json(audit, len(canonical_queries), output_dir / "dataset_validation.json")
    write_unique_queries_csv(canonical_queries, output_dir / "unique_queries.csv")
    print(f"Output directory: {output_dir}\n")

    mapper = build_mapper(provider=provider, run_config=run_config)

    total = len(canonical_queries)
    remaining = total - len(resume_query_ids)
    print(f"Mapping {remaining}/{total} remaining unique queries (local SapBERT, strict=False, EFO) ...")

    rows_completed = len(resume_query_ids)
    max_consecutive_local_errors = args.max_consecutive_local_retrieval_errors
    consecutive_local_errors = 0
    stop_reason: str | None = None
    run_start = time.perf_counter()
    with IncrementalPredictionsCsvWriter(output_dir / "predictions.csv", append=bool(args.resume)) as writer:
        for row in iter_predictions(
            mapper=mapper, canonical_queries=canonical_queries, pricing=pricing, skip_query_ids=resume_query_ids
        ):
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
            # Checkpoint the row FIRST -- the consecutive-failure guard below
            # must never lose the very row that tripped it.
            writer.write_row(row_to_csv_dict(row, row_metrics=row_metrics, graph=graph))
            rows_completed += 1
            print(
                f"[{rows_completed}/{total}] query_id={row.query_id} status={row.status} "
                f"gold_rank={row_metrics.gold_rank} latency={row.end_to_end_seconds:.2f}s"
                + (f" error_stage={row.error_stage}" if row.status == "error" else "")
            )

            # ── Consecutive local-SapBERT-retrieval-failure guard ──────────
            # mapped/unmapped are terminal scientific results -> reset.
            # A non-local error (planner/reranker/pipeline) must NOT move
            # this counter in either direction -- it is not evidence the
            # SapBERT service is down, and it must not mask a real outage
            # by resetting a streak in progress either.
            if row.status in ("mapped", "unmapped"):
                consecutive_local_errors = 0
            elif row.status == "error" and row.error_stage == ERROR_STAGE_LOCAL_RETRIEVAL:
                consecutive_local_errors += 1
                if consecutive_local_errors == 1:
                    # First failure of a new streak: one cheap /health GET
                    # (no LLM call) to fail fast instead of waiting for the
                    # full threshold when the outage is already confirmed.
                    try:
                        check_sapbert_health(args.sapbert_url)
                    except SapBertHealthError as health_exc:
                        print(
                            "\n" + "=" * 78 + "\n"
                            "SapBERT health check failed immediately after a local retrieval "
                            f"error (query_id={row.query_id}): {health_exc}\n"
                            "Aborting Scenario 1 now rather than waiting for "
                            f"{max_consecutive_local_errors} consecutive failures.\n" + "=" * 78
                        )
                        stop_reason = "sapbert_health_recheck_failed"
                        break
                if consecutive_local_errors >= max_consecutive_local_errors:
                    print(
                        "\n" + "=" * 78 + "\n"
                        f"Aborting Scenario 1 after {consecutive_local_errors} consecutive local "
                        "SapBERT retrieval failures.\n"
                        "Existing mapped/unmapped checkpoints are safe.\n"
                        "Restore the SapBERT tunnel/service and resume the same output directory.\n"
                        "Error rows will be retried automatically on --resume.\n" + "=" * 78
                    )
                    stop_reason = "consecutive_local_retrieval_errors"
                    break
            # else: status="error" but not local_retrieval -- leave the
            # streak counter untouched (neither reset nor incremented).

    total_seconds = time.perf_counter() - run_start

    if stop_reason is not None:
        pending_error_rows = sum(
            1 for r in read_existing_predictions(output_dir / "predictions.csv") if r.get("status") == "error"
        )
        new_config["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        new_config["completed"] = False
        new_config["stop_reason"] = stop_reason
        new_config["rows_completed"] = rows_completed
        new_config["error_rows_pending_retry"] = pending_error_rows
        new_config["total_run_seconds"] = total_seconds
        write_experiment_config(new_config, output_dir / "experiment_config.json")
        print(
            f"\n=== Scenario 1 run ABORTED (PARTIAL, completed=false): "
            f"{rows_completed}/{total} rows attempted, {pending_error_rows} row(s) pending retry -- "
            f"{output_dir} ===\n"
            f"Resume with: uv run python scripts/run_scenario1_ols_efo.py --dataset {args.dataset} "
            f"--provider {args.provider} --model {model_cfg.model} "
            + (f"--reasoning-effort {model_cfg.reasoning_effort} " if model_cfg.reasoning_effort else "")
            + f"--sapbert-url {args.sapbert_url} --resume {output_dir}"
        )
        return 1

    new_config["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    new_config["completed"] = True
    new_config["rows_completed"] = rows_completed
    new_config["total_run_seconds"] = total_seconds
    write_experiment_config(new_config, output_dir / "experiment_config.json")

    try:
        _finalize_outputs(
            output_dir=output_dir,
            canonical_queries=canonical_queries,
            mapping_pairs=mapping_pairs,
            published_baselines_path=Path(args.published_baselines),
        )
    except EfoGraphDataError as exc:
        print(
            f"ERROR: predictions.csv was written successfully ({rows_completed}/{total} rows) "
            f"but report generation failed: {exc}\nRerun with --evaluate-existing {output_dir} "
            "once the EFO graph reference data is available -- no mapping will be repeated."
        )
        return 1
    print(f"\n=== Scenario 1 run complete: {rows_completed}/{total} rows -- {output_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
