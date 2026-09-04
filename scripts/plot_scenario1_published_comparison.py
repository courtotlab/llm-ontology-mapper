#!/usr/bin/env python
"""
Scenario 1 published-baseline comparison figure suite: reads three ALREADY-
COMPLETED Scenario 1 run directories (UKBB-EFO, Biomappings-EFO, OLS-EFO
full) plus a published-baseline CSV, and writes publication figures + derived
data tables under --output-dir (default
outputs/evaluation_figures/scenario1/published_comparison/).

Analysis/visualization only. Makes ZERO OpenAI/mapper/retrieval/ontology-
validator/network calls -- every input is a file already on disk. Never
modifies predictions.csv, scenario1_metrics.csv, or any other original run
artifact, and never modifies the pre-existing model-comparison or Scenario 1
figures.

Also builds the Top-1 ontology graph-relationship comparison against the
ORIGINAL text2term publication (Figures 13/14/16 -- a different source than
the controlled Top-k baseline above; see FIGURES.md "CONTROLLED TOP-K
BASELINE vs. GRAPH-RELATIONSHIP BASELINE"), unless --skip-graph-relationship
is passed.

If --text2term-data (or its default directory) is present, ALSO builds the
STRICT common-query-aligned comparison (Figures 15/15b/15c): both methods
evaluated on the identical benchmark records, classified with the identical
EFO graph evaluator. That directory must be populated FIRST via the separate,
explicitly network-touching step:

    uv run python scripts/fetch_text2term_evaluation_outputs.py

This plotting command itself makes ZERO network calls either way -- it only
reads whatever is already vendored on disk.

Every figure is saved as PNG + SVG only -- never PDF.

Usage
─────
    uv run python scripts/fetch_text2term_evaluation_outputs.py   # one-time, network step

    uv run python scripts/plot_scenario1_published_comparison.py \\
        --ols-dir outputs/evaluation/scenario1_ols_efo/2026-08-26T15-04-18Z \\
        --ukbb-dir outputs/evaluation/scenario1_ukbb_efo/2026-08-31T13-54-53Z \\
        --biomappings-dir outputs/evaluation/scenario1_biomappings_efo/2026-08-31T16-10-24Z \\
        --baselines outputs/evaluation_figures/scenario1/published_comparison/data/published_baselines_used.csv \\
        --text2term-graph-baseline outputs/evaluation_figures/scenario1/published_comparison/data/text2term_graph_relationship_baseline.csv \\
        --text2term-data data/text2term_evaluation/original_outputs

Optional:
    --output-dir outputs/evaluation_figures/scenario1/published_comparison   (default)
    --skip-graph-relationship   (skip Figures 13/14/16 and the FIGURES.md section they add)
    --skip-graph-delta          (build Figures 13/14 but omit the optional Figure 16 delta view)
    --skip-aligned-transitions  (build Figures 15/15b but omit the optional Figure 15c transition matrices)

All three run directories must have experiment_config.json "completed": true
and match the official expected N for their benchmark (UKBB-EFO=888,
Biomappings-EFO=795, OLS-EFO full=7377) -- see
llm_ontology_mapper.benchmarking.figures.published_comparison.load_official_run.
Our values are reconciled against each run's predictions.csv before any
figure is drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_ontology_mapper.benchmarking import text2term_alignment as align  # noqa: E402
from llm_ontology_mapper.benchmarking.figures import (
    graph_relationship_comparison as graphfigs,  # noqa: E402
)
from llm_ontology_mapper.benchmarking.figures import published_comparison as pubfigs  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "evaluation_figures" / "scenario1" / "published_comparison"
DEFAULT_BASELINES = DEFAULT_OUTPUT_DIR / "data" / "published_baselines_used.csv"
DEFAULT_TEXT2TERM_GRAPH_BASELINE = DEFAULT_OUTPUT_DIR / "data" / "text2term_graph_relationship_baseline.csv"
DEFAULT_TEXT2TERM_DATA_DIR = REPO_DIR / "data" / "text2term_evaluation" / "original_outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario 1 published-baseline comparison figure suite (analysis-only, zero LLM/network calls)"
    )
    parser.add_argument("--ols-dir", required=True, metavar="DIR", help="Completed OLS-EFO (full) run directory")
    parser.add_argument("--ukbb-dir", required=True, metavar="DIR", help="Completed UKBB-EFO run directory")
    parser.add_argument(
        "--biomappings-dir", required=True, metavar="DIR", help="Completed Biomappings-EFO run directory"
    )
    parser.add_argument(
        "--baselines", default=str(DEFAULT_BASELINES), metavar="CSV", help=f"Default: {DEFAULT_BASELINES}"
    )
    parser.add_argument(
        "--text2term-graph-baseline", default=str(DEFAULT_TEXT2TERM_GRAPH_BASELINE), metavar="CSV",
        help=f"Original text2term Table 1 graph-relationship values. Default: {DEFAULT_TEXT2TERM_GRAPH_BASELINE}",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument(
        "--skip-graph-relationship", action="store_true",
        help="Skip the Top-1 graph-relationship comparison (Figures 13/14/16) entirely.",
    )
    parser.add_argument(
        "--skip-graph-delta", action="store_true",
        help="Build Figures 13/14 but omit the optional descriptive delta view (Figure 16).",
    )
    parser.add_argument(
        "--text2term-data", default=str(DEFAULT_TEXT2TERM_DATA_DIR), metavar="DIR",
        help=(
            "Directory of vendored text2term-evaluation raw output files (see "
            "scripts/fetch_text2term_evaluation_outputs.py). If present, builds the strict common-query-"
            f"aligned comparison (Figures 15/15b/15c). Default: {DEFAULT_TEXT2TERM_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--skip-aligned-comparison", action="store_true",
        help="Skip the common-query-aligned comparison (Figures 15/15b/15c) even if --text2term-data is present.",
    )
    parser.add_argument(
        "--skip-aligned-transitions", action="store_true",
        help="Build Figures 15/15b but omit the optional Figure 15c paired transition matrices.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    print("=" * 78)
    print("Scenario 1 -- published-baseline comparison -- figure suite")
    print("=" * 78)
    print(f"OLS-EFO (full) dir:   {args.ols_dir}")
    print(f"UKBB-EFO dir:         {args.ukbb_dir}")
    print(f"Biomappings-EFO dir:  {args.biomappings_dir}")
    print(f"Baselines CSV:        {args.baselines}")
    print(f"Output dir:           {output_dir}")
    print("=" * 78)

    try:
        result = pubfigs.build_all(
            ols_dir=Path(args.ols_dir),
            ukbb_dir=Path(args.ukbb_dir),
            biomappings_dir=Path(args.biomappings_dir),
            baselines_path=Path(args.baselines),
            output_dir=output_dir,
        )
    except pubfigs.PublishedComparisonError as exc:
        print(f"\nERROR: {exc}")
        return 1

    print(f"\n=== Scenario 1 published-comparison figure suite complete: {result.output_dir} ===")
    for benchmark in pubfigs.BENCHMARK_ORDER:
        m = result.all_methods[benchmark]
        print(
            f"  {benchmark:18s} ours(n={m['ours'].n}) top1={m['ours'].top1:.4f}  "
            f"OM(n={m['metaharmonizer_om'].n}) top1={m['metaharmonizer_om'].top1:.4f}  "
            f"t2t(n={m['text2term'].n}) top1={m['text2term'].top1:.4f}"
        )

    if not args.skip_graph_relationship:
        print("\n" + "=" * 78)
        print("Scenario 1 -- Top-1 graph-relationship comparison vs. original text2term")
        print("=" * 78)

        text2term_data_dir = Path(args.text2term_data)
        use_aligned = (not args.skip_aligned_comparison) and text2term_data_dir.exists()
        if not args.skip_aligned_comparison and not text2term_data_dir.exists():
            print(
                f"NOTE: {text2term_data_dir} not found -- skipping the common-query-aligned comparison "
                "(Figures 15/15b/15c). Run scripts/fetch_text2term_evaluation_outputs.py first to enable it."
            )

        try:
            graph_result = graphfigs.build_all(
                ols_dir=Path(args.ols_dir),
                ukbb_dir=Path(args.ukbb_dir),
                biomappings_dir=Path(args.biomappings_dir),
                text2term_baseline_path=Path(args.text2term_graph_baseline),
                output_dir=output_dir,
                figures_md_path=output_dir / "FIGURES.md",
                generate_delta=not args.skip_graph_delta,
                text2term_data_dir=text2term_data_dir if use_aligned else None,
                generate_15c=not args.skip_aligned_transitions,
            )
        except (graphfigs.GraphComparisonError, align.AlignmentError) as exc:
            print(f"\nERROR: {exc}")
            return 1
        for benchmark in pubfigs.BENCHMARK_ORDER:
            our = graph_result.data.our[benchmark]
            t2t = graph_result.data.text2term[benchmark]
            print(
                f"  {benchmark:18s} ours(n={our.n}, mapped={our.mapped_count}, no_top1={our.no_top1_count})  "
                f"t2t(n={t2t.n}, published)"
            )
        if graph_result.alignment_results is not None:
            print("\n  --- common-query-aligned ---")
            for benchmark in pubfigs.BENCHMARK_ORDER:
                r = graph_result.alignment_results[benchmark]
                quality = align.alignment_quality_label(r.match_rate_ours)
                print(
                    f"  {benchmark:18s} aligned_n={r.strict_matched_n} "
                    f"match_rate_ours={r.match_rate_ours:.1%} quality={quality}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
