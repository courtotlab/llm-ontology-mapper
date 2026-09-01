#!/usr/bin/env python
"""
Scenario 2 (retrieval-mode ablation) figure suite: reads three ALREADY-
COMPLETED public/local/disabled run directories and writes publication
figures + derived data tables under --output-dir (default
outputs/evaluation_figures/scenario2/).

Analysis/visualization only. Makes ZERO OpenAI/mapper/retrieval/ontology-
validator/network calls -- every input is a file already on disk. Never
modifies predictions.csv or any other original run artifact.

Usage
─────
    uv run python scripts/plot_scenario2_results.py \\
        --public-dir outputs/evaluation/scenario2_retrieval_ablation/public/<ts> \\
        --local-dir outputs/evaluation/scenario2_retrieval_ablation/local/<ts> \\
        --disabled-dir outputs/evaluation/scenario2_retrieval_ablation/disabled/<ts>

Optional:
    --output-dir outputs/evaluation_figures/scenario2   (this is the default)

All three run directories must be completed (experiment_config.json
"completed": true) and mutually compatible (identical dataset SHA/N/row IDs/
source fields/golds, identical provider/model/reasoning_effort/temperature/
seed/max_alternatives/strict_target_ontology; only retrieval_mode and
mode-specific retrieval metadata may differ) -- see
llm_ontology_mapper.benchmarking.figures.scenario2.load_completed_runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_ontology_mapper.benchmarking.figures import scenario2 as s2figs  # noqa: E402
from llm_ontology_mapper.benchmarking.scenario2_compare import (  # noqa: E402
    CompareConfigMismatchError,
    CompareDatasetMismatchError,
)

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "evaluation_figures" / "scenario2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario 2 retrieval-ablation figure suite (analysis-only, zero LLM/network calls)"
    )
    parser.add_argument("--public-dir", required=True, metavar="DIR", help="Completed public-mode run directory")
    parser.add_argument("--local-dir", required=True, metavar="DIR", help="Completed local-mode run directory")
    parser.add_argument("--disabled-dir", required=True, metavar="DIR", help="Completed disabled-mode run directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Default: {DEFAULT_OUTPUT_DIR}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    print("=" * 78)
    print("Scenario 2 -- retrieval-mode ablation -- figure suite")
    print("=" * 78)
    print(f"Public dir:    {args.public_dir}")
    print(f"Local dir:     {args.local_dir}")
    print(f"Disabled dir:  {args.disabled_dir}")
    print(f"Output dir:    {output_dir}")
    print("=" * 78)

    try:
        result = s2figs.build_all(
            public_dir=Path(args.public_dir),
            local_dir=Path(args.local_dir),
            disabled_dir=Path(args.disabled_dir),
            output_dir=output_dir,
        )
    except (
        s2figs.ScenarioCompatibilityError,
        s2figs.ReconciliationError,
        CompareDatasetMismatchError,
        CompareConfigMismatchError,
    ) as exc:
        print(f"\nERROR: {exc}")
        return 1

    print(f"\n=== Scenario 2 figure suite complete: {result.output_dir} ===")
    for mode in s2figs.MODES:
        print(f"  {mode:10s} n={result.mode_summaries[mode].get('n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
