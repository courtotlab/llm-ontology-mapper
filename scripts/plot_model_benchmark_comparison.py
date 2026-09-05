#!/usr/bin/env python3
"""Publication-quality benchmark comparison for the four canonical model runs.

Analysis/visualization only: this script reads existing files under
outputs/benchmarks/<model>/<timestamp>/ and writes figures/tables under
outputs/benchmark_figures/. It never modifies a source benchmark file, never
calls an LLM/API, and never re-runs the benchmark.

Execution-error handling (see module docstring of
src/llm_ontology_mapper/benchmarking/scoring.py for the locked scoring
contract this mirrors): the benchmark's own summaries drop status="error"
rows out of both the numerator and denominator of weighted precision/recall/F1
and top1/top5 rates (rows_evaluated < rows_total). For the primary "end-to-end"
(e2e) metrics in this script, every row of the original 218-row dataset stays
in the denominator: an execution error is scored exactly like "unmapped with
a gold code present" already is in the locked contract -- TP=0, FP=0, FN=1,
top1=0, top5=0. Original reported metrics are preserved unchanged alongside
the recomputed e2e metrics, suffixed accordingly, everywhere both appear.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "outputs" / "benchmarks"
OUT_ROOT = REPO_ROOT / "outputs" / "benchmark_figures"
MAIN_DIR = OUT_ROOT / "main"
SUPP_DIR = OUT_ROOT / "supplementary"
DATA_DIR = OUT_ROOT / "data"

# --------------------------------------------------------------------------
# Model identity: order, display names, colorblind-safe colors (Okabe-Ito)
# --------------------------------------------------------------------------

MODEL_ORDER = ["gpt-4.1-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.6-luna"]

DISPLAY_NAMES = {
    "gpt-4.1-mini": "GPT-4.1 mini",
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.6-luna": "GPT-5.6 Luna",
}

COLORS = {
    "gpt-4.1-mini": "#D55E00",  # vermillion
    "gpt-5-mini": "#009E73",  # bluish green
    "gpt-5.4-mini": "#E69F00",  # orange
    "gpt-5.6-luna": "#0072B2",  # blue
}

# Canonical run selector: (start_timestamp, expected reasoning_effort)
CANONICAL_RUNS = {
    "gpt-5.6-luna": ("2026-08-21T15:17:32.852013+00:00", "low"),
    "gpt-5.4-mini": ("2026-08-21T16:02:32.757119+00:00", "low"),
    "gpt-4.1-mini": ("2026-08-21T17:05:26.221518+00:00", "N/A"),
    "gpt-5-mini": ("2026-08-21T18:24:05.610603+00:00", "low"),
}

EXPECTED_CONFIG = {
    "input_filename": "dict_mapped_all.xlsx",
    "input_file_sha256": "91980c4df28781e5ef8d33d614c4f768966fa88b3db24996f3da38fc01bbddfd",
    "runs": 2,
    "retrieval_mode": "public",
    "max_alternatives": 4,
    "seed": 42,
    "temperature_mode": "provider_default",
}

EXPECTED_ROWS_PER_RUN = 218
EXPECTED_ATTEMPTS_PER_MODEL = 436

ISSUES: list[str] = []  # reconciliation issues; reported, never silently fixed


def issue(msg: str) -> None:
    ISSUES.append(msg)
    print(f"[RECONCILIATION ISSUE] {msg}")


# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.axisbelow": True,
    }
)


def save_figure(fig: plt.Figure, name: str, main: bool) -> None:
    target_dir = MAIN_DIR if main else SUPP_DIR
    for ext in ("svg", "pdf", "png"):
        fig.savefig(target_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_usd_per_term(x: float) -> str:
    return f"${x:.5f}"


# --------------------------------------------------------------------------
# Loading + config validation
# --------------------------------------------------------------------------


def find_canonical_dir(model_key: str) -> Path:
    expected_ts, expected_reasoning = CANONICAL_RUNS[model_key]
    model_root = BENCH_ROOT / model_key
    candidates = sorted(p for p in model_root.iterdir() if p.is_dir())
    matches = []
    for cand in candidates:
        cfg_path = cand / "benchmark_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("start_timestamp") == expected_ts:
            matches.append((cand, cfg))
    if not matches:
        raise SystemExit(
            f"No benchmark directory under {model_root} has start_timestamp={expected_ts!r}"
        )
    if len(matches) > 1:
        raise SystemExit(f"Multiple directories under {model_root} match start_timestamp={expected_ts!r}")
    cand, cfg = matches[0]

    for key, expected_val in EXPECTED_CONFIG.items():
        actual_val = cfg.get(key)
        if actual_val != expected_val:
            issue(
                f"{model_key}: config field {key!r} = {actual_val!r}, expected {expected_val!r} ({cand})"
            )
    actual_reasoning = cfg.get("reasoning_effort")
    if actual_reasoning != expected_reasoning:
        issue(
            f"{model_key}: reasoning_effort = {actual_reasoning!r}, expected {expected_reasoning!r} ({cand})"
        )
    return cand


STR_COLS_FILLNA = [
    "mapped_code_normalized",
    "mapped_code",
    "alternative_1_code",
    "alternative_2_code",
    "alternative_3_code",
    "alternative_4_code",
    "error_type",
    "error_message",
]


def load_raw(path: Path, model_key: str, run_number: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in STR_COLS_FILLNA:
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["model"] = model_key
    df["run_number"] = run_number
    return df


def compute_e2e_row_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add end-to-end (e2e) tp/fp/fn/tn/top1/top5 columns.

    status == "error" is remapped to TP=0, FP=0, FN=1, top1=0, top5=0 -- the
    same tuple the locked scoring contract already uses for "unmapped, gold
    exists" -- so an execution error contributes a miss, not a silently
    dropped row. All other rows keep their original scored values unchanged.
    """
    is_error = df["mapped_status"] == "error"
    df = df.copy()
    df["e2e_tp"] = np.where(is_error, 0.0, df["tp"])
    df["e2e_fp"] = np.where(is_error, 0.0, df["fp"])
    df["e2e_fn"] = np.where(is_error, 1.0, df["fn"])
    df["e2e_tn"] = np.where(is_error, 0.0, df["tn"])
    df["e2e_top1"] = np.where(is_error, False, df["top1_correct"].astype(bool))
    df["e2e_top5"] = np.where(is_error, False, df["top5_hit"].astype(bool))
    return df


def weighted_prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate_e2e(df: pd.DataFrame) -> dict:
    tp, fp, fn = df["e2e_tp"].sum(), df["e2e_fp"].sum(), df["e2e_fn"].sum()
    precision, recall, f1 = weighted_prf(tp, fp, fn)
    n = len(df)
    return {
        "n": n,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "top1_accuracy": df["e2e_top1"].sum() / n if n else 0.0,
        "top5_hit_rate": df["e2e_top5"].sum() / n if n else 0.0,
        "execution_error_count": int((df["mapped_status"] == "error").sum()),
    }


# --------------------------------------------------------------------------
# Load all four models
# --------------------------------------------------------------------------


class ModelData:
    def __init__(self, model_key: str, run_dir: Path):
        self.model_key = model_key
        self.run_dir = run_dir
        self.config = json.loads((run_dir / "benchmark_config.json").read_text())
        self.model_summary = pd.read_csv(run_dir / "model_summary.csv").iloc[0]
        self.run_summary = {
            1: pd.read_csv(run_dir / "run_1_summary.csv").iloc[0],
            2: pd.read_csv(run_dir / "run_2_summary.csv").iloc[0],
        }
        self.reproducibility_summary = pd.read_csv(run_dir / "reproducibility_summary.csv").iloc[0]
        raw1 = load_raw(run_dir / "run_1_raw.csv", model_key, 1)
        raw2 = load_raw(run_dir / "run_2_raw.csv", model_key, 2)
        raw1 = compute_e2e_row_scores(raw1)
        raw2 = compute_e2e_row_scores(raw2)
        self.raw = {1: raw1, 2: raw2}
        self.pooled = pd.concat([raw1, raw2], ignore_index=True)


def load_all_models() -> dict[str, ModelData]:
    data = {}
    for model_key in MODEL_ORDER:
        run_dir = find_canonical_dir(model_key)
        data[model_key] = ModelData(model_key, run_dir)
    return data


# --------------------------------------------------------------------------
# Validation (section 20)
# --------------------------------------------------------------------------


def validate(data: dict[str, ModelData]) -> None:
    for model_key, md in data.items():
        for run_number, raw in md.raw.items():
            if len(raw) != EXPECTED_ROWS_PER_RUN:
                issue(
                    f"{model_key} run {run_number}: {len(raw)} raw rows, expected {EXPECTED_ROWS_PER_RUN}"
                )
        if len(md.pooled) != EXPECTED_ATTEMPTS_PER_MODEL:
            issue(
                f"{model_key}: {len(md.pooled)} pooled attempts, expected {EXPECTED_ATTEMPTS_PER_MODEL}"
            )

        # Reproduce original run-level metrics exactly when a run has zero
        # execution errors (e2e must equal originally reported in that case).
        for run_number, raw in md.raw.items():
            error_count = int((raw["mapped_status"] == "error").sum())
            summary_row = md.run_summary[run_number]
            if error_count != int(summary_row["error_count"]):
                issue(
                    f"{model_key} run {run_number}: raw error rows={error_count} != "
                    f"run_summary error_count={summary_row['error_count']}"
                )
            if error_count == 0:
                e2e = aggregate_e2e(raw)
                for metric, summary_key in [
                    ("weighted_precision", "weighted_precision"),
                    ("weighted_recall", "weighted_recall"),
                    ("weighted_f1", "weighted_f1"),
                    ("top1_accuracy", "top1_accuracy"),
                    ("top5_hit_rate", "top5_hit_rate"),
                ]:
                    if not np.isclose(e2e[metric], summary_row[summary_key], atol=1e-9):
                        issue(
                            f"{model_key} run {run_number}: recomputed e2e {metric}="
                            f"{e2e[metric]!r} != reported {summary_row[summary_key]!r}"
                        )

        # Ontology subgroup Ns must reconcile to total pooled attempts.
        ontology_n_sum = md.pooled.groupby("target_ontology").size().sum()
        if ontology_n_sum != len(md.pooled):
            issue(f"{model_key}: ontology group sizes sum to {ontology_n_sum}, expected {len(md.pooled)}")

        # Cost reconciliation: combined_api_cost_usd / 436 should equal the
        # reported mean_api_cost_per_term_usd. When a run had an execution
        # error, the benchmark's own mean divides by rows_evaluated (217)
        # for that run rather than rows_total (218), so the two will not
        # match exactly -- flagged here rather than silently accepted.
        ms = md.model_summary
        implied_mean_cost = ms["combined_api_cost_usd"] / EXPECTED_ATTEMPTS_PER_MODEL
        if not np.isclose(implied_mean_cost, ms["mean_api_cost_per_term_usd"], rtol=1e-6):
            issue(
                f"{model_key}: combined_api_cost_usd/{EXPECTED_ATTEMPTS_PER_MODEL}={implied_mean_cost!r} "
                f"!= reported mean_api_cost_per_term_usd={ms['mean_api_cost_per_term_usd']!r} "
                "(the benchmark's own mean divides by rows_evaluated, not rows_total, for any run "
                "with an execution error, so the per-term cost denominator is inconsistent with the "
                "436-attempt denominator used everywhere else in this analysis)"
            )

    print(f"Validation complete. {len(ISSUES)} issue(s) recorded." if ISSUES else "Validation complete. No issues found.")


# --------------------------------------------------------------------------
# Combined model analysis table (sections 3, 19)
# --------------------------------------------------------------------------


def build_combined_model_analysis(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        ms = md.model_summary
        e2e_run = {rn: aggregate_e2e(md.raw[rn]) for rn in (1, 2)}
        e2e_mean_precision = np.mean([e2e_run[1]["weighted_precision"], e2e_run[2]["weighted_precision"]])
        e2e_mean_recall = np.mean([e2e_run[1]["weighted_recall"], e2e_run[2]["weighted_recall"]])
        e2e_mean_f1 = np.mean([e2e_run[1]["weighted_f1"], e2e_run[2]["weighted_f1"]])
        e2e_mean_top1 = np.mean([e2e_run[1]["top1_accuracy"], e2e_run[2]["top1_accuracy"]])
        e2e_mean_top5 = np.mean([e2e_run[1]["top5_hit_rate"], e2e_run[2]["top5_hit_rate"]])
        e2e_run1_f1, e2e_run2_f1 = e2e_run[1]["weighted_f1"], e2e_run[2]["weighted_f1"]

        error_count = int((md.pooled["mapped_status"] == "error").sum())
        retry_rows = int((md.pooled["retrieval_retry_count"].fillna(0) > 0).sum())
        final_err_rows = int((md.pooled["retrieval_final_error_count"].fillna(0) > 0).sum())
        total_retries = int(md.pooled["retrieval_retry_count"].fillna(0).sum())

        top1_agree_e2e, top5_agree_e2e = e2e_reproducibility_agreement(md)

        rows.append(
            {
                "model": model_key,
                "model_display": DISPLAY_NAMES[model_key],
                "reasoning_effort": md.config.get("reasoning_effort"),
                # --- original reported metrics (rows_evaluated denominator) ---
                "mean_weighted_precision_reported": ms["mean_weighted_precision"],
                "mean_weighted_recall_reported": ms["mean_weighted_recall"],
                "mean_weighted_f1_reported": ms["mean_weighted_f1"],
                "mean_top1_accuracy_reported": ms["mean_top1_accuracy"],
                "mean_top5_hit_rate_reported": ms["mean_top5_hit_rate"],
                "run_1_weighted_f1_reported": ms["run_1_weighted_f1"],
                "run_2_weighted_f1_reported": ms["run_2_weighted_f1"],
                "weighted_f1_absolute_difference_reported": ms["weighted_f1_absolute_difference"],
                # --- end-to-end recomputed metrics (218-row denominator, error=miss) ---
                "mean_weighted_precision_e2e": e2e_mean_precision,
                "mean_weighted_recall_e2e": e2e_mean_recall,
                "mean_weighted_f1_e2e": e2e_mean_f1,
                "mean_top1_accuracy_e2e": e2e_mean_top1,
                "mean_top5_hit_rate_e2e": e2e_mean_top5,
                "run_1_weighted_f1_e2e": e2e_run1_f1,
                "run_2_weighted_f1_e2e": e2e_run2_f1,
                "weighted_f1_absolute_difference_e2e": abs(e2e_run1_f1 - e2e_run2_f1),
                # --- reproducibility ---
                "top1_exact_agreement_reported": ms["top1_exact_agreement"],
                "top5_set_agreement_reported": ms["top5_set_agreement"],
                "top1_exact_agreement_e2e_all218": top1_agree_e2e,
                "top5_set_agreement_e2e_all218": top5_agree_e2e,
                # --- cost / latency (reused verbatim from model_summary.csv; unaffected by error handling) ---
                "mean_end_to_end_seconds_per_term": ms["mean_end_to_end_seconds_per_term"],
                "mean_llm_seconds_per_term": ms["mean_llm_seconds_per_term"],
                "combined_api_cost_usd": ms["combined_api_cost_usd"],
                "mean_api_cost_per_term_usd": ms["mean_api_cost_per_term_usd"],
                # --- reliability ---
                "execution_error_count": error_count,
                "execution_error_rate": error_count / EXPECTED_ATTEMPTS_PER_MODEL,
                "total_retrieval_retries": total_retries,
                "rows_with_retrieval_retries": retry_rows,
                "rows_with_final_retrieval_errors": final_err_rows,
            }
        )
    return pd.DataFrame(rows)


def e2e_reproducibility_agreement(md: ModelData) -> tuple[float, float]:
    """Top-1 exact / top-5 set agreement over all 218 rows, treating an
    execution error as a distinct sentinel outcome (matches another error on
    the same row; never matches a real mapped code or an unmapped blank)."""
    r1 = md.raw[1].set_index("input_row")
    r2 = md.raw[2].set_index("input_row")
    common_rows = r1.index.intersection(r2.index)

    def top1_value(row):
        return "__ERROR__" if row["mapped_status"] == "error" else row["mapped_code_normalized"]

    def top5_set(row):
        if row["mapped_status"] == "error":
            return frozenset({"__ERROR__"})
        codes = [row["mapped_code_normalized"]] + [
            row[f"alternative_{i}_code"] for i in range(1, 5)
        ]
        return frozenset(c for c in codes if c)

    top1_agree = sum(top1_value(r1.loc[k]) == top1_value(r2.loc[k]) for k in common_rows)
    top5_agree = sum(top5_set(r1.loc[k]) == top5_set(r2.loc[k]) for k in common_rows)
    n = len(common_rows)
    return top1_agree / n, top5_agree / n


# --------------------------------------------------------------------------
# Ontology performance (section 10)
# --------------------------------------------------------------------------


def build_ontology_performance(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        for ontology, group in md.pooled.groupby("target_ontology"):
            tp, fp, fn = group["e2e_tp"].sum(), group["e2e_fp"].sum(), group["e2e_fn"].sum()
            precision, recall, f1 = weighted_prf(tp, fp, fn)
            n = len(group)
            rows.append(
                {
                    "model": model_key,
                    "model_display": DISPLAY_NAMES[model_key],
                    "ontology": ontology,
                    "n": n,
                    "weighted_precision": precision,
                    "weighted_recall": recall,
                    "weighted_f1": f1,
                    "top1_accuracy": group["e2e_top1"].sum() / n,
                    "top5_hit_rate": group["e2e_top5"].sum() / n,
                    "execution_error_count": int((group["mapped_status"] == "error").sum()),
                }
            )
    return pd.DataFrame(rows)


def ontology_order(ontology_perf: pd.DataFrame) -> list[str]:
    totals = ontology_perf.groupby("ontology")["n"].sum().sort_values(ascending=False)
    return list(totals.index)


# --------------------------------------------------------------------------
# Mapping outcome distribution (section 11)
# --------------------------------------------------------------------------

OUTCOME_CATEGORIES = [
    "Gold rank 1",
    "Gold rank 2",
    "Gold rank 3",
    "Gold rank 4",
    "Gold rank 5",
    "Mapped, gold not in Top 5",
    "Unmapped",
    "Execution error",
]


def classify_outcome(row) -> str:
    status = row["mapped_status"]
    if status == "error":
        return "Execution error"
    if status == "unmapped":
        return "Unmapped"
    # status == "mapped"
    rank = row["gold_rank"]
    if pd.isna(rank):
        return "Mapped, gold not in Top 5"
    return f"Gold rank {int(rank)}"


def build_outcome_distribution(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        outcomes = md.pooled.apply(classify_outcome, axis=1)
        counts = outcomes.value_counts().reindex(OUTCOME_CATEGORIES, fill_value=0)
        total = counts.sum()
        if total != EXPECTED_ATTEMPTS_PER_MODEL:
            issue(f"{model_key}: outcome categories sum to {total}, expected {EXPECTED_ATTEMPTS_PER_MODEL}")
        for cat in OUTCOME_CATEGORIES:
            rows.append(
                {
                    "model": model_key,
                    "model_display": DISPLAY_NAMES[model_key],
                    "outcome": cat,
                    "count": int(counts[cat]),
                    "percent": counts[cat] / total,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pipeline reliability (section 12)
# --------------------------------------------------------------------------


def build_pipeline_reliability(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        n = len(md.pooled)
        execution_errors = int((md.pooled["mapped_status"] == "error").sum())
        retry_rows = int((md.pooled["retrieval_retry_count"].fillna(0) > 0).sum())
        final_err_rows = int((md.pooled["retrieval_final_error_count"].fillna(0) > 0).sum())
        rows.append(
            {
                "model": model_key,
                "model_display": DISPLAY_NAMES[model_key],
                "n_attempts": n,
                "execution_error_count": execution_errors,
                "execution_errors_per_100": 100 * execution_errors / n,
                "rows_with_retrieval_retries_count": retry_rows,
                "rows_with_retrieval_retries_per_100": 100 * retry_rows / n,
                "rows_with_final_retrieval_errors_count": final_err_rows,
                "rows_with_final_retrieval_errors_per_100": 100 * final_err_rows / n,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Latency decomposition (section 9)
# --------------------------------------------------------------------------

LATENCY_ROUNDING_EPS = 0.05  # seconds; below this, treat negative "other" as float rounding


def build_latency_breakdown(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        pooled = md.pooled
        mean_e2e = pooled["end_to_end_seconds"].mean()
        mean_planner = pooled["query_planner_seconds"].mean()
        mean_retrieval = pooled["retrieval_seconds"].mean()
        mean_reranker = pooled["reranker_seconds"].mean()
        other = mean_e2e - (mean_planner + mean_retrieval + mean_reranker)
        if other < 0:
            if other > -LATENCY_ROUNDING_EPS:
                other = 0.0
            else:
                issue(
                    f"{model_key}: latency decomposition residual is substantially negative "
                    f"({other:.4f}s) -- stages sum to more than measured end-to-end time"
                )
        rows.append(
            {
                "model": model_key,
                "model_display": DISPLAY_NAMES[model_key],
                "mean_query_planner_seconds": mean_planner,
                "mean_retrieval_seconds": mean_retrieval,
                "mean_reranker_seconds": mean_reranker,
                "mean_other_seconds": other,
                "mean_end_to_end_seconds": mean_e2e,
                "mean_llm_seconds": pooled["llm_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Token usage (section 13)
# --------------------------------------------------------------------------


def build_token_usage(data: dict[str, ModelData]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_ORDER:
        md = data[model_key]
        pooled = md.pooled
        supports_reasoning = md.config.get("reasoning_effort") not in (None, "N/A")
        rows.append(
            {
                "model": model_key,
                "model_display": DISPLAY_NAMES[model_key],
                "mean_total_input_tokens": pooled["total_input_tokens"].mean(),
                "mean_total_cached_input_tokens": pooled["total_cached_input_tokens"].mean(),
                "mean_total_output_tokens": pooled["total_output_tokens"].mean(),
                "mean_total_reasoning_tokens": (
                    pooled["total_reasoning_tokens"].mean() if supports_reasoning else np.nan
                ),
                "reasoning_supported": supports_reasoning,
                "mean_planner_input_tokens": pooled["planner_input_tokens"].mean(),
                "mean_planner_output_tokens": pooled["planner_output_tokens"].mean(),
                "mean_planner_reasoning_tokens": (
                    pooled["planner_reasoning_tokens"].mean() if supports_reasoning else np.nan
                ),
                "mean_reranker_input_tokens": pooled["reranker_input_tokens"].fillna(0).mean(),
                "mean_reranker_output_tokens": pooled["reranker_output_tokens"].fillna(0).mean(),
                "mean_reranker_reasoning_tokens": (
                    pooled["reranker_reasoning_tokens"].fillna(0).mean() if supports_reasoning else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pareto analysis (section 16)
# --------------------------------------------------------------------------


def pareto_efficient_max_f1_min_x(models: list[str], f1: dict[str, float], x: dict[str, float]) -> set[str]:
    efficient = set()
    for m in models:
        dominated = False
        for other in models:
            if other == m:
                continue
            if f1[other] >= f1[m] and x[other] <= x[m] and (f1[other] > f1[m] or x[other] < x[m]):
                dominated = True
                break
        if not dominated:
            efficient.add(m)
    return efficient


def build_pareto_summary(combined: pd.DataFrame) -> pd.DataFrame:
    f1 = dict(zip(combined["model"], combined["mean_weighted_f1_e2e"]))
    cost = dict(zip(combined["model"], combined["mean_api_cost_per_term_usd"]))
    latency = dict(zip(combined["model"], combined["mean_end_to_end_seconds_per_term"]))
    models = list(combined["model"])

    cost_frontier = pareto_efficient_max_f1_min_x(models, f1, cost)
    latency_frontier = pareto_efficient_max_f1_min_x(models, f1, latency)

    rows = []
    for m in models:
        rows.append(
            {
                "model": m,
                "model_display": DISPLAY_NAMES[m],
                "mean_weighted_f1_e2e": f1[m],
                "mean_api_cost_per_term_usd": cost[m],
                "mean_end_to_end_seconds_per_term": latency[m],
                "pareto_efficient_cost_vs_f1": m in cost_frontier,
                "pareto_efficient_latency_vs_f1": m in latency_frontier,
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier_points(models: list[str], f1: dict[str, float], x: dict[str, float]) -> list[tuple[float, float]]:
    """Return frontier points sorted by x, for drawing a thin frontier line."""
    pts = sorted(((x[m], f1[m]) for m in models), key=lambda p: p[0])
    frontier = []
    best_f1 = -np.inf
    # Walk by increasing cost/latency and keep points that raise the running-max F1.
    for xv, f1v in pts:
        if f1v > best_f1:
            frontier.append((xv, f1v))
            best_f1 = f1v
    return frontier


# --------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------


def style_axis(ax):
    ax.tick_params(axis="both", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def bar_positions(n_groups: int, n_series: int, width: float = 0.8):
    group_centers = np.arange(n_groups)
    series_width = width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * series_width
    return group_centers, offsets, series_width


# --------------------------------------------------------------------------
# Figure 1: overall performance
# --------------------------------------------------------------------------


def fig_01_overall_performance(combined: pd.DataFrame):
    metrics = [
        ("mean_weighted_f1_e2e", "Weighted F1"),
        ("mean_top1_accuracy_e2e", "Top-1 accuracy"),
        ("mean_top5_hit_rate_e2e", "Top-5 hit rate"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    n_groups, n_series = len(metrics), len(MODEL_ORDER)
    centers, offsets, width = bar_positions(n_groups, n_series)

    for i, model_key in enumerate(MODEL_ORDER):
        row = combined[combined["model"] == model_key].iloc[0]
        values = [row[key] for key, _ in metrics]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=COLORS[model_key], label=DISPLAY_NAMES[model_key])
        for b, v in zip(bars, values):
            ax.annotate(pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.3)

    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("End-to-end score (all 218 mappings/run)")
    ax.set_title("Overall mapping performance, end-to-end (execution errors scored as a miss)")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "figure_01_overall_performance", main=True)


def fig_01b_precision_recall_f1(combined: pd.DataFrame):
    metrics = [
        ("mean_weighted_precision_e2e", "Weighted precision"),
        ("mean_weighted_recall_e2e", "Weighted recall"),
        ("mean_weighted_f1_e2e", "Weighted F1"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    n_groups, n_series = len(metrics), len(MODEL_ORDER)
    centers, offsets, width = bar_positions(n_groups, n_series)

    for i, model_key in enumerate(MODEL_ORDER):
        row = combined[combined["model"] == model_key].iloc[0]
        values = [row[key] for key, _ in metrics]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=COLORS[model_key], label=DISPLAY_NAMES[model_key])
        for b, v in zip(bars, values):
            ax.annotate(pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.3)

    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("End-to-end score (all 218 mappings/run)")
    ax.set_title("Weighted precision, recall, and F1, end-to-end")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "figure_01b_precision_recall_f1", main=True)


# --------------------------------------------------------------------------
# Figures 2 & 3: performance vs cost / latency
# --------------------------------------------------------------------------


def scatter_with_pareto(combined: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str,
                         title: str, x_formatter, name: str, main: bool):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    f1 = dict(zip(combined["model"], combined[y_col]))
    xvals = dict(zip(combined["model"], combined[x_col]))
    frontier = pareto_efficient_max_f1_min_x(list(combined["model"]), f1, xvals)
    frontier_pts = pareto_frontier_points(list(combined["model"]), f1, xvals)

    if len(frontier_pts) > 1:
        fx, fy = zip(*frontier_pts)
        ax.plot(fx, fy, color="#888888", linewidth=1.1, linestyle="--", zorder=1)

    # Alternate label placement above/below by x-rank so text for points
    # that are close together on the x-axis (the common collision case
    # here) lands on opposite sides; keep labels short (name only) and mark
    # Pareto-efficient points via the black outline + legend proxy instead.
    order = sorted(combined["model"], key=lambda m: combined.set_index("model").loc[m, x_col])
    for rank, model_key in enumerate(order):
        row = combined[combined["model"] == model_key].iloc[0]
        x, y = row[x_col], row[y_col]
        is_efficient = model_key in frontier
        ax.scatter(
            x, y, s=130, color=COLORS[model_key], zorder=3,
            edgecolors="black" if is_efficient else "none", linewidths=1.3 if is_efficient else 0,
        )
        dy = 10 if rank % 2 == 0 else -14
        va = "bottom" if dy > 0 else "top"
        ax.annotate(DISPLAY_NAMES[model_key], (x, y), xytext=(9, dy), textcoords="offset points",
                    fontsize=9.2, ha="left", va=va)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black",
               markersize=9, markeredgewidth=1.3, label="Pareto-efficient"),
        Line2D([0], [0], color="#888888", linewidth=1.1, linestyle="--", label="Pareto frontier"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(x_formatter)
    ax.set_title(title)
    style_axis(ax)
    save_figure(fig, name, main=main)


def fig_02_performance_vs_cost(combined: pd.DataFrame):
    scatter_with_pareto(
        combined,
        x_col="mean_api_cost_per_term_usd",
        y_col="mean_weighted_f1_e2e",
        x_label="Mean API cost per mapped term (USD)",
        y_label="Mean end-to-end weighted F1",
        title="Mapping performance vs. API cost per term",
        x_formatter=FuncFormatter(lambda v, _: fmt_usd_per_term(v)),
        name="figure_02_performance_vs_cost",
        main=True,
    )


def fig_03_performance_vs_latency(combined: pd.DataFrame):
    scatter_with_pareto(
        combined,
        x_col="mean_end_to_end_seconds_per_term",
        y_col="mean_weighted_f1_e2e",
        x_label="Mean end-to-end latency per term (seconds)",
        y_label="Mean end-to-end weighted F1",
        title="Mapping performance vs. end-to-end latency",
        x_formatter=FuncFormatter(lambda v, _: f"{v:.1f}s"),
        name="figure_03_performance_vs_latency",
        main=True,
    )


def fig_03b_performance_vs_llm_latency(combined: pd.DataFrame):
    scatter_with_pareto(
        combined,
        x_col="mean_llm_seconds_per_term",
        y_col="mean_weighted_f1_e2e",
        x_label="Mean LLM-only latency per term (seconds)",
        y_label="Mean end-to-end weighted F1",
        title="Mapping performance vs. LLM-only latency\n(excludes public retrieval latency)",
        x_formatter=FuncFormatter(lambda v, _: f"{v:.1f}s"),
        name="figure_03b_performance_vs_llm_latency",
        main=True,
    )


# --------------------------------------------------------------------------
# Figure 4: reproducibility
# --------------------------------------------------------------------------


def fig_04_reproducibility(combined: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    specs = [
        ("top1_exact_agreement_reported", "top1_exact_agreement_e2e_all218", "Top-1 exact agreement"),
        ("top5_set_agreement_reported", "top5_set_agreement_e2e_all218", "Top-5 set agreement"),
    ]
    for ax, (reported_col, e2e_col, title) in zip(axes, specs):
        centers, offsets, width = bar_positions(len(MODEL_ORDER), 2)
        for i, model_key in enumerate(MODEL_ORDER):
            row = combined[combined["model"] == model_key].iloc[0]
            reported_v, e2e_v = row[reported_col], row[e2e_col]
            xpos = centers[i] + offsets
            bars = ax.bar(
                xpos, [reported_v, e2e_v], width=width * 0.9,
                color=[COLORS[model_key], COLORS[model_key]],
                alpha=1.0,
                hatch=[None, "///"],
                edgecolor="black", linewidth=0.4,
            )
            for b, v in zip(bars, [reported_v, e2e_v]):
                ax.annotate(pct(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.6, rotation=0)
        ax.set_xticks(centers)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER], rotation=12, ha="right")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(title)
        style_axis(ax)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Reported (excludes execution-error rows)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="End-to-end (all 218 rows, error = distinct outcome)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Run-to-run reproducibility of individual mapping decisions", y=1.02)
    save_figure(fig, "figure_04_reproducibility", main=True)


def fig_04b_run_to_run_f1(combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ys = np.arange(len(MODEL_ORDER))
    for i, model_key in enumerate(MODEL_ORDER):
        row = combined[combined["model"] == model_key].iloc[0]
        r1, r2 = row["run_1_weighted_f1_e2e"], row["run_2_weighted_f1_e2e"]
        y = ys[i]
        ax.plot([r1, r2], [y, y], color=COLORS[model_key], linewidth=2.2, zorder=1)
        ax.scatter([r1], [y], color=COLORS[model_key], marker="o", s=90, zorder=2, label="Run 1" if i == 0 else None)
        ax.scatter([r2], [y], color=COLORS[model_key], marker="D", s=80, zorder=2, label="Run 2" if i == 0 else None)
        mid = (r1 + r2) / 2
        diff = abs(r1 - r2)
        ax.annotate(f"|Δ|={diff:.3f}", (mid, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_xlabel("End-to-end weighted F1")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Run-to-run end-to-end weighted F1 stability\n(circle = Run 1, diamond = Run 2; smaller |Δ| = more stable)")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "figure_04b_run_to_run_f1", main=True)


# --------------------------------------------------------------------------
# Figure 5: latency breakdown
# --------------------------------------------------------------------------


def fig_05_latency_breakdown(latency_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    stages = [
        ("mean_query_planner_seconds", "Query planner", "#56B4E9"),
        ("mean_retrieval_seconds", "Retrieval", "#CC79A7"),
        ("mean_reranker_seconds", "LLM reranker", "#009E73"),
        ("mean_other_seconds", "Other / unattributed", "#BBBBBB"),
    ]
    x = np.arange(len(MODEL_ORDER))
    bottoms = np.zeros(len(MODEL_ORDER))
    for col, label, color in stages:
        values = np.array([latency_df[latency_df["model"] == m][col].iloc[0] for m in MODEL_ORDER])
        ax.bar(x, values, bottom=bottoms, color=color, label=label, width=0.62, edgecolor="white", linewidth=0.6)
        bottoms += values
    for i, m in enumerate(MODEL_ORDER):
        total = latency_df[latency_df["model"] == m]["mean_end_to_end_seconds"].iloc[0]
        ax.annotate(f"{total:.1f}s", (x[i], total), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylabel("Mean seconds per term (pooled across both runs)")
    ax.set_title("End-to-end latency decomposition by pipeline stage")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "figure_05_latency_breakdown", main=True)


def supp_fig_latency_distribution(data: dict[str, ModelData]):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    box_data = [data[m].pooled["end_to_end_seconds"].dropna().values for m in MODEL_ORDER]
    bp = ax.boxplot(
        box_data, patch_artist=True, showfliers=True, widths=0.55,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker="o", markersize=3, alpha=0.45, markeredgecolor="none"),
    )
    for patch, model_key in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[model_key])
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylabel("End-to-end seconds per mapping attempt")
    ax.set_title("Per-row end-to-end latency distribution (n=436/model, both runs pooled)")
    style_axis(ax)
    save_figure(fig, "supp_figure_latency_distribution", main=False)


# --------------------------------------------------------------------------
# Figure 6: ontology heatmaps
# --------------------------------------------------------------------------


def draw_heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, name: str, fmt="{:.0%}"):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v > 0.6 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, name, main=True)


def fig_06_ontology_heatmaps(ontology_perf: pd.DataFrame):
    order = ontology_order(ontology_perf)
    top1 = ontology_perf.pivot(index="model", columns="ontology", values="top1_accuracy").reindex(
        index=MODEL_ORDER, columns=order
    )
    top1.index = [DISPLAY_NAMES[m] for m in top1.index]
    draw_heatmap(
        top1,
        "Top-1 accuracy by ontology (end-to-end, both runs pooled)",
        "Top-1 accuracy",
        "figure_06_ontology_top1_heatmap",
    )

    f1 = ontology_perf.pivot(index="model", columns="ontology", values="weighted_f1").reindex(
        index=MODEL_ORDER, columns=order
    )
    f1.index = [DISPLAY_NAMES[m] for m in f1.index]
    draw_heatmap(
        f1,
        "Weighted F1 by ontology (end-to-end, both runs pooled)",
        "Weighted F1",
        "figure_06b_ontology_weighted_f1_heatmap",
    )


# --------------------------------------------------------------------------
# Figure 7: outcome distribution
# --------------------------------------------------------------------------

OUTCOME_COLORS = {
    "Gold rank 1": "#08519c",
    "Gold rank 2": "#3182bd",
    "Gold rank 3": "#6baed6",
    "Gold rank 4": "#9ecae1",
    "Gold rank 5": "#c6dbef",
    "Mapped, gold not in Top 5": "#fdae6b",
    "Unmapped": "#bdbdbd",
    "Execution error": "#d62728",
}


def fig_07_outcome_distribution(outcome_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    x = np.arange(len(MODEL_ORDER))
    bottoms = np.zeros(len(MODEL_ORDER))
    for cat in OUTCOME_CATEGORIES:
        values = np.array(
            [
                outcome_df[(outcome_df["model"] == m) & (outcome_df["outcome"] == cat)]["percent"].iloc[0]
                for m in MODEL_ORDER
            ]
        )
        bars = ax.bar(x, values, bottom=bottoms, color=OUTCOME_COLORS[cat], label=cat, width=0.6,
                       edgecolor="white", linewidth=0.6)
        for b, v, bottom in zip(bars, values, bottoms):
            if v >= 0.03:
                ax.text(b.get_x() + b.get_width() / 2, bottom + v / 2, f"{v * 100:.0f}%",
                        ha="center", va="center", fontsize=7.6,
                        color="white" if cat in ("Gold rank 1", "Gold rank 2", "Execution error") else "black")
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel(f"Share of {EXPECTED_ATTEMPTS_PER_MODEL} mapping attempts (both runs pooled)")
    ax.set_title("Mapping outcome distribution")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", alpha=0.2)
    style_axis(ax)
    save_figure(fig, "figure_07_mapping_outcome_distribution", main=True)


# --------------------------------------------------------------------------
# Figure 8: reliability
# --------------------------------------------------------------------------


def fig_08_pipeline_reliability(reliability_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    metrics = [
        ("execution_errors_per_100", "Execution errors"),
        ("rows_with_retrieval_retries_per_100", "Rows with retrieval retries"),
        ("rows_with_final_retrieval_errors_per_100", "Rows with final retrieval errors"),
    ]
    n_groups, n_series = len(metrics), len(MODEL_ORDER)
    centers, offsets, width = bar_positions(n_groups, n_series)
    for i, model_key in enumerate(MODEL_ORDER):
        row = reliability_df[reliability_df["model"] == model_key].iloc[0]
        values = [row[key] for key, _ in metrics]
        xpos = centers + offsets[i]
        bars = ax.bar(xpos, values, width=width * 0.92, color=COLORS[model_key], label=DISPLAY_NAMES[model_key])
        for b, v in zip(bars, values):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Rate per 100 mapping attempts")
    ax.set_title("Pipeline reliability and retrieval robustness\n(a final retrieval error is not the same as an execution error)")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "figure_08_pipeline_reliability", main=True)


# --------------------------------------------------------------------------
# Supplementary: token usage
# --------------------------------------------------------------------------


def supp_fig_token_usage(token_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    categories = [
        ("mean_total_input_tokens", "Input"),
        ("mean_total_cached_input_tokens", "Cached input"),
        ("mean_total_output_tokens", "Output"),
        ("mean_total_reasoning_tokens", "Reasoning"),
    ]
    n_groups, n_series = len(categories), len(MODEL_ORDER)
    centers, offsets, width = bar_positions(n_groups, n_series)
    for i, model_key in enumerate(MODEL_ORDER):
        row = token_df[token_df["model"] == model_key].iloc[0]
        xpos = centers + offsets[i]
        for j, (col, _) in enumerate(categories):
            v = row[col]
            if pd.isna(v):
                ax.text(xpos[j], 5, "N/A", ha="center", va="bottom", fontsize=8, rotation=90, color="dimgray")
                continue
            b = ax.bar(xpos[j], v, width=width * 0.92, color=COLORS[model_key],
                       label=DISPLAY_NAMES[model_key] if j == 0 else None)
            ax.annotate(f"{v:.0f}", (xpos[j], v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in categories])
    ax.set_ylabel("Mean tokens per mapping attempt (both runs pooled)")
    ax.set_title("Token usage by model\n(GPT-4.1 mini does not expose reasoning tokens: shown as N/A, not zero)")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    save_figure(fig, "supp_figure_token_usage", main=False)


# --------------------------------------------------------------------------
# Supplementary: normalized metric heatmap
# --------------------------------------------------------------------------

HEATMAP_METRICS = [
    ("mean_weighted_f1_e2e", "Weighted F1", "higher"),
    ("mean_top1_accuracy_e2e", "Top-1", "higher"),
    ("mean_top5_hit_rate_e2e", "Top-5", "higher"),
    ("top1_exact_agreement_reported", "Top-1 reproducibility", "higher"),
    ("top5_set_agreement_reported", "Top-5 reproducibility", "higher"),
    ("mean_api_cost_per_term_usd", "Cost / term", "lower"),
    ("mean_end_to_end_seconds_per_term", "E2E latency", "lower"),
    ("execution_error_rate", "Execution error rate", "lower"),
]


def supp_fig_model_metric_heatmap(combined: pd.DataFrame):
    labels = [label for _, label, _ in HEATMAP_METRICS]
    normalized = np.zeros((len(MODEL_ORDER), len(HEATMAP_METRICS)))
    annotations = np.empty_like(normalized, dtype=object)

    for j, (col, _, direction) in enumerate(HEATMAP_METRICS):
        values = combined.set_index("model")[col].reindex(MODEL_ORDER)
        vmin, vmax = values.min(), values.max()
        span = vmax - vmin
        for i, model_key in enumerate(MODEL_ORDER):
            v = values[model_key]
            if span == 0:
                norm = 1.0
            else:
                norm = (v - vmin) / span
                if direction == "lower":
                    norm = 1 - norm
            normalized[i, j] = norm
            if col == "mean_api_cost_per_term_usd":
                annotations[i, j] = fmt_usd_per_term(v)
            elif col == "mean_end_to_end_seconds_per_term":
                annotations[i, j] = f"{v:.1f}s"
            elif col == "execution_error_rate":
                annotations[i, j] = f"{v * 100:.2f}%"
            else:
                annotations[i, j] = pct(v)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(normalized, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8.6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Normalized rank (green = better)", labelpad=10)
    ax.set_title(
        "Model comparison across key metrics\n"
        "(per-column min-max normalized for display only, not a composite score -- see figure_captions.md)"
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, "supp_figure_model_metric_heatmap", main=False)


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------


def write_captions():
    captions = f"""# Figure captions

All figures compare four models: GPT-4.1 mini, GPT-5 mini, GPT-5.4 mini, and
GPT-5.6 Luna, evaluated on the same {EXPECTED_ROWS_PER_RUN}-row dictionary
mapping dataset (`dict_mapped_all.xlsx`, sha256
`91980c4df28781e5ef8d33d614c4f768966fa88b3db24996f3da38fc01bbddfd`), public
retrieval mode, 2 runs each ({EXPECTED_ATTEMPTS_PER_MODEL} mapping attempts
per model when both runs are pooled). Colors are held constant for each model
across every figure. Only two full benchmark repetitions exist per model; no
inferential confidence intervals are computed or implied from run-to-run
variation.

**figure_01_overall_performance.** Weighted F1, Top-1 accuracy, and Top-5 hit
rate per model, averaged across the two runs. Values are end-to-end: every
one of the 218 rows per run stays in the denominator, and any row where the
pipeline raised an execution error is scored as a complete miss (TP=0, FP=0,
FN=1, Top-1=0, Top-5=0), identically to how an "unmapped, gold exists" row is
already scored under the benchmark's locked scoring contract. Execution
errors therefore receive zero mapping credit rather than being dropped from
the average.

**figure_01b_precision_recall_f1.** Weighted precision, recall, and F1 per
model under the same end-to-end, execution-error-as-miss convention as
Figure 1, averaged across the two runs.

**figure_02_performance_vs_cost.** Scatter of mean end-to-end weighted F1
(y) against mean API cost per mapped term in USD (x, from each benchmark's
own pricing snapshot and observed token usage). One point per model,
annotated by name. Pareto-efficient models (no other model has both higher
F1 and lower cost) are outlined in black and connected by a thin dashed
frontier line. No combined cost/quality score is computed.

**figure_03_performance_vs_latency.** Scatter of mean end-to-end weighted F1
(y) against mean end-to-end wall-clock latency per term in seconds (x,
includes query planning, public retrieval, and LLM reranking). Pareto
frontier shown as in Figure 2.

**figure_03b_performance_vs_llm_latency.** Same as Figure 3 but the x-axis is
LLM-only latency per term (`mean_llm_seconds_per_term`), which excludes
public-retrieval wall-clock time that is not controlled by the LLM itself.
This isolates model-attributable latency from retrieval-service latency.

**figure_04_reproducibility.** Left: Top-1 exact agreement (identical
top-ranked code across the two runs). Right: Top-5 set agreement (identical
top-5 candidate code set across the two runs). Solid bars are the
benchmark's originally reported values, which exclude any row with an
execution error in either run from both the numerator and denominator.
Hatched bars are an end-to-end recomputation over all 218 rows, in which an
execution error is treated as its own distinct outcome state (it only
"agrees" with another execution error on the same row in the other run, and
never agrees with a real mapped code or an unmapped/blank result). The two
bars differ only for GPT-4.1 mini, which had one execution error in Run 1.

**figure_04b_run_to_run_f1.** Dumbbell plot of end-to-end weighted F1 in Run 1
(circle) vs. Run 2 (diamond) per model, with the absolute difference |Δ|
labeled. A small |Δ| indicates stable aggregate F1 across runs; it does not
imply the two runs produced identical per-row mappings -- that is a separate
property measured by Top-1 exact agreement in Figure 4, and the two need not
move together.

**figure_05_latency_breakdown.** Stacked mean seconds per mapping attempt by
pipeline stage (query planner, public retrieval, LLM reranker, and an
"other/unattributed" residual equal to measured end-to-end time minus the sum
of the three measured stages), pooled across both runs (n=436 rows/model).
Tiny negative residuals caused by floating-point rounding are clamped to
zero; a substantial negative residual would be reported as a reconciliation
issue rather than silently corrected. The residual is never labeled
"retrieval" -- it is only shown when it is not the measured retrieval-stage
timing.

**supp_figure_latency_distribution.** Boxplots of per-row end-to-end latency
(seconds), pooled across both runs, n=436 rows per model. These are
row-level empirical distributions over many individual mapping attempts, not
independent benchmark-run replicates; they characterize within-model spread,
not run-to-run performance confidence. Outliers are plotted as individual
points, not removed.

**figure_06_ontology_top1_heatmap** and **figure_06b_ontology_weighted_f1_heatmap.**
Top-1 accuracy and weighted F1 respectively, recomputed within each
`target_ontology` group using the same end-to-end, execution-error-as-miss
convention as Figure 1, pooling both runs. Cell values are annotated
directly; per-ontology sample sizes `n` are not shown on the heatmap but are
retained in `ontology_performance.csv` and should be consulted before
comparing percentages across ontologies of very different size.

**figure_07_mapping_outcome_distribution.** 100% stacked bar of every
mapping attempt (both runs pooled, {EXPECTED_ATTEMPTS_PER_MODEL} per model)
assigned to exactly one mutually exclusive outcome: gold rank 1-5 (the gold
code appeared at that position among the up-to-5 returned candidates),
"Mapped, gold not in Top 5" (the pipeline returned a mapping but the gold
code was not among the candidates), "Unmapped" (the pipeline declined to
map), or "Execution error" (the pipeline raised before producing a result).
Unmapped and execution error are never merged.

**figure_08_pipeline_reliability.** Rate per 100 mapping attempts (both runs
pooled) of three distinct events: execution errors (the pipeline raised and
produced no mapping), rows with at least one retrieval retry, and rows with
at least one *final* (unrecovered) retrieval error. A final retrieval error
is not the same as an execution error -- in this dataset, rows with a final
retrieval error still frequently produced a successful mapping (the pipeline
recovered or fell back), whereas an execution error always means no mapping
was produced for that row.

**supp_figure_token_usage.** Mean input, cached-input, output, and reasoning
tokens per mapping attempt (both runs pooled), from the pipeline's own
per-row token accounting (planner + reranker stages combined). GPT-4.1 mini
does not expose reasoning tokens (`reasoning_effort="N/A"`, a non-reasoning
model); its reasoning-token bar is marked "N/A" rather than plotted as zero,
since zero tokens and "not applicable" are not the same claim.

**supp_figure_model_metric_heatmap.** Eight metrics (Weighted F1, Top-1,
Top-5, Top-1 reproducibility, Top-5 reproducibility, cost/term, end-to-end
latency, execution error rate) per model. Colors are min-max normalized
independently within each column purely for visual legibility -- direction
is flipped for cost/latency/error-rate so that green always means "better on
this metric." Normalized colors are for display only and are not a
statistical claim; the underlying, non-normalized values are annotated in
each cell and are the same values reported in `combined_model_analysis.csv`.
This heatmap does not compute or imply any overall ranking or composite
score.
"""
    (OUT_ROOT / "figure_captions.md").write_text(captions)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    def fmt_cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body_lines = [
        "| " + " | ".join(fmt_cell(v) for v in row) + " |" for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body_lines]) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    for d in (MAIN_DIR, SUPP_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    data = load_all_models()
    validate(data)

    combined = build_combined_model_analysis(data)
    ontology_perf = build_ontology_performance(data)
    outcome_df = build_outcome_distribution(data)
    reliability_df = build_pipeline_reliability(data)
    latency_df = build_latency_breakdown(data)
    token_df = build_token_usage(data)
    pareto_df = build_pareto_summary(combined)

    combined.to_csv(OUT_ROOT / "combined_model_analysis.csv", index=False)
    ontology_perf.to_csv(DATA_DIR / "ontology_performance.csv", index=False)
    outcome_df.to_csv(DATA_DIR / "mapping_outcome_distribution.csv", index=False)
    reliability_df.to_csv(DATA_DIR / "pipeline_reliability.csv", index=False)
    latency_df.to_csv(DATA_DIR / "latency_breakdown.csv", index=False)
    token_df.to_csv(DATA_DIR / "token_usage.csv", index=False)
    pareto_df.to_csv(DATA_DIR / "pareto_summary.csv", index=False)

    md_cols = [
        "model_display", "mean_weighted_f1_e2e", "mean_top1_accuracy_e2e", "mean_top5_hit_rate_e2e",
        "run_1_weighted_f1_e2e", "run_2_weighted_f1_e2e", "weighted_f1_absolute_difference_e2e",
        "top1_exact_agreement_reported", "top5_set_agreement_reported",
        "mean_api_cost_per_term_usd", "mean_end_to_end_seconds_per_term",
        "execution_error_count", "execution_error_rate",
    ]
    (OUT_ROOT / "combined_model_analysis.md").write_text(dataframe_to_markdown(combined[md_cols]))

    fig_01_overall_performance(combined)
    fig_01b_precision_recall_f1(combined)
    fig_02_performance_vs_cost(combined)
    fig_03_performance_vs_latency(combined)
    fig_03b_performance_vs_llm_latency(combined)
    fig_04_reproducibility(combined)
    fig_04b_run_to_run_f1(combined)
    fig_05_latency_breakdown(latency_df)
    supp_fig_latency_distribution(data)
    fig_06_ontology_heatmaps(ontology_perf)
    fig_07_outcome_distribution(outcome_df)
    fig_08_pipeline_reliability(reliability_df)
    supp_fig_token_usage(token_df)
    supp_fig_model_metric_heatmap(combined)

    write_captions()

    print("\n=== Combined model analysis (key columns) ===")
    print(combined[md_cols].to_string(index=False))
    print(f"\nTotal reconciliation issues: {len(ISSUES)}")
    for i in ISSUES:
        print(" -", i)


if __name__ == "__main__":
    main()
