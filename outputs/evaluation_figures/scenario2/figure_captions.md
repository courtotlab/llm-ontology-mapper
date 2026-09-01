# Scenario 2 (retrieval-mode ablation) figure captions

All figures compare three retrieval modes (Public, Local, Disabled) evaluated
on the same 218-row `dict_mapped_all.xlsx` dataset, holding model
(gpt-5.6-luna), reasoning effort (low), temperature, seed (42), and
max_alternatives (4) fixed. The same 218 rows were mapped under all three
modes (a paired design). Mode colors are held constant across every figure:
Public=blue, Local=orange, Disabled=bluish-green (Okabe-Ito palette).

**Metric definitions** (see `src/llm_ontology_mapper/benchmarking/scenario2_metrics.py`,
`scenario2_validation.py`, `scenario2_grounding.py` for the canonical code):
- **Top-1**: exact match between the top-ranked predicted code and any
  acceptable gold code for that row (`semantic_correctness`, locked identical
  to `top1_hit`). An ontology-valid but semantically wrong code is NOT correct.
- **Abstention**: the pipeline declined to map (`status="unmapped"`) or
  returned the `UNKNOWN:UNMAPPED` sentinel. Execution errors are a distinct
  outcome and are never counted as an abstention.
- **Hallucination**: among mapped predictions with a *definitive* ontology
  validation outcome (VALID or INVALID -- excludes UNRESOLVED and
  NOT_APPLICABLE rows), the fraction that are INVALID. Unresolved validator
  outcomes are never counted as hallucinations; see validation coverage.
- **Validation coverage**: (VALID + INVALID) / mapped_count -- how much of
  hallucination's own denominator was actually resolved.
- **Grounding**: among mapped predictions, the fraction whose selected code
  was present among the retrieved candidates. Mechanically 0 for Disabled
  (no retrieval stage exists to ground against) -- this figure reads the
  stored `grounding_rate` rather than assuming that value.
- **ROC AUC / Brier / ECE**: computed on mapped rows only, with exact Top-1
  correctness as the binary outcome and mapper confidence as the score
  (`scenario2_calibration.py`). Raw values shown, never normalized against
  each other -- AUC is higher-is-better, Brier and ECE are lower-is-better.
- **Paired transition**: rows are paired by the same source row (`row_id`)
  across all three modes, since all three modes mapped the identical 218 rows.

**figure_01_mapping_performance.** Top-1, Top-3, MRR, and Recall@GT per mode,
from each run's `mode_summary.csv`. Top-5 is excluded from this chart because
it is numerically indistinguishable from Top-3 in all three modes (public
0.6239 vs 0.6239; local 0.7798 vs 0.7798; disabled 0.4908 vs 0.4908) --
identical to 4 decimal places -- but Top-5 remains in
`data/scenario2_mapping_performance.csv`.

**figure_02_retrieval_behavior.** Abstention, hallucination, and grounding
rates per mode. "cov=" annotations above each hallucination bar report that
mode's validation coverage, read directly from `mode_summary.csv` (not
hardcoded) -- a low coverage would mean the hallucination rate itself rests
on fewer resolved codes than mapped_count suggests.

**figure_03_calibration_metrics.** Three independent panels (ROC AUC, Brier,
ECE), each with its own y-axis scaled to its own values -- never combined into
one "higher is better" chart, since the three metrics have opposite
desirable directions and very different natural ranges.

**figure_04_reliability_diagram.** Reuses
`scenario2_reliability_plot.plot_reliability_diagram()` unmodified: mean
predicted confidence (x) vs. empirical Top-1 accuracy (y) over the same fixed
10 equal-width confidence bins for every mode, plus the y=x perfect-
calibration reference. Bins with zero predictions in a given mode are omitted
from that mode's line, never interpolated or fabricated. Marker size was
deliberately left unscaled by bin count to avoid touching a function whose
current rendering behavior other code (`--compare` in
`run_scenario2_retrieval_ablation.py`) and its own tests already depend on.

**figure_05_ontology_top1_heatmap.** Top-1 accuracy by `target_ontology` and
mode, recomputed directly from each mode's `predictions.csv` (never from a
pre-aggregated summary). Per-ontology N is shown in the column labels and in
`data/ontology_top1_by_mode.csv`. NCIT (n=14), RxNorm (n=13), SNOMED (n=15),
and ICD10 (n=16) are far smaller strata than HPO (n=64), MONDO (n=49), and
LOINC (n=47); percentages in the small strata should not be over-interpreted
as precisely as the larger ones.

**figure_06_paired_correctness_transitions.** Three 2x2 matrices (Public vs
Local, Public vs Disabled, Local vs Disabled), built from
`scenario2_compare.build_paired_predictions()` over the same 218 paired rows.
Each cell is a mutually exclusive outcome (both correct / A correct-B wrong /
A wrong-B correct / both correct... i.e. both wrong), counts and row-total
percentages annotated. The two single-direction mismatch cells in each matrix
are cross-checked against `scenario2_compare.transition_counts()` and must
agree exactly. Correctness is exact Top-1 gold match (`semantic_correctness`),
never ontology-valid-but-wrong.

**supp_figure_01_rank_outcome_distribution.** Every one of the 218 rows per
mode assigned to exactly one mutually exclusive outcome, in priority order:
execution error, then abstained, then gold rank 1 / ranks 2-3 / ranks 4-5 /
no gold in the top 5. An unmapped row is classified as "Abstained", never as
"No gold in Top 5" (that category is reserved for mapped rows whose returned
candidates did not include a gold code).

**supp_figure_02_confidence_by_correctness.** Confidence distributions
(boxplot, outliers as individual jittered points, matplotlib only) split by
exact Top-1 correctness, for mapped, non-abstained rows only. Visually
supports (does not duplicate) the ROC AUC / rank-sum / Cohen's d statistics
already in `mode_summary.csv`.

**supp_figure_03_latency_breakdown.** End-to-end latency (boxplot,
n=218/mode -- the only latency field populated for every row in every mode).
A per-stage (planner/retrieval/reranker/LLM) stacked breakdown across all
three modes, matching `figure_05_latency_breakdown` in the model-comparison
figures, was deliberately NOT built: `retrieval_seconds`/`reranker_seconds`
are inapplicable by design for Disabled (no retrieval stage), and
`planner_seconds`/`llm_seconds` are populated for only 13/218
and 13/218 rows respectively in the Disabled run -- exactly its
13 execution-error rows, not a representative sample of its 218 mapped/
unmapped/error outcomes. Public and Local do have complete stage timing;
their breakdown is written to
`data/scenario2_latency_stage_breakdown_public_local.csv` rather than forced
into a 3-way chart that would misrepresent Disabled.

**data/scenario2_summary_table.{csv,md}.** All headline metrics per mode in
one table (no dedicated cost figure was built -- three numbers do not
warrant a standalone chart).

**data/scenario2_comparison.{csv,md} and data/paired_predictions.csv.**
Written via the existing `scenario2_compare` module's own table/CSV builders
(same functions `run_scenario2_retrieval_ablation.py --compare` uses),
unmodified.

No confidence intervals or error bars are shown anywhere in this figure set
(v1 scope). Paired-transition derivations are structured so a paired
bootstrap CI could be added to figure_06 later without re-deriving pairing.
