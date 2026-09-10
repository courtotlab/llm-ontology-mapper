# Figure captions

All figures compare four models: GPT-4.1 mini, GPT-5 mini, GPT-5.4 mini, and
GPT-5.6 Luna, evaluated on the same 218-row dictionary
mapping dataset (`dict_mapped_all.xlsx`, sha256
`91980c4df28781e5ef8d33d614c4f768966fa88b3db24996f3da38fc01bbddfd`), public
retrieval mode, 2 runs each (436 mapping attempts
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
mapping attempt (both runs pooled, 436 per model)
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
