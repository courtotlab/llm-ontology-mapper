# Scenario 1 -- published-baseline comparison figures

This suite compares **LLM Ontology Mapper** (this repository's Scenario 1 EFO experiments) against two published baselines from the MetaHarmonizer paper: **MetaHarmonizer (OM)** and **text2term (t2t)**. It is analysis/visualization only -- it makes zero mapping/LLM/retrieval/ontology-validator/network calls, and never modifies the original completed run directories or the pre-existing model-comparison or Scenario 1 figures.

### Published method terminology

- **OM = OntologyMapper.** In the MetaHarmonizer paper, OM is the ontology-standardization component of MetaHarmonizer, not the whole system. Figures in this suite therefore label it **"MetaHarmonizer (OM)"** rather than bare "MetaHarmonizer", so a reader is never left to infer what the acronym covers.
- **t2t = text2term.** Figures label it **"text2term (t2t)"**.
- Our own method is labeled **"LLM Ontology Mapper"** (this repository's formal name, per `README.md`), never a bare "model" or "our model", since a proper system name already exists.
- Internally (CSV `tool` column, Python identifiers) OM is keyed as `metaharmonizer_ontology_mapper` / `metaharmonizer_om` and t2t as `text2term`; only the *display* labels above are used on any figure or in prose.

### Benchmark denominators

| Benchmark | LLM Ontology Mapper (n) | MetaHarmonizer (OM) (n) | text2term (t2t) (n) |
| --- | --- | --- | --- |
| UKBB-EFO | 888 | 888 | 888 |
| Biomappings-EFO | 795 | 795 | 795 |
| OLS-EFO (full) | 7377 | 7377 | 7504 |

**OLS-EFO (full) denominator caveat.** LLM Ontology Mapper and MetaHarmonizer (OM) are both evaluated on n=7377 unique queries; text2term's published OLS-EFO (full) figure is evaluated on a *different* n=7504 (it reports over the full 7,504-row mapping-pair set rather than the 7,377 deduplicated unique queries used here). Every OLS-EFO figure and table in this suite that includes text2term shows both denominators; this is **not** an identical-N comparison and should not be presented as one.

**UKBB-EFO caveat.** UKBB-EFO's gold namespace composition differs from the 100%-EFO-native OLS-EFO and Biomappings-EFO gold sets used elsewhere in this repository's Scenario 1 evaluation (see the Scenario 1 UKBB run's own dataset_validation.json / README notes). Top-k/MRR values remain directly comparable across methods *within* UKBB-EFO (all three methods are scored against the same gold set), but UKBB-EFO's absolute numbers should not be read as equivalent in difficulty to OLS-EFO or Biomappings-EFO.

### Source of published baseline values

OM and text2term values come from a single structured CSV, `published_baselines_used.csv` (schema: benchmark, tool, metric, value, denominator, source_publication, source_table_or_figure, notes; unit-fraction values), snapshotted unchanged into `data/published_baselines_used.csv` on every run of this figure suite. Publication: the MetaHarmonizer paper's benchmark table, as supplied to this repository by the project maintainer. The exact table/figure number has **not** been independently re-verified against the published PDF in this codebase, so `source_table_or_figure` is recorded honestly as unverified rather than a fabricated citation. OLS-EFO (disease) rows, if ever added to that CSV, are never consumed by this figure suite -- our Scenario 1 experiments did not run that subset, so there is no matching three-method comparison for it.

### Source of our (LLM Ontology Mapper) values

Our Top-1/Top-3/Top-5/MRR values come from these exact completed Scenario 1 runs' `scenario1_metrics.csv`, reconciled against each run's own `predictions.csv` using the unmodified `scenario1_metrics.score_prediction`/`aggregate` utilities:

- **UKBB-EFO**: `outputs/evaluation/scenario1_ukbb_efo/2026-08-31T13-54-53Z`
- **Biomappings-EFO**: `outputs/evaluation/scenario1_biomappings_efo/2026-08-31T16-10-24Z`
- **OLS-EFO (full)**: `outputs/evaluation/scenario1_ols_efo/2026-08-26T15-04-18Z`

Derived from those runs' `experiment_config.json` (not hardcoded): model=`gpt-5.6-luna`, retrieval_mode=`local`, target_ontology=`EFO`, strict_target_ontology=`False`. All three runs share this configuration; see `data/our_scenario1_metrics_used.csv` for the per-benchmark values.

## Outcome-distribution derivation

Four mutually-exclusive first-gold-rank bins are reconstructed identically for every method (LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t)) from cumulative Top-1/Top-3/Top-5 alone:

- Gold rank 1 = Top-1
- Gold rank 2-3 = Top-3 − Top-1
- Gold rank 4-5 = Top-5 − Top-3
- No gold in Top 5 = 1 − Top-5

These are mutually exclusive categories reconstructed from cumulative Top-k metrics -- not a figure copied from the MetaHarmonizer paper (which does not report a first-gold-rank composition directly; see the online source note below) and not derived from any richer per-row status. The published MetaHarmonizer/text2term aggregate does not expose execution errors, abstentions, and ordinary no-hit mapped predictions as separate components, so those cannot be separated out for OM or text2term. For a fair, symmetric comparison, LLM Ontology Mapper's own richer per-row outcomes (execution error, abstained, mapped-but-missed) are likewise collapsed into "No gold in Top 5" in **these cross-method figures only** -- the existing internal Scenario 1/2 figures elsewhere in this repository continue to show that richer breakdown for our method alone.

Each method's four bins are checked to sum to 1.0 within a 0.01 tolerance (published Top-k values are reported to one decimal percentage point, so tiny rounding slack is expected) before any figure is drawn; a larger discrepancy, or a negative implied bin from non-monotonic Top-k values, hard-fails rather than silently plotting.

### Online source note: MetaHarmonizer vs. original text2term

**MetaHarmonizer paper, Figure 3.** Panel A reports cumulative Top-k performance for OntologyMapper (OM) and text2term (t2t) under MetaHarmonizer's own controlled comparison protocol -- this is the source of every Top-1/Top-3/Top-5 value used in this suite, including as the sole input to the four-bin derivation above. Panel B reports the composition of *correct* Top-1 OM predictions by pipeline resolving stage, and Panel C reports confidence distributions for correct vs. incorrect Top-1 predictions. **None of Figure 3's panels report a first-gold-rank distribution directly** -- the OM/t2t bars in Figures 10-12 are derived, not copied, from Panel A's cumulative Top-k values.

**Original text2term publication.** Separately, the original text2term paper reports its own Top-1 mapping-*relationship* distribution (Same / More Specific / More General / Sibling / Unrelated) for UKBB-EFO, Biomappings, and OLS, with public evaluation code/data. This is intentionally **not** used for the rank-composition figures in this suite, because (1) it classifies the *graph relationship* of a Top-1 prediction, a different quantity than first-gold rank; and (2) it comes from the original text2term paper's own evaluation protocol, not the MetaHarmonizer-controlled rerun that this suite uses as its t2t baseline everywhere else (see `data/published_baselines_used.csv`). Mixing the two would silently compare text2term under two different protocols inside the same figure -- the original graph-relation percentages are **not** interchangeable with the controlled rerun used here. A graph-distance comparison against the *original* text2term paper, if wanted, should be a separate analysis with its own explicit protocol caveat.

## Figures

### Figure 1 — All-method Top-k comparison

**Files**
- main/figure_01_all_methods_topk.png
- main/figure_01_all_methods_topk.svg

**Question.** How does LLM Ontology Mapper's Top-1/Top-3/Top-5 accuracy compare to MetaHarmonizer (OM) and text2term (t2t) on each benchmark?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t) (three bars per metric group, this fixed order, never sorted by performance).

**Metrics.** Top-1, Top-3, Top-5 accuracy (fraction of queries where an acceptable gold code appears within the top-k ranked predictions).

**Axes.** x: Top-1 / Top-3 / Top-5 (grouped within each panel); y: Accuracy, 0-100%, shared across all three panels so bar heights are directly comparable.

**Interpretation.** Higher is better for every bar. Values annotated to one decimal percentage point.

**Denominators.** See the denominators table above; the OLS-EFO (full) panel subtitle explicitly shows both n=7,377 (ours/OM) and n=7,504 (text2term).

**Caveats.** Do not read the OLS-EFO (full) text2term bars as scored on the identical query set as the other two methods in that panel.

**Source data**
- data/all_methods_topk.csv
- data/all_methods_comparison.csv

### Figure 2 — All-method MRR comparison

**Files**
- main/figure_02_all_methods_mrr.png
- main/figure_02_all_methods_mrr.svg

**Question.** How does ranking quality (not just Top-1 hit/miss) compare across methods?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t).

**Metrics.** Mean Reciprocal Rank (MRR), a unit-fraction ranking-quality score -- explicitly NOT a percentage.

**Axes.** x: Benchmark; y: MRR, 0-1, shared across the whole chart. Values annotated to three decimal places.

**Interpretation.** Higher MRR means the correct code tends to rank closer to position 1 on average.

**Denominators.** Same as Figure 1 -- see denominators table.

**Caveats.** Kept as a separate figure from Top-k on purpose (Part 11) rather than mixed into the Top-k panels, since MRR is a continuous ranking score, not an accuracy fraction.

**Source data**
- data/all_methods_comparison.csv

### Figure 3 — LLM Ontology Mapper vs. MetaHarmonizer (OM) -- Top-k

**Files**
- pairwise/figure_03_our_model_vs_metaharmonizer_topk.png
- pairwise/figure_03_our_model_vs_metaharmonizer_topk.svg

**Question.** Head-to-head: how does our method's Top-k accuracy compare specifically to MetaHarmonizer's OntologyMapper component?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM) only (text2term omitted from this pair).

**Metrics.** Top-1, Top-3, Top-5 accuracy.

**Axes.** x: Top-1 / Top-3 / Top-5 (grouped within each panel); y: Accuracy, 0-100%, shared across panels.

**Interpretation.** Higher is better. Same method colors as every other figure in this suite.

**Denominators.** OLS-EFO (full): both methods share n=7,377 here (this pair has no denominator mismatch -- the mismatch is specific to text2term).

**Caveats.** None beyond the general UKBB-EFO gold-namespace caveat above.

**Source data**
- data/pairwise_vs_metaharmonizer.csv

### Figure 4 — LLM Ontology Mapper vs. MetaHarmonizer (OM) -- MRR

**Files**
- pairwise/figure_04_our_model_vs_metaharmonizer_mrr.png
- pairwise/figure_04_our_model_vs_metaharmonizer_mrr.svg

**Question.** Head-to-head ranking quality vs. MetaHarmonizer (OM).

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM).

**Metrics.** MRR (unit fraction, not a percentage).

**Axes.** x: Benchmark; y: MRR, 0-1.

**Interpretation.** Higher is better.

**Denominators.** n=7,377 shared on OLS-EFO (full) for this pair.

**Caveats.** None beyond the general UKBB-EFO caveat.

**Source data**
- data/pairwise_vs_metaharmonizer.csv

### Figure 5 — LLM Ontology Mapper vs. text2term (t2t) -- Top-k

**Files**
- pairwise/figure_05_our_model_vs_text2term_topk.png
- pairwise/figure_05_our_model_vs_text2term_topk.svg

**Question.** Head-to-head: how does our method's Top-k accuracy compare specifically to text2term?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, text2term (t2t) only.

**Metrics.** Top-1, Top-3, Top-5 accuracy.

**Axes.** x: Top-1 / Top-3 / Top-5 (grouped within each panel); y: Accuracy, 0-100%, shared across panels.

**Interpretation.** Higher is better.

**Denominators.** OLS-EFO (full) panel subtitle explicitly shows n ours=7,377 vs. n text2term=7,504 -- the one panel in this whole suite where the two bars being compared do not share a denominator.

**Caveats.** Do not read the OLS-EFO (full) panel as an identical-N comparison.

**Source data**
- data/pairwise_vs_text2term.csv

### Figure 6 — LLM Ontology Mapper vs. text2term (t2t) -- MRR

**Files**
- pairwise/figure_06_our_model_vs_text2term_mrr.png
- pairwise/figure_06_our_model_vs_text2term_mrr.svg

**Question.** Head-to-head ranking quality vs. text2term.

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, text2term (t2t).

**Metrics.** MRR (unit fraction, not a percentage).

**Axes.** x: Benchmark; y: MRR, 0-1.

**Interpretation.** Higher is better.

**Denominators.** OLS-EFO (full): n ours=7,377 vs. n text2term=7,504 -- see caveat above.

**Caveats.** Same OLS-EFO (full) denominator caveat as Figure 5.

**Source data**
- data/pairwise_vs_text2term.csv

### Figure 7 — Δ vs. MetaHarmonizer (OM)

**Files**
- pairwise/figure_07_delta_vs_metaharmonizer.png
- pairwise/figure_07_delta_vs_metaharmonizer.svg

**Question.** By how many percentage points does LLM Ontology Mapper's Top-k accuracy exceed or trail MetaHarmonizer (OM), per benchmark and per k?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper minus MetaHarmonizer (OM). Δ = ours − OM.

**Metrics.** ΔTop-1, ΔTop-3, ΔTop-5, expressed in **percentage points**, never called a percentage change.

**Axes.** x: Percentage-point difference (zero-centered, zero always visible); y: One horizontal bar per (benchmark, k) combination.

**Interpretation.** A bar extending right of zero (positive) means LLM Ontology Mapper is higher on that metric; a bar extending left (negative) means MetaHarmonizer (OM) is higher. No statistical test is implied by bar length alone.

**Denominators.** n=7,377 shared with MetaHarmonizer (OM) on every benchmark in this chart (no OLS denominator mismatch here -- that only affects the text2term comparison).

**Caveats.** ΔMRR is not plotted here (it is a raw unit-fraction difference, not percentage points) -- see the per-benchmark ΔMRR list below and data/delta_vs_metaharmonizer.csv.

- UKBB-EFO: ΔTop-1=+1.2 pp, ΔTop-3=+1.7 pp, ΔTop-5=+0.4 pp, ΔMRR=+0.014
- Biomappings-EFO: ΔTop-1=-0.2 pp, ΔTop-3=-1.6 pp, ΔTop-5=-2.1 pp, ΔMRR=-0.009
- OLS-EFO (full): ΔTop-1=-5.7 pp, ΔTop-3=-5.7 pp, ΔTop-5=-6.1 pp, ΔMRR=-0.057

**Source data**
- data/delta_vs_metaharmonizer.csv

### Figure 8 — Δ vs. text2term (t2t)

**Files**
- pairwise/figure_08_delta_vs_text2term.png
- pairwise/figure_08_delta_vs_text2term.svg

**Question.** By how many percentage points does LLM Ontology Mapper's Top-k accuracy exceed or trail text2term, per benchmark and per k?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper minus text2term (t2t). Δ = ours − t2t.

**Metrics.** ΔTop-1, ΔTop-3, ΔTop-5, in percentage points.

**Axes.** x: Percentage-point difference (zero-centered, zero always visible); y: One horizontal bar per (benchmark, k) combination.

**Interpretation.** Same reading as Figure 7, against text2term instead of MetaHarmonizer (OM).

**Denominators.** OLS-EFO (full) bars in this chart compare n=7,377 (ours) against n=7,504 (text2term) -- the same denominator mismatch as Figures 1 and 5.

**Caveats.** ΔMRR is not plotted (unit-fraction difference, not percentage points) -- see below and data/delta_vs_text2term.csv.

- UKBB-EFO: ΔTop-1=+7.5 pp, ΔTop-3=+7.7 pp, ΔTop-5=+6.5 pp, ΔMRR=+0.075
- Biomappings-EFO: ΔTop-1=+16.2 pp, ΔTop-3=+6.9 pp, ΔTop-5=+3.4 pp, ΔMRR=+0.112
- OLS-EFO (full): ΔTop-1=+4.2 pp, ΔTop-3=+2.3 pp, ΔTop-5=+1.8 pp, ΔMRR=+0.033

**Source data**
- data/delta_vs_text2term.csv

### Figure 9 — Top-1-only headline summary

**Files**
- main/figure_09_top1_summary.png
- main/figure_09_top1_summary.svg

**Question.** A single, presentation-friendly headline view of Top-1 accuracy across all three benchmarks and methods, without the Top-3/Top-5 facets of Figure 1.

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t).

**Metrics.** Top-1 accuracy only.

**Axes.** x: Benchmark; y: Accuracy, 0-100%.

**Interpretation.** Higher is better. Generated because a single-panel, single-metric chart is easier to drop into a slide than Figure 1's three-panel Top-1/3/5 facet grid -- it is a genuinely different layout, not a duplicate.

**Denominators.** Same denominator caveats as Figure 1 (OLS-EFO (full) text2term n=7,504 vs. n=7,377).

**Caveats.** Read alongside Figure 1 for Top-3/Top-5; this figure intentionally omits them.

**Source data**
- data/all_methods_comparison.csv

### Figure 10 — All-method ranked-outcome distribution

**Files**
- main/figure_10_all_methods_outcome_distribution.png
- main/figure_10_all_methods_outcome_distribution.svg

**Question.** How is each method's probability mass distributed across first-gold-rank buckets, not just collapsed into a single Top-1 hit/miss number?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM), text2term (t2t) -- one 100%-stacked bar per method, per panel, in this fixed order.

**Metrics.** Four mutually-exclusive rank bins -- Gold rank 1, Gold rank 2-3, Gold rank 4-5, No gold in Top 5 -- see 'Outcome-distribution derivation' above for the exact formulas.

**Axes.** x: Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / MetaHarmonizer (OM) / text2term (t2t) names); y: Share of evaluated queries, 0-100%, shared across all three panels.

**Interpretation.** Segment **color encodes outcome category, not method** -- the same four colors are reused for every method and every panel in this whole suite; method identity comes only from the x-axis label. A larger 'Gold rank 1' share is better; a larger 'No gold in Top 5' share is worse. Segments below 3% are left unlabeled to avoid unreadable clutter -- exact values are always in the CSV regardless of whether they were annotated on the chart.

**Denominators.** OLS-EFO (full): text2term's bar is explicitly labeled n=7,504 while LLM Ontology Mapper's and MetaHarmonizer (OM)'s bars are labeled n=7,377, directly under each bar.

**Caveats.** This composition is **reconstructed, not measured directly**, for MetaHarmonizer (OM) and text2term (t2t): their published aggregate does not expose execution errors, abstentions, or ordinary no-hit predictions separately, so all non-Top-5 outcomes collapse into 'No gold in Top 5' for every method shown here, including ours (see 'Outcome-distribution derivation' above). Do not read 'No gold in Top 5' as 'abstained' or 'errored' for any method.

**Source data**
- data/outcome_distribution_all_methods.csv
- data/outcome_distribution_all_methods.md

### Figure 11 — LLM Ontology Mapper vs. MetaHarmonizer (OM) — ranked-outcome distribution

**Files**
- pairwise/figure_11_outcome_distribution_vs_metaharmonizer.png
- pairwise/figure_11_outcome_distribution_vs_metaharmonizer.svg

**Question.** Head-to-head first-gold-rank composition against MetaHarmonizer (OM) only.

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, MetaHarmonizer (OM) only (text2term omitted from this pair).

**Metrics.** Same four rank bins as Figure 10.

**Axes.** x: Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / MetaHarmonizer (OM) / text2term (t2t) names); y: Share of evaluated queries, 0-100%, shared across panels.

**Interpretation.** Same reading as Figure 10, restricted to this pair; same category colors, same 3%-minimum labeling rule.

**Denominators.** n=7,377 shared by both methods on OLS-EFO (full) -- no denominator mismatch in this pair (the mismatch is specific to text2term).

**Caveats.** Same reconstruction caveat as Figure 10 -- 'No gold in Top 5' is not 'abstained' or 'errored' for either method.

**Source data**
- data/outcome_distribution_all_methods.csv

### Figure 12 — LLM Ontology Mapper vs. text2term (t2t) — ranked-outcome distribution

**Files**
- pairwise/figure_12_outcome_distribution_vs_text2term.png
- pairwise/figure_12_outcome_distribution_vs_text2term.svg

**Question.** Head-to-head first-gold-rank composition against text2term (t2t) only.

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full).

**Methods.** LLM Ontology Mapper, text2term (t2t) only.

**Metrics.** Same four rank bins as Figure 10.

**Axes.** x: Method (short axis labels -- 'Our method'/'OM'/'t2t' -- with n shown directly underneath; the legend and prose elsewhere use the full LLM Ontology Mapper / MetaHarmonizer (OM) / text2term (t2t) names); y: Share of evaluated queries, 0-100%, shared across panels.

**Interpretation.** Same reading as Figure 10, restricted to this pair.

**Denominators.** OLS-EFO (full): n ours=7,377 vs. n text2term=7,504, labeled directly under each bar -- the same denominator mismatch as Figures 1, 5, and 8.

**Caveats.** Same reconstruction caveat as Figure 10. text2term's bars here use the MetaHarmonizer-controlled rerun values, NOT the original text2term paper's own Same/More-Specific/.../Unrelated Top-1 graph-relationship distribution -- see the online source note above for why those are never mixed.

**Source data**
- data/outcome_distribution_all_methods.csv

**On per-benchmark figures 13-15.** A dense per-benchmark breakout (figure_13_ukbb_outcome_distribution / figure_14_biomappings_outcome_distribution / figure_15_ols_outcome_distribution) was considered and deliberately **not generated**: Figure 10's three-panel layout already keeps each panel to three bars of four segments each, matches the visual density of the existing all-method Top-k/pairwise figures elsewhere in this suite, and was confirmed legible on inspection (labels not clipped, small segments cleanly omitted rather than overlapping). Splitting it into three single-benchmark figures would be redundant with Figure 10 without adding readability.

## Metrics not compared to published methods

**Precision/Recall/F1** and the **graph-distance taxonomy** (Same / More Specific / More General / Sibling / Unrelated) are Scenario 1's own graph-based metrics. The supplied published baseline table provides only Top-1/Top-3/Top-5/MRR for OM and text2term -- it does not provide comparable Precision/Recall/F1 or graph-distance values for those tools, so no cross-method figure is generated for them here. They remain available, for our method only, in the existing Scenario 1 figure suite.

## Interpretation notes

Every comparison above is a direct read of the plotted values ("LLM Ontology Mapper has a higher/lower Top-1 than OM on this benchmark"). No hypothesis test has been run, so no claim of statistical significance is made or implied, and no causal explanation (e.g. about *why* one architecture outperforms another) should be drawn from these figures alone.

# Top-1 graph-relationship comparison with text2term

This section compares **LLM Ontology Mapper**'s Top-1 predictions against the **original text2term publication**'s own Top-1 graph-relationship distribution -- a different comparison, with a different text2term source, than the controlled Top-k/MRR comparison above.

### Controlled Top-k baseline vs. graph-relationship baseline

| | Source | What it provides |
| --- | --- | --- |
| **CONTROLLED TOP-K BASELINE** (Figures 1-12 above) | MetaHarmonizer paper's own controlled rerun of text2term | Top-1 / Top-3 / Top-5 / MRR only -- no graph-relationship categories |
| **GRAPH-RELATIONSHIP BASELINE** (Figures 13-16 below) | Original text2term v4.1.2 publication (Table 1) / text2term-evaluation repository | Same / More Specific / More General / Sibling / Unrelated |

These are two different text2term executions under two different protocols and are **never** merged into one source or implied to be the same run. text2term's Top-k figures above and its graph-relationship figures below should not be read as describing the identical execution.

## Category definitions and priority

Reproduced from text2term-evaluation's `compare_ontology_mappings.compare_mappings()` (pinned commit `b999dbb670fa13c9ceb1ba631a7abc7557f3293b`), and audited line-for-line in `scenario1_graph_distance.py`'s module docstring:

- **Same** -- predicted mapping equals the benchmark (gold) mapping.
- **More Specific** -- the predicted term is a subclass/entailed descendant of the benchmark term.
- **More General** -- the predicted term is a superclass/entailed ancestor of the benchmark term.
- **Sibling** -- the predicted and benchmark terms share an asserted direct superclass.
- **Unrelated** -- none of the above graph relationships apply.

**Priority order** (checked first-match-wins, both here and in the reference implementation): Same → More Specific → More General → Sibling → Unrelated. This repository's `scenario1_graph_distance.ALL_RELATIONSHIPS` is verified equal to this exact tuple before any figure in this section is drawn (`verify_graph_evaluator_compatibility()`); a mismatch would hard-fail rather than silently plot mismatched semantics as comparable.

## Source provenance

**Our graph evaluator.** `scenario1_graph_distance.py` reimplements (never imports/executes) text2term-evaluation's `compare_mappings()` logic, against repository `https://github.com/rsgoncalves/text2term-evaluation`, file `compare_ontology_mappings.py (compare_mappings)`, pinned commit `b999dbb670fa13c9ceb1ba631a7abc7557f3293b`, using EFO v3.62.0 (`http://www.ebi.ac.uk/efo/releases/v3.62.0/efo.owl`). Every one of our three official runs' own `graph_reference_metadata.json` was verified identical and consistent with these constants before this section was generated.

**text2term's own graph-relationship values.** Sourced from the *original* text2term publication's Table 1 (text2term v4.1.2, EFO v3.62.0, `https://github.com/rsgoncalves/text2term-evaluation`), as supplied to this repository -- stored in `data/text2term_graph_relationship_baseline.csv` (columns: benchmark, source, text2term_version, efo_version, n, relationship, count, proportion, publication) and never hardcoded a second time in any plotting function. The text2term version tag is recorded as reported; it has not been independently re-verified against a PyPI/source-code artifact in this repository.

## Published original-text2term Table 1 values

| Benchmark | n | Same | More Specific | More General | Sibling | Unrelated |
| --- | --- | --- | --- | --- | --- | --- |
| UKBB-EFO | 899 | 73.4% (660) | 3.8% (34) | 2.2% (20) | 1.4% (13) | 19.1% (172) |
| Biomappings-EFO | 795 | 78.7% (626) | 0.0% (0) | 0.3% (2) | 5.9% (47) | 15.1% (120) |
| OLS-EFO (full) | 8,143 | 80.9% (6,588) | 1.1% (91) | 0.7% (55) | 1.1% (89) | 16.2% (1,320) |

Percentages above are computed as count/n at full precision (the same arithmetic used for every figure and CSV in this section, so bar heights always sum to exactly 100%) and may therefore differ by up to 0.1 percentage point from the paper's own independently-rounded percentage for the same count -- the underlying counts are the authoritative published values and are reproduced exactly.

## Two denominator views

Our own graph-distance summary's denominator audit (Part 5) confirmed that `sum(Same, More Specific, More General, Sibling, Unrelated)` equals `mapped_count` (the classifiable Top-1 predictions), **not** the full evaluated N -- the gap is exactly `unmapped_count + error_count`, stored as `Not Applicable` in `graph_distance_summary.csv`. Two figures make this explicit rather than picking one silently:

- **Figure 13 (mapped-only).** Denominator = our classifiable Top-1 predictions only (`mapped_count`); text2term's published categories already exhaust its full N. Answers: "given a classifiable Top-1 mapping, what was its relationship to gold?"
- **Figure 14 (end-to-end).** Denominator = the full evaluated N for both methods, with a sixth category, **No Top-1 prediction** (= unmapped + execution error for us; verified 0 for text2term because its published counts were confirmed to sum exactly to its own n before being assigned 0). Answers: "across the entire evaluated set, what happened at Top-1?"

### Our denominator breakdown (per benchmark)

| Benchmark | Total n | Mapped | Unmapped | Execution error | No Top-1 prediction |
| --- | --- | --- | --- | --- | --- |
| UKBB-EFO | 888 | 884 | 4 | 0 | 4 |
| Biomappings-EFO | 795 | 781 | 14 | 0 | 14 |
| OLS-EFO (full) | 7,377 | 7,262 | 115 | 0 | 115 |

## Comparability limitations

| Benchmark | Our n | text2term (original) n | Denominators match? |
| --- | --- | --- | --- |
| UKBB-EFO | 888 | 899 | No |
| Biomappings-EFO | 795 | 795 | Yes |
| OLS-EFO (full) | 7,377 | 8,143 | No |

- **UKBB-EFO denominator mismatch.** Our run evaluates n=888; the original text2term Table 1 evaluates n=899. These are NOT presented as an exact head-to-head comparison.
- **OLS-EFO denominator mismatch -- THREE different OLS numbers appear across this whole figure suite, and they must not be confused:** our controlled-comparison/graph-distance N is n=7,377 unique queries (Figures 1-14); the MetaHarmonizer-controlled text2term rerun used for the Top-k comparison (Figures 1, 5, 8, 12) reports n=7,504 mapping pairs; and the *original* text2term publication's own OLS-EFO graph-relationship evaluation used n=8,143 rows -- a materially different, larger set than either of the other two. All three are legitimate numbers from different sources/protocols, never interchangeable.
- **Biomappings-EFO appears most directly comparable**: our n and the original text2term n are both 795, though this equality of N alone is not proof of identical row identity (see common-query alignment below).
- **Original text2term protocol was single-gold only.** The original text2term paper's comparison was limited to queries with exactly one benchmark mapping. Our own gold-count audit (`data/our_multi_gold_audit.csv`):
  - UKBB-EFO: 888 queries with 1 gold
  - Biomappings-EFO: 795 queries with 1 gold
  - OLS-EFO (full): 7,257 queries with 1 gold, 113 queries with 2 gold, 7 queries with 3 gold
  OLS-EFO (full) includes multi-gold queries (113 with 2 acceptable golds, 7 with 3), so our full-run OLS-EFO graph distribution is not perfectly protocol-identical to the original text2term Table 1 distribution even where n happened to align; UKBB-EFO and Biomappings-EFO are 100% single-gold, matching the original text2term protocol on this dimension.
- **The original text2term run used here is NOT the MetaHarmonizer-controlled t2t rerun** used everywhere else in this suite -- see the table at the top of this section.

## Common-query alignment audit

A stronger, common-query-aligned comparison (Figure 15) was considered: intersecting our evaluated query/gold records with the raw per-query outputs published in the `rsgoncalves/text2term-evaluation` repository (`output/*_t2t_mappings.csv`, `output/*_mappings.tsv`, `output/*_results.tsv`), then re-evaluating both methods on the identical matched rows with the same graph evaluator.

**This alignment was NOT attempted and Figure 15 was NOT generated.** The raw per-query output files are not present in this repository or vendored under `data/text2term_evaluation/` (only the EFO edge/entailed-edge reference tables used by our own graph evaluator are vendored there). Fetching them now would require a live network call to GitHub, which is out of scope for this analysis-only, zero-network plotting task, and per this suite's reproducibility policy any such raw files would first need to be explicitly vendored under a reproducible data directory with their source URL, repository commit, and SHA256 recorded -- not fetched silently as a side effect of plotting. No fuzzy string matching, row-identity inference, or ambiguous-duplicate resolution was performed or would be acceptable as a substitute. Figures 13 and 14 above therefore remain descriptive published-protocol comparisons with the explicit denominator caveats documented in this section, not a common-query-aligned comparison.

### Figure 13 — Top-1 graph relationship — classifiable predictions only

**Files**
- pairwise/figure_13_graph_relationships_mapped_only.png
- pairwise/figure_13_graph_relationships_mapped_only.svg

**Question.** Given a classifiable Top-1 mapping, what was its relationship to the benchmark gold?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).
**Methods.** Our method, text2term (original publication) only -- MetaHarmonizer (OM) is NOT included in this figure (Part 16): the published OM table provides cumulative Top-k metrics only, with no equivalent five-category graph-relationship breakdown.
**Categories.** Five mutually-exclusive graph-relationship categories (Same, More Specific, More General, Sibling, Unrelated), each method normalized to its OWN classifiable-predictions denominator (see 'Two denominator views' above).
**Colors.** Category colors, not method colors -- the same six hex values are reused across Figures 13/14 (and 16, if generated) for the same category.
**Denominators.** n is shown directly under each bar's label; see the denominator table and caveats above.
**Caveats.** Descriptive published-protocol comparison, not common-query aligned (see above); mismatched denominators for UKBB-EFO and OLS-EFO (full).
**Source data.** `data/graph_relationship_mapped_only.csv`

### Figure 14 — Top-1 graph relationship — end-to-end

**Files**
- pairwise/figure_14_graph_relationships_end_to_end.png
- pairwise/figure_14_graph_relationships_end_to_end.svg

**Question.** Across the entire evaluated set, what happened at Top-1 (including no prediction at all)?

**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (full) (three panels, this fixed order).
**Methods.** Our method, text2term (original publication) only -- MetaHarmonizer (OM) is NOT included in this figure (Part 16): the published OM table provides cumulative Top-k metrics only, with no equivalent five-category graph-relationship breakdown.
**Categories.** Six mutually-exclusive categories -- the five graph relationships plus 'No Top-1 prediction' -- both methods normalized to their own full evaluated N.
**Colors.** Category colors, not method colors -- the same six hex values are reused across Figures 13/14 (and 16, if generated) for the same category.
**Denominators.** n is shown directly under each bar's label; see the denominator table and caveats above.
**Caveats.** Descriptive published-protocol comparison, not common-query aligned (see above); mismatched denominators for UKBB-EFO and OLS-EFO (full).
**Source data.** `data/graph_relationship_end_to_end.csv`

### Figure 16 — Δ graph relationship vs. text2term (descriptive)

**Files**
- pairwise/figure_16_graph_relationship_delta_vs_text2term.png
- pairwise/figure_16_graph_relationship_delta_vs_text2term.svg

**Question.** By how many percentage points does each graph-relationship category differ between our method (mapped-only view) and the published text2term values?
**Labeling.** Explicitly titled and documented as a **descriptive published-protocol difference** -- the two sides use different, mismatched denominators with no common-query alignment (see above), so this is never framed as, or implying, a statistically validated or common-query comparison. No hypothesis test has been run.
**Source data.** `data/graph_relationship_delta_vs_text2term.csv`

## Which comparison is strongest for publication use

**Figures 13/14 are a published-protocol descriptive comparison, not a common-query-aligned one.** Biomappings-EFO's matching N (795 vs. 795) makes it the most directly comparable of the three, but even there row-level identity was not verified. UKBB-EFO and OLS-EFO (full) have materially different denominators from the original text2term evaluation and must be presented with that caveat. No common-query-aligned analysis (Figure 15) exists in this repository -- if one is produced in the future by explicitly vendoring and documenting the `rsgoncalves/text2term-evaluation` raw output files (Part 19), it would be the stronger, preferred comparison and should supersede Figures 13/14/16 for any claim of near-identical row-level comparison.

# Common-query-aligned comparison with original text2term

This section supersedes the descriptive comparison above with a STRICT common-query-aligned comparison: the same benchmark records, evaluated by both LLM Ontology Mapper and the original text2term evaluation, classified with the identical EFO v3.62.0 graph evaluator.

## Source and provenance

Vendored via `scripts/fetch_text2term_evaluation_outputs.py` from repository `https://github.com/rsgoncalves/text2term-evaluation`, pinned commit `b999dbb670fa13c9ceb1ba631a7abc7557f3293b` -- the SAME commit already used as this repository's graph-evaluator reference (audited: that commit's git tree contains all nine required `output/*.{tsv,csv}` files, so no second commit was ever consulted). Files: `UKBB-EFO_results.tsv`, `Biomappings_results.tsv`, `OLS-EFO_results.tsv` (the row-level per-query comparison outputs used for alignment), plus the corresponding `_mappings.tsv` and `_t2t_mappings.csv` files fetched for completeness. Every file's SHA256 was verified against a pinned expected value before being accepted; the exact hashes and fetch timestamp are recorded in `data/text2term_evaluation/original_outputs/provenance.json`. The fetch step is the ONLY part of this whole workflow that touches the network -- alignment and plotting read only these already-vendored files.

## Alignment identity and normalization

**Alignment key**: `(normalized(source_query), normalized(benchmark_gold_curie))`. This was chosen after auditing candidate keys: the upstream `Source Term ID` column is 100% unique within each vendored file but is the ORIGINAL benchmark source's own row identifier (never persisted in our own records), so it cannot serve as a cross-dataset join key. `(source_term, gold)` pair identity, by contrast, is directly present and near-uniquely populated on both sides.

**Normalization (conservative, documented, no fuzzy matching)**: Unicode NFC normalization + whitespace trim on source text; the same plus canonicalizing ONE well-known CURIE-prefix alias (`Orphanet:` <-> `ORDO:`, both naming the Orphanet Rare Disease Ontology identifier space) on gold codes, uppercasing only the prefix (local codes are preserved verbatim, never case-folded). This alias was not a guess: all 27 UKBB-EFO rows whose only difference from a text2term row was this exact prefix spelling resolved cleanly once aliased, taking UKBB-EFO's match rate from 97.0% to 100%. No edit distance, embeddings, lowercasing of arbitrary identifiers, or label-similarity matching was used anywhere in this alignment.

**An incidental but important correctness finding**: our own vendored `efo_edges.tsv` indexes Orphanet Rare Disease Ontology nodes under the `ORDO:` prefix, not `Orphanet:`. Our OWN persisted Scenario 1 UKBB-EFO run stores gold codes as `Orphanet:<id>` (unaliased) for 27 rows, meaning `classify()` against that literal gold code cannot find the node and silently falls through to the absent-node "Unrelated" result for those rows in our *already-persisted* `graph_relationship` column -- this is a pre-existing data-quality quirk in completed Scenario 1 output, NOT introduced or fixed here (this task does not modify completed run outputs). This alignment module always canonicalizes the gold prefix before calling `classify()` for both methods, so the aligned figures in this section are unaffected; the `our_original_graph_relationship` column in `text2term_common_query_alignment_rows.csv` preserves the original (potentially prefix-degraded) persisted value for comparison against `ours_recomputed_relationship`.

## Duplicate / ambiguity policy

Exact-duplicate upstream rows (identical source term, gold, t2t prediction, AND classification) are safely collapsed to one representative row. Duplicate identity keys whose rows DISAGREE on any of those fields are marked ambiguous and excluded from the strict aligned analysis entirely (never resolved by picking the first row) -- see `text2term_alignment_unmatched.csv` for every excluded record and its reason.

## OLS-EFO single-gold restriction

The original text2term protocol evaluated each benchmark record against exactly one benchmark mapping. Our OLS-EFO Scenario 1 run supports multiple acceptable golds per query (7,257 single-gold, 113 with 2 golds, 7 with 3), which would silently advantage our method if multi-gold queries were included in a comparison against text2term's single-gold protocol. The PRIMARY strict OLS-EFO alignment is therefore restricted to our 7,257 single-gold queries only; multi-gold queries are excluded from this alignment entirely (not scored, not credited, not penalized). UKBB-EFO and Biomappings-EFO required no such restriction -- both are verified 100% single-gold by their own `original_mapping_pair_count` field in `unique_queries.csv` (the same field/definition `dataset_validation.json`'s `gold_count_distribution` uses; naively counting `|` characters in the `gold_codes` string is NOT equivalent -- a handful of UKBB-EFO rows carry a single canonical gold whose own composite source-benchmark label happens to already contain literal `|` text).

## Alignment quality

| Benchmark | Our N | t2t N | Our single-gold N | Candidate matches | Strict matched N | Ambiguous | Gold mismatch | Match rate (ours) | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UKBB-EFO | 888 | 899 | 888 | 888 | 888 | 0 | 0 | 100.0% | **STRONG** |
| Biomappings-EFO | 795 | 795 | 795 | 794 | 794 | 0 | 0 | 99.9% | **STRONG** |
| OLS-EFO (full) | 7,504 | 8,143 | 7,257 | 7,255 | 7,255 | 0 | 0 | 100.0% | **STRONG** |

Quality thresholds (reporting heuristics, not statistical rules; never tuned to hit a target): >=95% STRONG, 90-95% GOOD (disclose exclusions), 75-90% PARTIAL (supplementary only), <75% insufficient for a primary claim.

## Table-1 reproducibility check (Part 4)

Before aligning, the published Table-1 aggregate was recomputed directly from the vendored raw per-row `Classification` column (never trusted as primary for this analysis -- see Part 4) and compared against the earlier-recorded published baseline used by Figures 13/14 as a reproducibility check:

- **UKBB-EFO**: disagreement found and investigated -- Same: recomputed=658 vs. published=660, More Specific: recomputed=36 vs. published=34.
- **Biomappings-EFO**: exact agreement on all five categories.
- **OLS-EFO (full)**: disagreement found and investigated -- More Specific: recomputed=55 vs. published=91, More General: recomputed=91 vs. published=55.

These small discrepancies were investigated (not concealed): the raw per-row data is internally self-consistent (every benchmark's per-row tally sums exactly to its own n) and matches Biomappings-EFO's published values exactly; UKBB-EFO differs by 2/899 rows between Same and More Specific, and OLS-EFO's More Specific/More General counts appear swapped relative to the earlier-recorded published values (91 and 55 exchanged) -- most consistent with a transcription/column-order slip when those published values were originally recorded, not a wrong file or wrong graph version. Per Part 4, this analysis treats the raw vendored per-row data as authoritative rather than blocking on this small, explained discrepancy; full counts are in `data/text2term_table1_reproducibility_check.csv`.

## text2term stored-vs-recomputed classification agreement (Part 15)

For every STRICT matched row, text2term's own stored `Classification` was compared against the classification our reused `scenario1_graph_distance.classify()` computes for text2term's Top-1 prediction against the same gold code:

| Benchmark | Matched N | Agreement N | Disagreement N | Agreement rate |
| --- | --- | --- | --- | --- |
| UKBB-EFO | 888 | 888 | 0 | 100.00% |
| Biomappings-EFO | 794 | 794 | 0 | 100.00% |
| OLS-EFO (full) | 7,255 | 7,255 | 0 | 100.00% |

High agreement here is strong evidence that this repository's from-scratch reimplementation of `compare_mappings()` faithfully reproduces the original graph-comparison semantics on real data, not just on the priority-order/EFO-version metadata check.

## Aligned graph-relationship distributions

| Benchmark | Method | Same | More Specific | More General | Sibling | Unrelated | No Top-1 prediction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UKBB-EFO | LLM Ontology Mapper | 79.2% | 1.7% | 2.7% | 1.4% | 14.6% | 0.5% |
| UKBB-EFO | text2term | 73.2% | 4.1% | 2.3% | 1.5% | 19.0% | 0.0% |
| Biomappings-EFO | LLM Ontology Mapper | 95.5% | 0.0% | 0.1% | 0.5% | 2.1% | 1.8% |
| Biomappings-EFO | text2term | 78.7% | 0.0% | 0.3% | 5.9% | 15.1% | 0.0% |
| OLS-EFO (full) | LLM Ontology Mapper | 83.6% | 0.7% | 0.9% | 0.5% | 12.7% | 1.5% |
| OLS-EFO (full) | text2term | 81.0% | 0.6% | 1.2% | 1.1% | 16.0% | 0.0% |

## Paired exact-match transitions and McNemar's test

Because aligned rows are paired (both methods scored on the identical record), row-level rescue behavior is quantified directly rather than by subtracting aggregate percentages:

| Benchmark | Both exact | Ours only | text2term only | Neither | Aligned N |
| --- | --- | --- | --- | --- | --- |
| UKBB-EFO | 616 | 87 | 34 | 151 | 888 |
| Biomappings-EFO | 601 | 157 | 24 | 12 | 794 |
| OLS-EFO (full) | 5,742 | 326 | 136 | 1,051 | 7,255 |

**McNemar's exact test** (binomial, on the discordant pairs -- appropriate specifically because Top-1-exact correctness is paired on identical records here):

| Benchmark | Ours-only correct | text2term-only correct | Discordant N | p-value |
| --- | --- | --- | --- | --- |
| UKBB-EFO | 87 | 34 | 121 | 1.57e-06 |
| Biomappings-EFO | 157 | 24 | 181 | 3.84e-25 |
| OLS-EFO (full) | 326 | 136 | 462 | 4.57e-19 |

A small p-value indicates the discordant pairs are asymmetric beyond chance -- it does NOT by itself establish which method is better in any absolute sense, only that the two methods' Top-1-exact outcomes disagree asymmetrically on this aligned set.

**Paired bootstrap CI on the Same-proportion difference**: not implemented in this pass (Part 22 marks it explicitly optional) -- flagged here rather than silently omitted.

### Figure 15 — Common-query-aligned Top-1 graph relationship comparison (PRIMARY)

**Files**
- pairwise/figure_15_graph_relationships_common_query_aligned.png
- pairwise/figure_15_graph_relationships_common_query_aligned.svg

**Question.** On the SAME benchmark records, evaluated by both methods, what is the Top-1 graph-relationship composition, end-to-end (including no-prediction)?
**Datasets.** UKBB-EFO, Biomappings-EFO, OLS-EFO (strict single-gold common subset) -- three panels.
**Methods.** LLM Ontology Mapper, text2term -- two 100%-stacked bars per panel, both bars sharing the IDENTICAL aligned N (enforced by a hard assertion in the plotting code).
**Categories.** Same, More Specific, More General, Sibling, Unrelated, No Top-1 prediction -- same category colors as Figures 13/14.
**Denominators.** Aligned n shown under each bar; both bars in a panel always match by construction.
**Caveats.** OLS-EFO panel is restricted to the single-gold common subset (see above) -- not the full 7,377-query OLS-EFO run.
**Source data.** `data/text2term_common_query_alignment_rows.csv`, `data/text2term_common_query_alignment_summary.csv`

### Figure 15b — Common-query-aligned, classifiable predictions only (supplementary)

**Files**
- pairwise/figure_15b_graph_relationships_common_query_aligned_mapped_only.png
- pairwise/figure_15b_graph_relationships_common_query_aligned_mapped_only.svg

**Question.** Given that a method emitted a classifiable Top-1 mapping on the aligned subset, how related was it to gold? Each method's "No Top-1 prediction" rows are excluded and the remaining five categories independently renormalized to 100% per method (so the two bars in a panel may have different denominators here, shown under each bar) -- supplementary to Figure 15, which remains primary because it does not normalize away abstention.
**Source data.** `data/text2term_common_query_alignment_rows.csv`

### Figure 15c — Paired exact-match transitions (supplementary)

**Files**
- pairwise/figure_15c_exact_match_transitions_vs_text2term.png
- pairwise/figure_15c_exact_match_transitions_vs_text2term.svg

**Question.** How many aligned records did each method get exactly right ("Same"), broken down into both-correct / ours-only / text2term-only / neither? Three 2x2 grids (UKBB-EFO, Biomappings-EFO, OLS-EFO single-gold subset), each cell annotated with count and percentage of that panel's aligned N.
**Source data.** `data/text2term_aligned_exact_match_transitions.csv`

## Recommendation

**Figure 15 is promoted to the PRIMARY graph-relationship comparison for manuscript use.** Every benchmark reached STRONG or GOOD alignment quality (>=90% of our eligible single-gold records matched deterministically) and text2term's own stored Classification agreed with our independently recomputed classification on effectively all matched rows (>=99%) -- both conditions required by this suite's promotion policy are satisfied.

