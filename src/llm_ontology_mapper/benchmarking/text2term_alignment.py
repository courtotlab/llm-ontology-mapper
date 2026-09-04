"""
Scenario 1 -- strict common-query alignment between our Top-1 predictions and
the ORIGINAL text2term evaluation's raw per-query outputs (vendored via
scripts/fetch_text2term_evaluation_outputs.py from ONE pinned commit of
https://github.com/rsgoncalves/text2term-evaluation -- the same commit
already used as scenario1_graph_distance.py's graph-evaluator reference).

Goal: identify benchmark records evaluated by BOTH methods via deterministic
exact-identity matching (no fuzzy matching, no embeddings, no edit distance),
then classify BOTH methods' Top-1 predictions against the SAME gold code
using the SAME EfoGraphIndex/classify() (scenario1_graph_distance.py,
reused unmodified) -- producing a comparison where the two methods share an
identical denominator, unlike the descriptive aggregate comparison in
figures/graph_relationship_comparison.py.

Pure analysis logic -- no plotting, no network, no mapper/LLM calls. Every
input is a file already on disk: our own persisted Scenario 1 run artifacts
(unique_queries.csv, mapping_pair_expanded_predictions.csv, predictions.csv)
and the vendored text2term-evaluation output/*.tsv files.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import binomtest  # type: ignore[import-untyped]

from llm_ontology_mapper.benchmarking import scenario1_graph_distance as graph_distance

BENCHMARK_ORDER: tuple[str, ...] = ("UKBB-EFO", "Biomappings-EFO", "OLS-EFO (full)")

# Upstream vendored results.tsv filename per benchmark (Part 2/3).
T2T_RESULTS_FILENAME: dict[str, str] = {
    "UKBB-EFO": "UKBB-EFO_results.tsv",
    "Biomappings-EFO": "Biomappings_results.tsv",
    "OLS-EFO (full)": "OLS-EFO_results.tsv",
}

OUTCOME_CATEGORIES: tuple[str, ...] = (*graph_distance.ALL_RELATIONSHIPS, "No Top-1 prediction")
NO_TOP1_CATEGORY = "No Top-1 prediction"


class AlignmentError(RuntimeError):
    """Base class for every hard-fail condition in this module."""


class ProvenanceError(AlignmentError):
    pass


class SchemaError(AlignmentError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Normalization (Part 8) -- conservative, documented, no fuzzy matching.
# ─────────────────────────────────────────────────────────────────────────────

_CURIE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*):(.+)$")

# Well-known, identity-preserving CURIE prefix alias: "Orphanet" and "ORDO"
# (Orphanet Rare Disease Ontology) name the same identifier space. Audited
# empirically: all 27 UKBB-EFO rows whose ONLY difference from a vendored
# text2term row was this prefix spelling resolved cleanly once aliased (see
# FIGURES.md "Common-query-aligned comparison" for the full audit). This is
# NOT a fuzzy match -- the local code is preserved verbatim and only a known,
# named ontology-prefix synonym is canonicalized. Scoped to this module only
# (the shared ontology_identity.py config does not register this alias, and
# this analysis-only module intentionally never modifies shared mapper
# config or completed run outputs).
_CURIE_PREFIX_ALIASES: dict[str, str] = {"ORPHANET": "ORDO"}


def normalize_source_text(value: str | None) -> str:
    """Unicode NFC normalization + leading/trailing whitespace trim only.
    No case folding, no punctuation stripping, no synonym inference."""
    if value is None:
        return ""
    return unicodedata.normalize("NFC", value).strip()


def normalize_gold_curie(value: str | None) -> str:
    """normalize_source_text() plus canonicalizing a known CURIE-prefix
    alias and uppercasing the prefix (namespace prefixes are conventionally
    case-insensitive). The local code after the colon is preserved verbatim
    -- never case-folded, since local codes can be case sensitive."""
    text = normalize_source_text(value)
    match = _CURIE_RE.match(text)
    if not match:
        return text
    prefix, local = match.groups()
    prefix_upper = _CURIE_PREFIX_ALIASES.get(prefix.upper(), prefix.upper())
    return f"{prefix_upper}:{local.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# Parsing the vendored text2term-evaluation results.tsv (Part 2)
# ─────────────────────────────────────────────────────────────────────────────

# The upstream results.tsv is a raw spreadsheet export: only these seven
# named columns are per-row data. Additional trailing columns (some
# unnamed/duplicated, e.g. a second "IsDisease") are a side aggregate
# summary table appended to the right of the first ~5 rows in the same
# sheet -- audited directly against the vendored file and never read here.
T2T_RESULTS_REQUIRED_FIELDS: tuple[str, ...] = (
    "Source Term ID", "Source Term", "t2t.Mapping", "t2t.MappingLabel",
    "Benchmark.Mapping", "Benchmark.MappingLabel", "Classification",
)


@dataclass(frozen=True)
class T2tResultRow:
    source_term_id: str
    source_term: str
    t2t_mapping: str | None
    t2t_mapping_label: str
    benchmark_mapping: str
    benchmark_mapping_label: str
    classification: str


def load_t2t_results(path: Path) -> list[T2tResultRow]:
    if not path.exists():
        raise AlignmentError(
            f"missing vendored text2term results file: {path} -- run "
            "scripts/fetch_text2term_evaluation_outputs.py first"
        )
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = set(T2T_RESULTS_REQUIRED_FIELDS) - fieldnames
        if missing:
            raise SchemaError(f"{path}: missing expected columns {sorted(missing)} -- possible upstream schema drift")
        rows = []
        for row in reader:
            t2t_mapping = (row["t2t.Mapping"] or "").strip() or None
            rows.append(
                T2tResultRow(
                    source_term_id=row["Source Term ID"],
                    source_term=row["Source Term"],
                    t2t_mapping=t2t_mapping,
                    t2t_mapping_label=row["t2t.MappingLabel"],
                    benchmark_mapping=row["Benchmark.Mapping"],
                    benchmark_mapping_label=row["Benchmark.MappingLabel"],
                    classification=row["Classification"],
                )
            )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: recompute the aggregate from raw per-row data (never trust the
# published Table-1 percentages as primary for this analysis) and use the
# earlier-established published baseline only as a reproducibility CHECK.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReproducibilityCheckResult:
    benchmark: str
    recomputed_counts: dict[str, int]
    published_counts: dict[str, int]
    agreement: bool
    mismatches: dict[str, tuple[int, int]]  # relationship -> (recomputed, published)


def recompute_aggregate_from_raw(rows: list[T2tResultRow]) -> dict[str, int]:
    counts = dict.fromkeys(graph_distance.ALL_RELATIONSHIPS, 0)
    for row in rows:
        if row.classification not in counts:
            raise SchemaError(
                f"unexpected Classification value {row.classification!r} in vendored text2term results -- "
                f"expected one of {graph_distance.ALL_RELATIONSHIPS}"
            )
        counts[row.classification] += 1
    return counts


def check_table1_reproducibility(
    benchmark: str, recomputed_counts: dict[str, int], published_counts: dict[str, int]
) -> ReproducibilityCheckResult:
    """A soft check (never hard-fails on its own): compares the counts
    tallied directly from the vendored raw per-row Classification column
    against the previously-recorded published Table-1 baseline. Any
    disagreement is returned for the caller to report -- never concealed --
    but investigation (see FIGURES.md) found the raw per-row data internally
    self-consistent (sums to n, matches Biomappings exactly) with only small,
    explicable deltas elsewhere, so this module treats the raw data as
    authoritative per Part 4 rather than blocking on the comparison."""
    mismatches = {
        rel: (recomputed_counts[rel], published_counts[rel])
        for rel in graph_distance.ALL_RELATIONSHIPS
        if recomputed_counts[rel] != published_counts[rel]
    }
    return ReproducibilityCheckResult(
        benchmark=benchmark, recomputed_counts=recomputed_counts, published_counts=published_counts,
        agreement=not mismatches, mismatches=mismatches,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Our persisted Scenario 1 records (Part 5) -- never manually typed.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OurMappingPairRow:
    query_id: str
    source_query: str
    gold_code: str
    rank_1_code: str | None
    status: str
    is_single_gold: bool


def load_single_gold_query_ids(run_dir: Path) -> set[str]:
    """A query is single-gold iff its canonical `original_mapping_pair_count`
    (unique_queries.csv) is exactly 1 -- the same field/definition
    dataset_validation.json's gold_count_distribution uses. Naively counting
    "|" separators in `gold_codes` is NOT equivalent: a handful of UKBB-EFO
    rows carry a single canonical gold mapping whose own label/CURIE field
    happens to already contain literal " | " text from the source benchmark
    (e.g. a composite raw label), which is not our multi-gold separator
    convention at all. Verified against dataset_validation.json for all
    three benchmarks before relying on this field."""
    path = run_dir / "unique_queries.csv"
    if not path.exists():
        raise AlignmentError(f"missing unique_queries.csv: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["query_id"] for row in csv.DictReader(fh) if row["original_mapping_pair_count"] == "1"}


def load_our_mapping_pairs(run_dir: Path) -> list[OurMappingPairRow]:
    single_gold_qids = load_single_gold_query_ids(run_dir)
    path = run_dir / "mapping_pair_expanded_predictions.csv"
    if not path.exists():
        raise AlignmentError(f"missing mapping_pair_expanded_predictions.csv: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                OurMappingPairRow(
                    query_id=row["query_id"],
                    source_query=row["source_query"],
                    gold_code=row["gold_code"],
                    rank_1_code=(row["rank_1_code"] or "").strip() or None,
                    status=row["status"],
                    is_single_gold=row["query_id"] in single_gold_qids,
                )
            )
    return rows


def load_our_original_graph_relationship(run_dir: Path) -> dict[str, str]:
    """query_id -> our own already-persisted graph_relationship from
    predictions.csv (query-level; valid for single-gold queries, which is
    all this module ever aligns). Read-only, never modified."""
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise AlignmentError(f"missing predictions.csv: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["query_id"]: row["graph_relationship"] for row in csv.DictReader(fh)}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment (Part 6/7/9/10) -- exact deterministic join only.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnmatchedRecord:
    benchmark: str
    side: str  # "ours" | "t2t"
    source: str
    gold: str
    reason: str


@dataclass(frozen=True)
class AlignedRow:
    benchmark: str
    alignment_id: str
    our_query_id: str
    our_source_original: str
    our_source_normalized: str
    t2t_source_id: str
    t2t_source_original: str
    t2t_source_normalized: str
    our_gold_original: str
    t2t_gold_original: str
    normalized_gold: str
    our_prediction: str | None
    t2t_prediction: str | None
    our_status: str
    our_original_graph_relationship: str
    t2t_original_published_classification: str
    ours_recomputed_relationship: str
    t2t_recomputed_relationship: str
    alignment_status: str


@dataclass(frozen=True)
class AlignmentResult:
    benchmark: str
    ours_total_n: int
    t2t_total_n: int
    ours_single_gold_n: int
    candidate_exact_matches: int
    aligned_rows: list[AlignedRow]
    unmatched: list[UnmatchedRecord]
    ambiguous_n: int
    gold_mismatch_n: int
    reproducibility: ReproducibilityCheckResult

    @property
    def strict_matched_n(self) -> int:
        return len(self.aligned_rows)

    @property
    def ours_unmatched_n(self) -> int:
        return sum(1 for u in self.unmatched if u.side == "ours")

    @property
    def t2t_unmatched_n(self) -> int:
        return sum(1 for u in self.unmatched if u.side == "t2t")

    @property
    def match_rate_ours(self) -> float:
        return self.strict_matched_n / self.ours_single_gold_n if self.ours_single_gold_n else 0.0

    @property
    def match_rate_t2t(self) -> float:
        return self.strict_matched_n / self.t2t_total_n if self.t2t_total_n else 0.0


def align_benchmark(
    benchmark: str, run_dir: Path, t2t_results_path: Path, published_counts: dict[str, int],
) -> AlignmentResult:
    our_rows = load_our_mapping_pairs(run_dir)
    ours_total_n = len(our_rows)
    single_gold_rows = [r for r in our_rows if r.is_single_gold]
    ours_single_gold_n = len(single_gold_rows)

    our_original_relationship = load_our_original_graph_relationship(run_dir)

    t2t_rows = load_t2t_results(t2t_results_path)
    t2t_total_n = len(t2t_rows)
    reproducibility = check_table1_reproducibility(benchmark, recompute_aggregate_from_raw(t2t_rows), published_counts)

    t2t_by_key: dict[tuple[str, str], list[T2tResultRow]] = defaultdict(list)
    for row in t2t_rows:
        key = (normalize_source_text(row.source_term), normalize_gold_curie(row.benchmark_mapping))
        t2t_by_key[key].append(row)

    ambiguous_key_set: set[tuple[str, str]] = set()
    t2t_representative: dict[tuple[str, str], T2tResultRow] = {}
    for key, group in t2t_by_key.items():
        if len(group) == 1:
            t2t_representative[key] = group[0]
            continue
        signature = {(g.t2t_mapping, g.classification) for g in group}
        if len(signature) == 1:
            # Identical duplicates (Part 9) -- safe to collapse to one row.
            t2t_representative[key] = group[0]
        else:
            ambiguous_key_set.add(key)

    aligned_rows: list[AlignedRow] = []
    unmatched: list[UnmatchedRecord] = []
    ambiguous_n = 0
    gold_mismatch_n = 0
    candidate_exact_matches = 0

    for our_row in single_gold_rows:
        our_source_norm = normalize_source_text(our_row.source_query)
        our_gold_norm = normalize_gold_curie(our_row.gold_code)
        key = (our_source_norm, our_gold_norm)

        if key in ambiguous_key_set:
            ambiguous_n += 1
            unmatched.append(UnmatchedRecord(benchmark, "ours", our_row.source_query, our_row.gold_code, "ambiguous duplicate on t2t side"))
            continue

        t2t_row = t2t_representative.get(key)
        if t2t_row is None:
            # Distinguish "same source text, different gold" (gold mismatch
            # between the two benchmark snapshots) from "no identity match
            # at all" for the unmatched-record audit (Part 6/24).
            same_source_different_gold = any(
                normalize_source_text(r.source_term) == our_source_norm for r in t2t_rows
            )
            reason = "gold mismatch: same source term, different benchmark gold CURIE" if same_source_different_gold else "no exact identity match"
            if same_source_different_gold:
                gold_mismatch_n += 1
            unmatched.append(UnmatchedRecord(benchmark, "ours", our_row.source_query, our_row.gold_code, reason))
            continue

        candidate_exact_matches += 1

        # Reuse the SAME EfoGraphIndex/classify() for BOTH methods, against
        # the SAME (alias-canonicalized) gold code (Part 13). Canonicalizing
        # the gold prefix here matters for correctness, not just matching:
        # our own efo_edges.tsv indexes Orphanet Rare Disease Ontology nodes
        # under the "ORDO:" prefix, so an un-aliased "Orphanet:" gold is
        # absent from the graph and would silently fall through to the
        # absent-node "Unrelated" fallback for BOTH sides -- see FIGURES.md
        # for the audited example and the resulting (documented, not
        # silently fixed) divergence from our own persisted
        # `graph_relationship` column for the handful of UKBB rows affected.
        gold_for_classify = our_gold_norm

        our_no_pred = our_row.status in ("unmapped", "error")
        our_pred_for_classify = None if our_no_pred else our_row.rank_1_code
        ours_recomputed = graph_distance.classify(our_pred_for_classify, [gold_for_classify]).graph_relationship

        t2t_pred_for_classify = t2t_row.t2t_mapping
        t2t_recomputed = graph_distance.classify(t2t_pred_for_classify, [gold_for_classify]).graph_relationship

        aligned_rows.append(
            AlignedRow(
                benchmark=benchmark,
                alignment_id=f"{benchmark}::{our_row.query_id}",
                our_query_id=our_row.query_id,
                our_source_original=our_row.source_query,
                our_source_normalized=our_source_norm,
                t2t_source_id=t2t_row.source_term_id,
                t2t_source_original=t2t_row.source_term,
                t2t_source_normalized=normalize_source_text(t2t_row.source_term),
                our_gold_original=our_row.gold_code,
                t2t_gold_original=t2t_row.benchmark_mapping,
                normalized_gold=our_gold_norm,
                our_prediction=our_row.rank_1_code,
                t2t_prediction=t2t_row.t2t_mapping,
                our_status=our_row.status,
                our_original_graph_relationship=our_original_relationship.get(our_row.query_id, ""),
                t2t_original_published_classification=t2t_row.classification,
                ours_recomputed_relationship=(NO_TOP1_CATEGORY if our_no_pred or ours_recomputed == graph_distance.RELATIONSHIP_NOT_APPLICABLE else ours_recomputed),
                t2t_recomputed_relationship=(NO_TOP1_CATEGORY if t2t_pred_for_classify is None or t2t_recomputed == graph_distance.RELATIONSHIP_NOT_APPLICABLE else t2t_recomputed),
                alignment_status="matched",
            )
        )

    # t2t-side unmatched: t2t rows never claimed by any of our aligned rows.
    matched_t2t_keys = {(row.t2t_source_normalized, row.normalized_gold) for row in aligned_rows}
    for key, row in t2t_representative.items():
        if key not in matched_t2t_keys and key not in ambiguous_key_set:
            unmatched.append(UnmatchedRecord(benchmark, "t2t", row.source_term, row.benchmark_mapping, "no exact identity match in our single-gold subset"))
    for key in ambiguous_key_set:
        for row in t2t_by_key[key]:
            unmatched.append(UnmatchedRecord(benchmark, "t2t", row.source_term, row.benchmark_mapping, "ambiguous duplicate on t2t side"))

    return AlignmentResult(
        benchmark=benchmark,
        ours_total_n=ours_total_n,
        t2t_total_n=t2t_total_n,
        ours_single_gold_n=ours_single_gold_n,
        candidate_exact_matches=candidate_exact_matches,
        aligned_rows=aligned_rows,
        unmatched=unmatched,
        ambiguous_n=ambiguous_n,
        gold_mismatch_n=gold_mismatch_n,
        reproducibility=reproducibility,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Six-category outcome distributions from aligned rows (Part 16/17)
# ─────────────────────────────────────────────────────────────────────────────


def outcome_counts(aligned_rows: list[AlignedRow], *, field: str) -> dict[str, int]:
    counts = dict.fromkeys(OUTCOME_CATEGORIES, 0)
    for row in aligned_rows:
        value = getattr(row, field)
        counts[value] += 1
    return counts


def outcome_proportions(counts: dict[str, int], n: int) -> dict[str, float]:
    return {k: (v / n if n else 0.0) for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Paired exact-match transitions + McNemar (Part 19/21)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairedTransitions:
    both_exact: int
    ours_only_exact: int
    t2t_only_exact: int
    neither_exact: int

    @property
    def total(self) -> int:
        return self.both_exact + self.ours_only_exact + self.t2t_only_exact + self.neither_exact


def compute_paired_transitions(aligned_rows: list[AlignedRow]) -> PairedTransitions:
    both = ours_only = t2t_only = neither = 0
    for row in aligned_rows:
        ours_exact = row.ours_recomputed_relationship == graph_distance.RELATIONSHIP_SAME
        t2t_exact = row.t2t_recomputed_relationship == graph_distance.RELATIONSHIP_SAME
        if ours_exact and t2t_exact:
            both += 1
        elif ours_exact:
            ours_only += 1
        elif t2t_exact:
            t2t_only += 1
        else:
            neither += 1
    return PairedTransitions(both_exact=both, ours_only_exact=ours_only, t2t_only_exact=t2t_only, neither_exact=neither)


@dataclass(frozen=True)
class McNemarResult:
    benchmark: str
    ours_only_correct: int
    t2t_only_correct: int
    discordant_n: int
    statistic_method: str
    p_value: float


def compute_mcnemar(benchmark: str, transitions: PairedTransitions) -> McNemarResult:
    """Exact (binomial) McNemar test on the discordant pairs -- appropriate
    here because Top-1-exact correctness is paired on identical aligned
    records for both methods (Part 21)."""
    b, c = transitions.ours_only_exact, transitions.t2t_only_exact
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(
        benchmark=benchmark, ours_only_correct=b, t2t_only_correct=c, discordant_n=discordant,
        statistic_method="exact binomial McNemar (scipy.stats.binomtest)", p_value=p_value,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alignment-quality threshold (Part 23) -- reporting heuristic, not a
# statistical rule; never manipulated to hit a target.
# ─────────────────────────────────────────────────────────────────────────────


def alignment_quality_label(match_rate_ours: float) -> str:
    if match_rate_ours >= 0.95:
        return "STRONG"
    if match_rate_ours >= 0.90:
        return "GOOD"
    if match_rate_ours >= 0.75:
        return "PARTIAL"
    return "INSUFFICIENT"


# ─────────────────────────────────────────────────────────────────────────────
# Data writers
# ─────────────────────────────────────────────────────────────────────────────


def _write_csv(rows: list[dict[str, Any]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def write_alignment_summary_csv(results: dict[str, AlignmentResult], path: Path) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        r = results[benchmark]
        rows.append(
            {
                "benchmark": benchmark,
                "ours_total_n": r.ours_total_n,
                "t2t_total_n": r.t2t_total_n,
                "ours_single_gold_n": r.ours_single_gold_n,
                "candidate_exact_matches": r.candidate_exact_matches,
                "strict_matched_n": r.strict_matched_n,
                "ours_unmatched_n": r.ours_unmatched_n,
                "t2t_unmatched_n": r.t2t_unmatched_n,
                "ambiguous_n": r.ambiguous_n,
                "gold_mismatch_n": r.gold_mismatch_n,
                "match_rate_ours": r.match_rate_ours,
                "match_rate_t2t": r.match_rate_t2t,
                "alignment_key": "normalized(source_query) + normalized(benchmark_gold_curie)",
            }
        )
    _write_csv(
        rows,
        ["benchmark", "ours_total_n", "t2t_total_n", "ours_single_gold_n", "candidate_exact_matches",
         "strict_matched_n", "ours_unmatched_n", "t2t_unmatched_n", "ambiguous_n", "gold_mismatch_n",
         "match_rate_ours", "match_rate_t2t", "alignment_key"],
        path,
    )


_ALIGNED_ROW_FIELDS: list[str] = [
    "benchmark", "alignment_id", "our_query_id", "our_source_original", "our_source_normalized",
    "t2t_source_id", "t2t_source_original", "t2t_source_normalized", "our_gold_original", "t2t_gold_original",
    "normalized_gold", "our_prediction", "t2t_prediction", "our_status", "our_original_graph_relationship",
    "t2t_original_published_classification", "ours_recomputed_relationship", "t2t_recomputed_relationship",
    "alignment_status",
]


def write_aligned_rows_csv(results: dict[str, AlignmentResult], path: Path) -> None:
    rows = [row.__dict__ for benchmark in BENCHMARK_ORDER for row in results[benchmark].aligned_rows]
    _write_csv(rows, _ALIGNED_ROW_FIELDS, path)


def write_unmatched_csv(results: dict[str, AlignmentResult], path: Path) -> None:
    rows = [
        {"benchmark": u.benchmark, "side": u.side, "source": u.source, "gold": u.gold, "reason": u.reason}
        for benchmark in BENCHMARK_ORDER
        for u in results[benchmark].unmatched
    ]
    _write_csv(rows, ["benchmark", "side", "source", "gold", "reason"], path)


def write_reclassification_audit_csv(results: dict[str, AlignmentResult], path: Path) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        r = results[benchmark]
        matched_n = len(r.aligned_rows)
        agreement_n = sum(
            1 for row in r.aligned_rows if row.t2t_original_published_classification == row.t2t_recomputed_relationship
        )
        rows.append(
            {
                "benchmark": benchmark,
                "matched_n": matched_n,
                "agreement_n": agreement_n,
                "disagreement_n": matched_n - agreement_n,
                "agreement_rate": (agreement_n / matched_n if matched_n else 0.0),
            }
        )
    _write_csv(rows, ["benchmark", "matched_n", "agreement_n", "disagreement_n", "agreement_rate"], path)


def write_table1_reproducibility_csv(results: dict[str, AlignmentResult], path: Path) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        check = results[benchmark].reproducibility
        for rel in graph_distance.ALL_RELATIONSHIPS:
            recomputed = check.recomputed_counts[rel]
            published = check.published_counts[rel]
            rows.append(
                {
                    "benchmark": benchmark, "relationship": rel, "recomputed_count": recomputed,
                    "published_count": published, "agrees": recomputed == published,
                }
            )
    _write_csv(rows, ["benchmark", "relationship", "recomputed_count", "published_count", "agrees"], path)


def write_transitions_csv(transitions: dict[str, PairedTransitions], path: Path) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        t = transitions[benchmark]
        rows.append(
            {
                "benchmark": benchmark, "both_exact": t.both_exact, "ours_only_exact": t.ours_only_exact,
                "t2t_only_exact": t.t2t_only_exact, "neither_exact": t.neither_exact, "aligned_n": t.total,
            }
        )
    _write_csv(rows, ["benchmark", "both_exact", "ours_only_exact", "t2t_only_exact", "neither_exact", "aligned_n"], path)


def write_mcnemar_csv(mcnemar: dict[str, McNemarResult], path: Path) -> None:
    rows = []
    for benchmark in BENCHMARK_ORDER:
        m = mcnemar[benchmark]
        rows.append(
            {
                "benchmark": benchmark, "ours_only_correct": m.ours_only_correct, "t2t_only_correct": m.t2t_only_correct,
                "discordant_n": m.discordant_n, "statistic_method": m.statistic_method, "p_value": m.p_value,
            }
        )
    _write_csv(rows, ["benchmark", "ours_only_correct", "t2t_only_correct", "discordant_n", "statistic_method", "p_value"], path)
