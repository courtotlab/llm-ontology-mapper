"""
Scenario 1 (OLS-EFO) dataset loading, auditing, and canonical-query grouping.

Source schema (see benchmarking/scenario1_dataset.py callers):
    query          -- free-text source term
    ref_match      -- gold EFO label
    ref_match_id   -- gold EFO code (or, in principle, any ontology code)

The raw file has one row per (query, gold) *mapping pair* -- the same query
string can repeat across multiple rows (identical gold pair, or a second
distinct gold code). Sending every raw row through the LLM would re-run the
same query many times for no benefit, so this module builds a canonical
unique-query dataset: one row per distinct `query` string, carrying the set
of all gold codes/labels ever paired with it.

Two denominators fall out of this and must never be silently conflated (see
scenario1_metrics.py):
    - unique-query   (primary):   one row per distinct query string
    - mapping-pair   (secondary): one row per distinct (query, ref_match_id) pair
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

REQUIRED_COLUMNS: tuple[str, ...] = ("query", "ref_match", "ref_match_id")

# ─────────────────────────────────────────────────────────────────────────────
# Multi-gold cell parsing (gold-parsing-bug fix)
#
# UKBB-EFO.csv is the only Scenario 1 source dataset that ever encodes more
# than one acceptable gold code in a single `ref_match_id` cell, joined by
# " | " -- verified: zero occurrences of any pipe delimiter in
# OLS-EFO_full.csv/Biomappings-EFO.csv, and every "|" in UKBB-EFO.csv's
# ref_match_id column is this exact " | " form (never a bare "|", never
# inside a single code token). Splitting unconditionally on " | " is
# therefore a no-op for the other two datasets and requires no per-dataset
# branching. Never split on ":" (that would break a single ontology code).
# ─────────────────────────────────────────────────────────────────────────────

GOLD_CODE_DELIMITER = " | "

# ref_match (label) delimiter is NOT consistent with the code delimiter --
# observed conventions are "||" (no spaces) and " | " (spaces), and at least
# one row (UKBB query "Paraplegia and tetraplegia") has only ONE label for
# TWO codes. Labels are display metadata only -- never consulted by scoring
# (PredictionRecord carries no gold_labels field) -- so a wrong guess here
# carries zero scoring risk, but we still never invent a pairing: a label is
# only attributed to a specific code when a delimiter split produces exactly
# as many labels as there are codes.
GOLD_LABEL_DELIMITERS: tuple[str, ...] = ("||", " | ")


def parse_gold_codes(raw_ref_match_id: Any) -> list[str]:
    """Split one raw `ref_match_id` cell into its acceptable gold code(s).

        "EFO:0009679"                    -> ["EFO:0009679"]
        "EFO:0009679 | EFO:0009684"      -> ["EFO:0009679", "EFO:0009684"]

    Trims whitespace around every resulting token and drops blank tokens
    defensively. Never splits on ":", ",", ";", or any delimiter other than
    the single verified " | " form.
    """
    if raw_ref_match_id is None:
        return []
    parts = [p.strip() for p in str(raw_ref_match_id).split(GOLD_CODE_DELIMITER)]
    return [p for p in parts if p]


def parse_gold_labels(raw_ref_match: Any, *, code_count: int) -> list[str | None]:
    """Split one raw `ref_match` cell into per-code labels, ONLY when a known
    delimiter's split count exactly matches `code_count`. If no delimiter is
    present and there is exactly one code, the whole (unsplit) label is
    attributed to it -- the existing single-gold behavior. If counts never
    match (or the cell is blank), every code's label is `None` (the existing
    missing-label sentinel) rather than an invented pairing."""
    if _blank(raw_ref_match):
        return [None] * code_count
    text = str(raw_ref_match)
    for delimiter in GOLD_LABEL_DELIMITERS:
        if delimiter not in text:
            continue
        parts: list[str | None] = [p.strip() for p in text.split(delimiter)]
        parts = [p if p else None for p in parts]
        if len(parts) == code_count:
            return parts
    if code_count == 1:
        return [text]
    return [None] * code_count


class Scenario1DatasetError(ValueError):
    """Raised when the OLS-EFO dataset does not satisfy the required schema."""


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return not str(value).strip()


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the OLS-EFO CSV and validate it has the required columns.

    Does NOT drop, dedupe, lowercase, or otherwise normalize rows -- that is
    the job of audit_dataset()/build_canonical_queries() so every transform
    stays inspectable and separately testable.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise Scenario1DatasetError(f"OLS-EFO dataset not found: {resolved}")

    df = pd.read_csv(resolved, dtype=str, keep_default_na=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise Scenario1DatasetError(
            f"{resolved} is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}. Found: {list(df.columns)}."
        )
    df = df.reset_index(drop=True)
    df.insert(0, "raw_row_index", df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 -- dataset audit
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetAudit:
    """Exact, derived-from-file counts required before launching the paid run."""

    raw_row_count: int
    missing_query_count: int
    missing_ref_match_count: int
    missing_ref_match_id_count: int
    ref_match_id_prefix_counts: dict[str, int]
    exact_duplicate_row_count: int  # extra copies beyond the first, over all 3 cols
    unique_mapping_pair_count: int  # unique (query, ref_match_id)
    unique_query_count: int
    gold_count_distribution: dict[int, int]  # {num_gold_codes: num_queries_with_that_count}
    max_gold_codes_per_query: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_row_count": self.raw_row_count,
            "missing_query_count": self.missing_query_count,
            "missing_ref_match_count": self.missing_ref_match_count,
            "missing_ref_match_id_count": self.missing_ref_match_id_count,
            "ref_match_id_prefix_counts": dict(self.ref_match_id_prefix_counts),
            "exact_duplicate_row_count": self.exact_duplicate_row_count,
            "unique_mapping_pair_count": self.unique_mapping_pair_count,
            "unique_query_count": self.unique_query_count,
            "gold_count_distribution": {str(k): v for k, v in self.gold_count_distribution.items()},
            "max_gold_codes_per_query": self.max_gold_codes_per_query,
        }


def audit_dataset(df: pd.DataFrame) -> DatasetAudit:
    """Derive every Part 2 count directly from the loaded frame. Never forces
    a target number -- if the file changes, these counts change with it."""
    raw_row_count = len(df)
    missing_query_count = int(df["query"].apply(_blank).sum())
    missing_ref_match_count = int(df["ref_match"].apply(_blank).sum())
    missing_ref_match_id_count = int(df["ref_match_id"].apply(_blank).sum())

    # Explode each row's ref_match_id cell into its parsed gold code(s) --
    # same parsing contract as build_canonical_queries() (parse_gold_codes),
    # so a compound UKBB cell like "EFO:0009676 | EFO:1001986" counts as TWO
    # gold codes here too, not one. This is a no-op for OLS/Biomappings
    # (every cell there parses to exactly one code, identical to the raw
    # trimmed value).
    per_row_codes: list[list[str]] = [
        [] if _blank(cell) else parse_gold_codes(cell) for cell in df["ref_match_id"]
    ]

    exploded_codes = [code for codes in per_row_codes for code in codes]
    ref_match_id_prefix_counts = dict(Counter(code.split(":", 1)[0] for code in exploded_codes))

    exact_dup_mask = df.duplicated(subset=["query", "ref_match", "ref_match_id"], keep="first")
    exact_duplicate_row_count = int(exact_dup_mask.sum())

    unique_pairs_set: set[tuple[str, str]] = set()
    for query, codes in zip(df["query"], per_row_codes, strict=True):
        if _blank(query):
            continue
        for code in codes:
            unique_pairs_set.add((query, code))
    unique_mapping_pair_count = len(unique_pairs_set)

    unique_query_count = len({query for query, _ in unique_pairs_set})

    gold_per_query: dict[str, set[str]] = defaultdict(set)
    for query, code in unique_pairs_set:
        gold_per_query[query].add(code)
    gold_count_distribution: dict[int, int] = dict(Counter(len(codes) for codes in gold_per_query.values()))
    max_gold_codes_per_query = max((len(codes) for codes in gold_per_query.values()), default=0)

    return DatasetAudit(
        raw_row_count=raw_row_count,
        missing_query_count=missing_query_count,
        missing_ref_match_count=missing_ref_match_count,
        missing_ref_match_id_count=missing_ref_match_id_count,
        ref_match_id_prefix_counts=ref_match_id_prefix_counts,
        exact_duplicate_row_count=exact_duplicate_row_count,
        unique_mapping_pair_count=unique_mapping_pair_count,
        unique_query_count=unique_query_count,
        gold_count_distribution=gold_count_distribution,
        max_gold_codes_per_query=max_gold_codes_per_query,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 -- canonical unique-query dataset
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CanonicalQuery:
    """One unique source query with every gold code/label ever paired with it.

    query_id is a stable 0-based index assigned in order of first appearance
    in the raw file -- reproducible across runs on the same dataset file.
    """

    query_id: int
    source_query: str
    gold_codes: list[str] = field(default_factory=list)
    gold_labels: list[str | None] = field(default_factory=list)
    gold_first_row_indices: list[int] = field(default_factory=list)
    original_row_indices: list[int] = field(default_factory=list)

    @property
    def original_mapping_pair_count(self) -> int:
        return len(self.gold_codes)

    @property
    def gold_count(self) -> int:
        return len(self.gold_codes)


def build_canonical_queries(df: pd.DataFrame) -> list[CanonicalQuery]:
    """Group raw rows into one CanonicalQuery per distinct `query` string.

    Rows with a blank query are excluded entirely (nothing to map). Rows with
    a blank ref_match_id contribute no gold code but their raw_row_index is
    still recorded against the query via original_row_indices for
    traceability. Within a query, gold codes/labels are kept in first-seen
    order and deduplicated by ref_match_id; the dataset is exact-matched
    (query, ref_match_id, ref_match) is 1:1, so the first-seen ref_match
    label for a given ref_match_id is authoritative.

    No lowercasing, whitespace normalization, or fuzzy matching is applied --
    query identity is exact string equality on the raw `query` cell.
    """
    queries_in_order: list[str] = []
    seen_queries: dict[str, int] = {}
    per_query_row_indices: dict[str, list[int]] = {}
    per_query_gold_codes: dict[str, list[str]] = {}
    per_query_gold_labels: dict[str, list[str | None]] = {}
    per_query_gold_first_row: dict[str, list[int]] = {}
    per_query_seen_codes: dict[str, set[str]] = {}

    for record in df.itertuples(index=False):
        query = record.query
        if _blank(query):
            continue
        raw_row_index = int(record.raw_row_index)

        if query not in seen_queries:
            seen_queries[query] = len(queries_in_order)
            queries_in_order.append(query)
            per_query_row_indices[query] = []
            per_query_gold_codes[query] = []
            per_query_gold_labels[query] = []
            per_query_gold_first_row[query] = []
            per_query_seen_codes[query] = set()

        per_query_row_indices[query].append(raw_row_index)

        ref_match_id = record.ref_match_id
        if _blank(ref_match_id):
            continue
        codes = parse_gold_codes(ref_match_id)
        if not codes:
            continue
        labels = parse_gold_labels(record.ref_match, code_count=len(codes))
        for code, label in zip(codes, labels, strict=True):
            # Dedup at the individual-code level (not the raw-cell level) so
            # a query with e.g. one row "CODE:A | CODE:B" and a second row
            # "CODE:A" contributes exactly {CODE:A, CODE:B}, not three entries.
            if code not in per_query_seen_codes[query]:
                per_query_seen_codes[query].add(code)
                per_query_gold_codes[query].append(code)
                per_query_gold_first_row[query].append(raw_row_index)
                per_query_gold_labels[query].append(label)

    return [
        CanonicalQuery(
            query_id=idx,
            source_query=query,
            gold_codes=per_query_gold_codes[query],
            gold_labels=per_query_gold_labels[query],
            gold_first_row_indices=per_query_gold_first_row[query],
            original_row_indices=per_query_row_indices[query],
        )
        for idx, query in enumerate(queries_in_order)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 -- mapping-pair-expanded (secondary) denominator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MappingPairRow:
    """One row of the SECONDARY (mapping-pair) denominator.

    Reuses the SAME prediction (ranked codes) produced for the parent
    unique query -- never triggers an additional mapper call. Scored against
    a single gold code rather than the query's full gold set.
    """

    query_id: int
    source_query: str
    gold_code: str
    gold_label: str | None
    raw_row_index: int


def expand_to_mapping_pairs(canonical_queries: list[CanonicalQuery]) -> list[MappingPairRow]:
    """Expand canonical queries back out to one row per unique (query, gold)
    pair, for the SECONDARY mapping-pair-expanded evaluation (Part 4/21).
    Preserves the original raw representation without re-querying the mapper.
    """
    rows: list[MappingPairRow] = []
    for cq in canonical_queries:
        for gold_code, gold_label, raw_row_index in zip(
            cq.gold_codes, cq.gold_labels, cq.gold_first_row_indices, strict=True
        ):
            rows.append(
                MappingPairRow(
                    query_id=cq.query_id,
                    source_query=cq.source_query,
                    gold_code=gold_code,
                    gold_label=gold_label,
                    raw_row_index=raw_row_index,
                )
            )
    return rows
