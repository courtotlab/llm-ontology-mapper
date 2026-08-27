"""
Unit tests for llm_ontology_mapper.benchmarking.scenario1_dataset.

Run with:  pytest tests/benchmarking/test_scenario1_dataset.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_ontology_mapper.benchmarking.scenario1_dataset import (
    Scenario1DatasetError,
    audit_dataset,
    build_canonical_queries,
    expand_to_mapping_pairs,
    load_raw_dataset,
)

pytestmark = pytest.mark.unit


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. dataset columns validated
# ─────────────────────────────────────────────────────────────────────────────


def test_load_raw_dataset_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, [{"query": "a", "ref_match": "b"}])  # missing ref_match_id
    with pytest.raises(Scenario1DatasetError, match="missing required columns"):
        load_raw_dataset(path)


def test_load_raw_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(Scenario1DatasetError, match="not found"):
        load_raw_dataset(tmp_path / "nope.csv")


def test_load_raw_dataset_assigns_stable_raw_row_index(tmp_path: Path) -> None:
    path = tmp_path / "ok.csv"
    _write_csv(
        path,
        [
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},
            {"query": "b", "ref_match": "B label", "ref_match_id": "EFO:0000002"},
        ],
    )
    df = load_raw_dataset(path)
    assert df["raw_row_index"].tolist() == [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. duplicate raw rows handled correctly (exact duplicates)
# ─────────────────────────────────────────────────────────────────────────────


def test_audit_counts_exact_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "dupe.csv"
    _write_csv(
        path,
        [
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},  # exact dup
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},  # exact dup
            {"query": "b", "ref_match": "B label", "ref_match_id": "EFO:0000002"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)

    assert audit.raw_row_count == 4
    assert audit.exact_duplicate_row_count == 2  # extra copies beyond the first


# ─────────────────────────────────────────────────────────────────────────────
# 3. duplicate query/gold pairs handled correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_audit_counts_unique_mapping_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    _write_csv(
        path,
        [
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},
            {"query": "a variant", "ref_match": "A label", "ref_match_id": "EFO:0000001"},  # same pair, diff query
            {"query": "a", "ref_match": "A label", "ref_match_id": "EFO:0000001"},  # exact dup of row 1
            {"query": "b", "ref_match": "B label", "ref_match_id": "EFO:0000002"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)

    # unique (query, ref_match_id): (a,EFO1), (a variant,EFO1), (b,EFO2) = 3
    assert audit.unique_mapping_pair_count == 3
    assert audit.unique_query_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. multi-gold queries grouped correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_build_canonical_queries_groups_multi_gold_query(tmp_path: Path) -> None:
    path = tmp_path / "multi.csv"
    _write_csv(
        path,
        [
            {"query": "leukemia", "ref_match": "Leukemia A", "ref_match_id": "EFO:0000001"},
            {"query": "leukemia", "ref_match": "Leukemia B", "ref_match_id": "EFO:0000002"},
            {"query": "leukemia", "ref_match": "Leukemia A", "ref_match_id": "EFO:0000001"},  # dup pair
            {"query": "other", "ref_match": "Other", "ref_match_id": "EFO:0000003"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)

    assert len(cqs) == 2
    leukemia = next(cq for cq in cqs if cq.source_query == "leukemia")
    assert leukemia.gold_codes == ["EFO:0000001", "EFO:0000002"]
    assert leukemia.gold_labels == ["Leukemia A", "Leukemia B"]
    assert leukemia.gold_count == 2
    assert leukemia.original_mapping_pair_count == 2
    assert leukemia.original_row_indices == [0, 1, 2]


def test_audit_gold_count_distribution(tmp_path: Path) -> None:
    path = tmp_path / "dist.csv"
    _write_csv(
        path,
        [
            {"query": "single", "ref_match": "S", "ref_match_id": "EFO:1"},
            {"query": "double", "ref_match": "D1", "ref_match_id": "EFO:2"},
            {"query": "double", "ref_match": "D2", "ref_match_id": "EFO:3"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.gold_count_distribution == {1: 1, 2: 1}
    assert audit.max_gold_codes_per_query == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. no case/punctuation normalization changes query identity
# ─────────────────────────────────────────────────────────────────────────────


def test_query_identity_is_exact_string_no_normalization(tmp_path: Path) -> None:
    path = tmp_path / "case.csv"
    _write_csv(
        path,
        [
            {"query": "Cough", "ref_match": "Cough", "ref_match_id": "EFO:1"},
            {"query": "cough", "ref_match": "Cough", "ref_match_id": "EFO:1"},
            {"query": "cough ", "ref_match": "Cough", "ref_match_id": "EFO:1"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    # Three distinct raw strings -> three distinct canonical queries;
    # no lowercasing/stripping applied.
    assert {cq.source_query for cq in cqs} == {"Cough", "cough", "cough "}
    assert len(cqs) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. expected denominator derivation (matches the file's real numbers)
# ─────────────────────────────────────────────────────────────────────────────


def test_denominators_match_manually_computed_expectation(tmp_path: Path) -> None:
    path = tmp_path / "denom.csv"
    _write_csv(
        path,
        [
            {"query": "q1", "ref_match": "L1", "ref_match_id": "EFO:1"},
            {"query": "q1", "ref_match": "L1", "ref_match_id": "EFO:1"},  # exact dup
            {"query": "q1", "ref_match": "L2", "ref_match_id": "EFO:2"},  # second gold for q1
            {"query": "q2", "ref_match": "L3", "ref_match_id": "EFO:3"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.raw_row_count == 4
    assert audit.unique_mapping_pair_count == 3  # (q1,EFO1) (q1,EFO2) (q2,EFO3)
    assert audit.unique_query_count == 2  # q1, q2


# ─────────────────────────────────────────────────────────────────────────────
# Mapping-pair expansion (Part 4 secondary denominator) never re-queries
# ─────────────────────────────────────────────────────────────────────────────


def test_expand_to_mapping_pairs_preserves_original_raw_row_indices(tmp_path: Path) -> None:
    path = tmp_path / "expand.csv"
    _write_csv(
        path,
        [
            {"query": "q1", "ref_match": "L1", "ref_match_id": "EFO:1"},
            {"query": "q1", "ref_match": "L2", "ref_match_id": "EFO:2"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    pairs = expand_to_mapping_pairs(cqs)
    assert len(pairs) == 2
    assert {(p.gold_code, p.raw_row_index) for p in pairs} == {("EFO:1", 0), ("EFO:2", 1)}


def test_blank_query_rows_excluded_from_canonical_queries(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    _write_csv(
        path,
        [
            {"query": "", "ref_match": "L1", "ref_match_id": "EFO:1"},
            {"query": "q1", "ref_match": "L2", "ref_match_id": "EFO:2"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    assert len(cqs) == 1
    assert cqs[0].source_query == "q1"


def test_audit_reports_missing_counts(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    _write_csv(
        path,
        [
            {"query": "", "ref_match": "L1", "ref_match_id": "EFO:1"},
            {"query": "q2", "ref_match": "L2", "ref_match_id": ""},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.missing_query_count == 1
    assert audit.missing_ref_match_id_count == 1


def test_ref_match_id_prefix_counts(tmp_path: Path) -> None:
    path = tmp_path / "prefix.csv"
    _write_csv(
        path,
        [
            {"query": "q1", "ref_match": "L1", "ref_match_id": "EFO:1"},
            {"query": "q2", "ref_match": "L2", "ref_match_id": "EFO:2"},
            {"query": "q3", "ref_match": "L3", "ref_match_id": "MONDO:1"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.ref_match_id_prefix_counts == {"EFO": 2, "MONDO": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Real file smoke check (skipped if not present -- keeps this suite hermetic)
# ─────────────────────────────────────────────────────────────────────────────


def test_real_ols_efo_file_denominators_if_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "OLS-EFO_full.csv"
    if not dataset_path.exists():
        pytest.skip("OLS-EFO_full.csv not present in this checkout")

    df = load_raw_dataset(dataset_path)
    audit = audit_dataset(df)

    assert audit.raw_row_count == 9998
    assert audit.unique_mapping_pair_count == 7504
    assert audit.unique_query_count == 7377
    assert audit.max_gold_codes_per_query == 3
