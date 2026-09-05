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
    parse_gold_codes,
    parse_gold_labels,
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
# 4b. multi-gold codes encoded in a single ref_match_id cell (UKBB gold-
# parsing-bug fix) -- "CODE:A | CODE:B" must behave like two raw rows for
# scoring purposes, per the confirmed audit finding on UKBB query_id 872.
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_gold_codes_single_code_unchanged() -> None:
    assert parse_gold_codes("EFO:0000001") == ["EFO:0000001"]


def test_parse_gold_codes_splits_compound_cell_and_trims_whitespace() -> None:
    assert parse_gold_codes("EFO:0009679 | EFO:0009684") == ["EFO:0009679", "EFO:0009684"]
    assert parse_gold_codes("EFO:0009679|EFO:0009684") != ["EFO:0009679", "EFO:0009684"]  # no bare "|" delimiter


def test_parse_gold_codes_never_splits_on_colon() -> None:
    codes = parse_gold_codes("EFO:0009679 | EFO:0009684")
    assert all(":" in c for c in codes)
    assert codes == ["EFO:0009679", "EFO:0009684"]


def test_parse_gold_codes_drops_blank_pieces() -> None:
    assert parse_gold_codes("EFO:0000001 |  | EFO:0000002") == ["EFO:0000001", "EFO:0000002"]
    assert parse_gold_codes("   ") == []
    assert parse_gold_codes(None) == []


def test_parse_gold_labels_symmetric_double_pipe_pairs_positionally() -> None:
    assert parse_gold_labels("fatigue||malaise", code_count=2) == ["fatigue", "malaise"]


def test_parse_gold_labels_symmetric_spaced_pipe_pairs_positionally() -> None:
    assert parse_gold_labels(
        "musculoskeletal system disease | connective tissue disease", code_count=2
    ) == ["musculoskeletal system disease", "connective tissue disease"]


def test_parse_gold_labels_asymmetric_count_never_invents_pairing() -> None:
    # UKBB query 872: one label ("paraplegia") for two codes -- must not
    # arbitrarily attribute the single label to either code.
    assert parse_gold_labels("paraplegia", code_count=2) == [None, None]


def test_parse_gold_labels_single_code_no_delimiter_unchanged() -> None:
    assert parse_gold_labels("atrial fibrillation", code_count=1) == ["atrial fibrillation"]


def test_build_canonical_queries_query_872_style_fixture_yields_two_codes_and_missing_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "q872.csv"
    _write_csv(
        path,
        [{"query": "Paraplegia and tetraplegia", "ref_match": "paraplegia", "ref_match_id": "EFO:0009679 | EFO:0009684"}],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    assert len(cqs) == 1
    cq = cqs[0]
    assert cq.gold_codes == ["EFO:0009679", "EFO:0009684"]
    assert cq.gold_labels == [None, None]
    assert cq.gold_count == 2


def test_build_canonical_queries_deduplicates_split_codes_across_rows(tmp_path: Path) -> None:
    """One raw row encoding "CODE:A | CODE:B" plus a second raw row for the
    same query repeating CODE:A must still contribute exactly {CODE:A,
    CODE:B} -- dedup happens at the individual-code level after splitting,
    not at the whole-cell level."""
    path = tmp_path / "dedup.csv"
    _write_csv(
        path,
        [
            {"query": "q", "ref_match": "a||b", "ref_match_id": "EFO:0000001 | EFO:0000002"},
            {"query": "q", "ref_match": "a again", "ref_match_id": "EFO:0000001"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    assert len(cqs) == 1
    assert cqs[0].gold_codes == ["EFO:0000001", "EFO:0000002"]
    assert cqs[0].gold_count == 2


def test_first_gold_rank_recognizes_either_split_gold_code() -> None:
    from llm_ontology_mapper.benchmarking.scenario1_metrics import first_gold_rank

    ranks = (None, "EFO:0009679", "EFO:0009684", "HP:0003470", "HP:0002385")
    gold_codes = ("EFO:0009679", "EFO:0009684")
    assert first_gold_rank(ranks, gold_codes) == 2  # first rank slot containing ANY acceptable gold code


def test_query_872_expected_topk_mrr_recall_after_gold_correction() -> None:
    """The real (already-completed) UKBB targeted rerun for query_id 872
    produced rank_1=blank, rank_2=EFO:0009679, rank_3=EFO:0009684 -- verify
    the canonical scorer, given the CORRECTED two-code gold set, reproduces
    exactly the audit's derived expected metrics."""
    from llm_ontology_mapper.benchmarking.scenario1_metrics import PredictionRecord, score_prediction

    record = PredictionRecord(
        query_id=872,
        query="Paraplegia and tetraplegia",
        gold_codes=("EFO:0009679", "EFO:0009684"),
        status="unmapped",
        ranks=(None, "EFO:0009679", "EFO:0009684", "HP:0003470", "HP:0002385"),
    )
    row_metrics = score_prediction(record)
    assert row_metrics.gold_rank == 2
    assert row_metrics.top1_hit is False
    assert row_metrics.top3_hit is True
    assert row_metrics.top5_hit is True
    assert row_metrics.reciprocal_rank == 0.5
    assert row_metrics.recall_at_gt == 0.5


def test_audit_dataset_counts_multi_code_cells_correctly(tmp_path: Path) -> None:
    path = tmp_path / "multi_code.csv"
    _write_csv(
        path,
        [
            {"query": "q1", "ref_match": "a||b", "ref_match_id": "EFO:0000001 | EFO:0000002"},
            {"query": "q2", "ref_match": "c", "ref_match_id": "EFO:0000003"},
        ],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.unique_mapping_pair_count == 3  # 2 (split) + 1
    assert audit.unique_query_count == 2


def test_audit_dataset_max_gold_codes_per_query_reflects_split_count(tmp_path: Path) -> None:
    path = tmp_path / "max_gold.csv"
    _write_csv(
        path,
        [{"query": "q", "ref_match": "a||b", "ref_match_id": "EFO:0000001 | EFO:0000002"}],
    )
    df = load_raw_dataset(path)
    audit = audit_dataset(df)
    assert audit.max_gold_codes_per_query == 2
    assert audit.gold_count_distribution == {2: 1}


def test_ols_style_single_code_dataset_unaffected_by_gold_parsing_fix(tmp_path: Path) -> None:
    """OLS-EFO_full.csv never encodes multiple codes in one cell -- the fix
    must be a complete no-op for it."""
    path = tmp_path / "ols_style.csv"
    _write_csv(
        path,
        [
            {"query": "progressive supranuclear palsy", "ref_match": "obsolete_supranuclear palsy, progressive", "ref_match_id": "EFO:0002512"},
            {"query": "osteoarthritis, knee", "ref_match": "osteoarthritis, knee", "ref_match_id": "EFO:0004616"},
        ],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    audit = audit_dataset(df)
    assert [cq.gold_codes for cq in cqs] == [["EFO:0002512"], ["EFO:0004616"]]
    assert audit.gold_count_distribution == {1: 2}
    assert audit.max_gold_codes_per_query == 1


def test_biomappings_style_single_code_dataset_unaffected_by_gold_parsing_fix(tmp_path: Path) -> None:
    path = tmp_path / "biomappings_style.csv"
    _write_csv(
        path,
        [{"query": "type 2 diabetes mellitus", "ref_match": "type 2 diabetes mellitus", "ref_match_id": "EFO:0001360"}],
    )
    df = load_raw_dataset(path)
    cqs = build_canonical_queries(df)
    audit = audit_dataset(df)
    assert cqs[0].gold_codes == ["EFO:0001360"]
    assert audit.max_gold_codes_per_query == 1


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
