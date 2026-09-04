"""Scenario 1 strict common-query-alignment tests
(text2term_alignment.py + the aligned-comparison additions to
figures/graph_relationship_comparison.py + scripts/fetch_text2term_evaluation_outputs.py).

Filesystem-only (tmp_path) synthetic fixtures for the alignment mechanics --
no network, no mapper, no LLM calls. `scenario1_graph_distance.classify()` is
called for real (against this repo's already-vendored EFO edge tables) since
that reuse is exactly the behavior under test; fixture codes are chosen from
already-verified real classifications so results are deterministic.

The real completed Scenario 1 run directories and the real vendored
text2term-evaluation output files are never modified by this suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking import scenario1_graph_distance as graph_distance
from llm_ontology_mapper.benchmarking import text2term_alignment as align

pytestmark = pytest.mark.unit

# Verified real EFO relationships (from this repo's own vendored graph +
# spot-checked against real Scenario 1 predictions.csv rows):
#   predicted == gold                              -> Same (trivial)
#   predicted "MONDO:0004975" vs gold "EFO:0000249" -> Unrelated

SAME_CODE = "EFO:0000465"
UNRELATED_PREDICTED = "MONDO:0004975"
UNRELATED_GOLD = "EFO:0000249"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 11-12. Normalization
# ─────────────────────────────────────────────────────────────────────────────


def test_whitespace_normalization() -> None:
    assert align.normalize_source_text("  Alzheimer's disease  ") == "Alzheimer's disease"
    assert align.normalize_source_text("term\n") == "term"


def test_curie_normalization_ordo_alias() -> None:
    assert align.normalize_gold_curie("Orphanet:2130") == "ORDO:2130"
    assert align.normalize_gold_curie("ORDO:2130") == "ORDO:2130"
    assert align.normalize_gold_curie("orphanet:2130") == "ORDO:2130"


def test_curie_normalization_preserves_local_code_case() -> None:
    # Local codes are never case-folded -- only the prefix is uppercased.
    assert align.normalize_gold_curie("EFO:AbC123") == "EFO:AbC123"


def test_curie_normalization_passthrough_for_non_curie() -> None:
    assert align.normalize_gold_curie("not-a-curie") == "not-a-curie"


def test_curie_normalization_unaffected_prefix_uppercased() -> None:
    assert align.normalize_gold_curie("efo:0000249") == "EFO:0000249"


# ─────────────────────────────────────────────────────────────────────────────
# 13. No fuzzy matching anywhere in this module
# ─────────────────────────────────────────────────────────────────────────────


def test_module_never_fuzzy_matches() -> None:
    source = Path(align.__file__).read_text(encoding="utf-8")
    forbidden = ["import difflib", "import rapidfuzz", "import fuzzywuzzy", "Levenshtein.",
                 "difflib.SequenceMatcher", "get_close_matches(", "process.extract("]
    for token in forbidden:
        assert token not in source


# ─────────────────────────────────────────────────────────────────────────────
# 5-7. Parsing t2t results schemas
# ─────────────────────────────────────────────────────────────────────────────


def _t2t_fields() -> list[str]:
    return list(align.T2T_RESULTS_REQUIRED_FIELDS)


def test_results_tsv_parsed_correctly(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    _write_tsv(path, _t2t_fields(), [
        {"Source Term ID": "id1", "Source Term": "term1", "t2t.Mapping": "EFO:0000465",
         "t2t.MappingLabel": "x", "Benchmark.Mapping": "EFO:0000465", "Benchmark.MappingLabel": "x",
         "Classification": "Same"},
    ])
    rows = align.load_t2t_results(path)
    assert len(rows) == 1
    assert rows[0].source_term == "term1"
    assert rows[0].t2t_mapping == "EFO:0000465"
    assert rows[0].classification == "Same"


def test_results_tsv_missing_column_hard_fails(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    fields = [f for f in _t2t_fields() if f != "Classification"]
    _write_tsv(path, fields, [{f: "x" for f in fields}])
    with pytest.raises(align.SchemaError):
        align.load_t2t_results(path)


def test_results_tsv_empty_t2t_mapping_becomes_none(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    _write_tsv(path, _t2t_fields(), [
        {"Source Term ID": "id1", "Source Term": "term1", "t2t.Mapping": "",
         "t2t.MappingLabel": "", "Benchmark.Mapping": "EFO:0000465", "Benchmark.MappingLabel": "x",
         "Classification": "Unrelated"},
    ])
    rows = align.load_t2t_results(path)
    assert rows[0].t2t_mapping is None


def test_missing_vendored_file_hard_fails(tmp_path: Path) -> None:
    with pytest.raises(align.AlignmentError):
        align.load_t2t_results(tmp_path / "does_not_exist.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: raw aggregate recomputation + reproducibility check
# ─────────────────────────────────────────────────────────────────────────────


def test_recompute_aggregate_tallies_classification_column(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    _write_tsv(path, _t2t_fields(), [
        {"Source Term ID": "1", "Source Term": "a", "t2t.Mapping": "X", "t2t.MappingLabel": "", "Benchmark.Mapping": "Y", "Benchmark.MappingLabel": "", "Classification": "Same"},
        {"Source Term ID": "2", "Source Term": "b", "t2t.Mapping": "X", "t2t.MappingLabel": "", "Benchmark.Mapping": "Y", "Benchmark.MappingLabel": "", "Classification": "Same"},
        {"Source Term ID": "3", "Source Term": "c", "t2t.Mapping": "X", "t2t.MappingLabel": "", "Benchmark.Mapping": "Y", "Benchmark.MappingLabel": "", "Classification": "Unrelated"},
    ])
    rows = align.load_t2t_results(path)
    counts = align.recompute_aggregate_from_raw(rows)
    assert counts["Same"] == 2
    assert counts["Unrelated"] == 1
    assert counts["Sibling"] == 0


def test_recompute_aggregate_rejects_unknown_classification(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    _write_tsv(path, _t2t_fields(), [
        {"Source Term ID": "1", "Source Term": "a", "t2t.Mapping": "X", "t2t.MappingLabel": "", "Benchmark.Mapping": "Y", "Benchmark.MappingLabel": "", "Classification": "TotallyMade Up"},
    ])
    rows = align.load_t2t_results(path)
    with pytest.raises(align.SchemaError):
        align.recompute_aggregate_from_raw(rows)


def test_reproducibility_check_reports_agreement_and_mismatch() -> None:
    recomputed = dict.fromkeys(graph_distance.ALL_RELATIONSHIPS, 0)
    recomputed["Same"] = 10
    published_agree = dict(recomputed)
    result = align.check_table1_reproducibility("X", recomputed, published_agree)
    assert result.agreement is True

    published_disagree = dict(recomputed)
    published_disagree["Same"] = 8
    published_disagree["More Specific"] = 2
    result2 = align.check_table1_reproducibility("X", recomputed, published_disagree)
    assert result2.agreement is False
    assert result2.mismatches["Same"] == (10, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Our-side loading + OLS multi-gold restriction (16-18)
# ─────────────────────────────────────────────────────────────────────────────


def _make_run_dir(
    tmp_path: Path, name: str, *, unique_queries: list[dict], expanded: list[dict], predictions_graph: dict[str, str],
) -> Path:
    out = tmp_path / name
    out.mkdir()
    _write_csv(
        out / "unique_queries.csv",
        ["query_id", "source_query", "gold_codes", "gold_labels", "original_row_indices", "original_mapping_pair_count"],
        unique_queries,
    )
    _write_csv(
        out / "mapping_pair_expanded_predictions.csv",
        ["query_id", "source_query", "gold_code", "gold_label", "raw_row_index", "rank_1_code", "rank_2_code",
         "rank_3_code", "rank_4_code", "rank_5_code", "first_gold_rank", "top1_hit", "top3_hit", "top5_hit",
         "reciprocal_rank", "status"],
        expanded,
    )
    pred_rows = [{"query_id": qid, "graph_relationship": rel} for qid, rel in predictions_graph.items()]
    _write_csv(out / "predictions.csv", ["query_id", "graph_relationship"], pred_rows)
    return out


def _blank_expanded(query_id: str, source_query: str, gold_code: str, rank_1_code: str | None, status: str) -> dict:
    return {
        "query_id": query_id, "source_query": source_query, "gold_code": gold_code, "gold_label": "",
        "raw_row_index": query_id, "rank_1_code": rank_1_code or "", "rank_2_code": "", "rank_3_code": "",
        "rank_4_code": "", "rank_5_code": "", "first_gold_rank": "", "top1_hit": "False", "top3_hit": "False",
        "top5_hit": "False", "reciprocal_rank": "0.0", "status": status,
    }


def test_single_gold_query_ids_use_mapping_pair_count_not_pipe_count(tmp_path: Path) -> None:
    # A query whose gold_codes literally contains " | " but whose canonical
    # original_mapping_pair_count is "1" must be treated as single-gold.
    run_dir = _make_run_dir(
        tmp_path, "quirky",
        unique_queries=[
            {"query_id": "0", "source_query": "term0", "gold_codes": "HP:0001 | EFO:0002", "gold_labels": "",
             "original_row_indices": "0", "original_mapping_pair_count": "1"},
            {"query_id": "1", "source_query": "term1", "gold_codes": "EFO:0003|EFO:0004", "gold_labels": "",
             "original_row_indices": "1", "original_mapping_pair_count": "2"},
        ],
        expanded=[], predictions_graph={},
    )
    single_gold = align.load_single_gold_query_ids(run_dir)
    assert single_gold == {"0"}


def test_ols_style_multi_gold_queries_excluded_from_primary_subset(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "ols_like",
        unique_queries=[
            {"query_id": "0", "source_query": "single", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "0", "original_mapping_pair_count": "1"},
            {"query_id": "1", "source_query": "multi", "gold_codes": f"{SAME_CODE}|EFO:9999999", "gold_labels": "",
             "original_row_indices": "1", "original_mapping_pair_count": "2"},
        ],
        expanded=[
            _blank_expanded("0", "single", SAME_CODE, SAME_CODE, "mapped"),
            _blank_expanded("1", "multi", SAME_CODE, SAME_CODE, "mapped"),
        ],
        predictions_graph={"0": "Same", "1": "Same"},
    )
    rows = align.load_our_mapping_pairs(run_dir)
    single_gold_rows = [r for r in rows if r.is_single_gold]
    assert len(single_gold_rows) == 1
    assert single_gold_rows[0].query_id == "0"


def test_single_gold_ols_rows_eligible(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "ols_eligible",
        unique_queries=[{"query_id": "0", "source_query": "t", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "t", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    rows = align.load_our_mapping_pairs(run_dir)
    assert rows[0].is_single_gold is True


# ─────────────────────────────────────────────────────────────────────────────
# Alignment mechanics: exact match / ambiguity / duplicate collapse (8-10)
# ─────────────────────────────────────────────────────────────────────────────


def _t2t_row_dict(source_term: str, gold: str, t2t_mapping: str, classification: str, sid: str = "id") -> dict:
    return {"Source Term ID": sid, "Source Term": source_term, "t2t.Mapping": t2t_mapping, "t2t.MappingLabel": "",
            "Benchmark.Mapping": gold, "Benchmark.MappingLabel": "", "Classification": classification}


def _make_published_counts(**overrides: int) -> dict[str, int]:
    counts = dict.fromkeys(graph_distance.ALL_RELATIONSHIPS, 0)
    counts.update(overrides)
    return counts


def test_exact_source_and_gold_match_succeeds(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.strict_matched_n == 1
    assert result.aligned_rows[0].alignment_status == "matched"


def test_source_match_with_different_gold_reports_gold_mismatch(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", "EFO:9999999", "EFO:9999999", "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.strict_matched_n == 0
    assert result.gold_mismatch_n == 1
    assert any(u.reason.startswith("gold mismatch") for u in result.unmatched if u.side == "ours")


def test_gold_only_match_with_different_source_is_insufficient(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term A", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term A", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term B (different)", SAME_CODE, SAME_CODE, "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.strict_matched_n == 0
    assert any(u.reason == "no exact identity match" for u in result.unmatched if u.side == "ours")


def test_identical_duplicate_t2t_rows_safely_collapsed(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [
        _t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same", sid="id1"),
        _t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same", sid="id2"),  # identical duplicate, different upstream ID
    ])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=2))
    assert result.strict_matched_n == 1
    assert result.ambiguous_n == 0


def test_conflicting_duplicate_t2t_rows_rejected_as_ambiguous(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [
        _t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same", sid="id1"),
        _t2t_row_dict("term", SAME_CODE, UNRELATED_PREDICTED, "Unrelated", sid="id2"),  # conflicting prediction
    ])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1, Unrelated=1))
    assert result.strict_matched_n == 0
    assert result.ambiguous_n == 1
    assert any(u.reason == "ambiguous duplicate on t2t side" for u in result.unmatched if u.side == "ours")


# ─────────────────────────────────────────────────────────────────────────────
# Graph reuse (19-23): same EfoGraphIndex/classify(), same priority
# ─────────────────────────────────────────────────────────────────────────────


def test_reclassification_reuses_scenario1_graph_distance_classify(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": UNRELATED_GOLD, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", UNRELATED_GOLD, UNRELATED_PREDICTED, "mapped")],
        predictions_graph={"0": "Unrelated"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", UNRELATED_GOLD, UNRELATED_PREDICTED, "Unrelated")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Unrelated=1))
    row = result.aligned_rows[0]
    expected = graph_distance.classify(UNRELATED_PREDICTED, [UNRELATED_GOLD]).graph_relationship
    assert row.ours_recomputed_relationship == expected == "Unrelated"
    assert row.t2t_recomputed_relationship == expected


def test_priority_order_matches_scenario1_graph_distance() -> None:
    assert graph_distance.ALL_RELATIONSHIPS == ("Same", "More Specific", "More General", "Sibling", "Unrelated")


def test_our_and_t2t_reclassification_use_same_gold_representation(tmp_path: Path) -> None:
    # Orphanet: vs already-ORDO: gold on the two sides -- reclassification
    # must canonicalize consistently so both are scored against the SAME node.
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": "Orphanet:2130", "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", "Orphanet:2130", "ORDO:2130", "mapped")],
        predictions_graph={"0": "Unrelated"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", "ORDO:2130", "ORDO:2130", "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.strict_matched_n == 1
    # ours predicted ORDO:2130 == gold ORDO:2130 -> Same, regardless of the
    # unaliased "Orphanet:" spelling in our own persisted gold_code.
    assert result.aligned_rows[0].ours_recomputed_relationship == "Same"


def test_stored_vs_recomputed_reclassification_audit(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[
            {"query_id": "0", "source_query": "t0", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "0", "original_mapping_pair_count": "1"},
            {"query_id": "1", "source_query": "t1", "gold_codes": UNRELATED_GOLD, "gold_labels": "",
             "original_row_indices": "1", "original_mapping_pair_count": "1"},
        ],
        expanded=[
            _blank_expanded("0", "t0", SAME_CODE, SAME_CODE, "mapped"),
            _blank_expanded("1", "t1", UNRELATED_GOLD, UNRELATED_PREDICTED, "mapped"),
        ],
        predictions_graph={"0": "Same", "1": "Unrelated"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [
        _t2t_row_dict("t0", SAME_CODE, SAME_CODE, "Same"),
        # deliberately WRONG stored classification to prove disagreement is detected
        _t2t_row_dict("t1", UNRELATED_GOLD, UNRELATED_PREDICTED, "Sibling"),
    ])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1, Sibling=1))
    agreement = sum(1 for r in result.aligned_rows if r.t2t_original_published_classification == r.t2t_recomputed_relationship)
    assert agreement == 1
    assert len(result.aligned_rows) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Outcome categories (24-28)
# ─────────────────────────────────────────────────────────────────────────────


def test_unmapped_ours_becomes_no_top1_prediction(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, None, "unmapped")],
        predictions_graph={"0": "Not Applicable"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.aligned_rows[0].ours_recomputed_relationship == align.NO_TOP1_CATEGORY


def test_error_ours_becomes_no_top1_prediction(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, None, "error")],
        predictions_graph={"0": "Not Applicable"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    assert result.aligned_rows[0].ours_recomputed_relationship == align.NO_TOP1_CATEGORY


def test_t2t_missing_prediction_becomes_no_top1_prediction(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", SAME_CODE, "", "Unrelated")])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Unrelated=1))
    assert result.aligned_rows[0].t2t_recomputed_relationship == align.NO_TOP1_CATEGORY


def test_six_outcome_categories_mutually_exclusive_and_sum_to_aligned_n(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[
            {"query_id": "0", "source_query": "t0", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "0", "original_mapping_pair_count": "1"},
            {"query_id": "1", "source_query": "t1", "gold_codes": UNRELATED_GOLD, "gold_labels": "",
             "original_row_indices": "1", "original_mapping_pair_count": "1"},
            {"query_id": "2", "source_query": "t2", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "2", "original_mapping_pair_count": "1"},
        ],
        expanded=[
            _blank_expanded("0", "t0", SAME_CODE, SAME_CODE, "mapped"),
            _blank_expanded("1", "t1", UNRELATED_GOLD, UNRELATED_PREDICTED, "mapped"),
            _blank_expanded("2", "t2", SAME_CODE, None, "unmapped"),
        ],
        predictions_graph={"0": "Same", "1": "Unrelated", "2": "Not Applicable"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [
        _t2t_row_dict("t0", SAME_CODE, SAME_CODE, "Same"),
        _t2t_row_dict("t1", UNRELATED_GOLD, UNRELATED_PREDICTED, "Unrelated"),
        _t2t_row_dict("t2", SAME_CODE, SAME_CODE, "Same"),
    ])
    result = align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=2, Unrelated=1))
    counts = align.outcome_counts(result.aligned_rows, field="ours_recomputed_relationship")
    assert sum(counts.values()) == len(result.aligned_rows) == 3
    assert set(counts) == set(align.OUTCOME_CATEGORIES)


# ─────────────────────────────────────────────────────────────────────────────
# Pairing (29-32)
# ─────────────────────────────────────────────────────────────────────────────


def _three_row_alignment(tmp_path: Path) -> align.AlignmentResult:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[
            {"query_id": "0", "source_query": "t0", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "0", "original_mapping_pair_count": "1"},
            {"query_id": "1", "source_query": "t1", "gold_codes": SAME_CODE, "gold_labels": "",
             "original_row_indices": "1", "original_mapping_pair_count": "1"},
            {"query_id": "2", "source_query": "t2", "gold_codes": UNRELATED_GOLD, "gold_labels": "",
             "original_row_indices": "2", "original_mapping_pair_count": "1"},
        ],
        expanded=[
            _blank_expanded("0", "t0", SAME_CODE, SAME_CODE, "mapped"),          # ours exact, t2t exact
            _blank_expanded("1", "t1", SAME_CODE, SAME_CODE, "mapped"),          # ours exact, t2t NOT exact
            _blank_expanded("2", "t2", UNRELATED_GOLD, UNRELATED_PREDICTED, "mapped"),  # neither exact
        ],
        predictions_graph={"0": "Same", "1": "Same", "2": "Unrelated"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [
        _t2t_row_dict("t0", SAME_CODE, SAME_CODE, "Same"),
        _t2t_row_dict("t1", SAME_CODE, UNRELATED_PREDICTED, "Unrelated"),
        _t2t_row_dict("t2", UNRELATED_GOLD, UNRELATED_PREDICTED, "Unrelated"),
    ])
    return align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1, Unrelated=2))


def test_both_methods_have_exactly_same_aligned_row_ids(tmp_path: Path) -> None:
    result = _three_row_alignment(tmp_path)
    # every aligned row carries one shared alignment_id used by both sides
    assert len({r.alignment_id for r in result.aligned_rows}) == len(result.aligned_rows) == 3


def test_pairwise_transitions_sum_to_aligned_n(tmp_path: Path) -> None:
    result = _three_row_alignment(tmp_path)
    transitions = align.compute_paired_transitions(result.aligned_rows)
    assert transitions.total == len(result.aligned_rows) == 3


def test_ours_only_and_t2t_only_correctness_calculated_correctly(tmp_path: Path) -> None:
    result = _three_row_alignment(tmp_path)
    transitions = align.compute_paired_transitions(result.aligned_rows)
    assert transitions.both_exact == 1
    assert transitions.ours_only_exact == 1
    assert transitions.t2t_only_exact == 0
    assert transitions.neither_exact == 1


def test_mcnemar_fixture() -> None:
    transitions = align.PairedTransitions(both_exact=10, ours_only_exact=8, t2t_only_exact=2, neither_exact=5)
    result = align.compute_mcnemar("X", transitions)
    assert result.ours_only_correct == 8
    assert result.t2t_only_correct == 2
    assert result.discordant_n == 10
    assert 0.0 <= result.p_value <= 1.0


def test_mcnemar_no_discordant_pairs_gives_p_one() -> None:
    transitions = align.PairedTransitions(both_exact=10, ours_only_exact=0, t2t_only_exact=0, neither_exact=5)
    result = align.compute_mcnemar("X", transitions)
    assert result.p_value == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Alignment quality threshold
# ─────────────────────────────────────────────────────────────────────────────


def test_alignment_quality_thresholds() -> None:
    assert align.alignment_quality_label(0.97) == "STRONG"
    assert align.alignment_quality_label(0.95) == "STRONG"
    assert align.alignment_quality_label(0.92) == "GOOD"
    assert align.alignment_quality_label(0.80) == "PARTIAL"
    assert align.alignment_quality_label(0.50) == "INSUFFICIENT"


# ─────────────────────────────────────────────────────────────────────────────
# Files (33-40)
# ─────────────────────────────────────────────────────────────────────────────


def test_alignment_summary_and_derived_files_written(tmp_path: Path) -> None:
    result = _three_row_alignment(tmp_path)
    results = {"UKBB-EFO": result,
               "Biomappings-EFO": result,  # reuse fixture for the other two benchmarks -- writer-shape test only
               "OLS-EFO (full)": result}

    summary_csv = tmp_path / "summary.csv"
    align.write_alignment_summary_csv(results, summary_csv)
    assert summary_csv.exists()
    with summary_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["strict_matched_n"] == "3"

    rows_csv = tmp_path / "rows.csv"
    align.write_aligned_rows_csv(results, rows_csv)
    assert rows_csv.exists()
    with rows_csv.open(newline="", encoding="utf-8") as fh:
        aligned = list(csv.DictReader(fh))
    assert len(aligned) == 9  # 3 rows x 3 "benchmarks" (fixture reused)

    unmatched_csv = tmp_path / "unmatched.csv"
    align.write_unmatched_csv(results, unmatched_csv)
    assert unmatched_csv.exists()

    reclass_csv = tmp_path / "reclass.csv"
    align.write_reclassification_audit_csv(results, reclass_csv)
    with reclass_csv.open(newline="", encoding="utf-8") as fh:
        reclass_rows = list(csv.DictReader(fh))
    assert len(reclass_rows) == 3

    repro_csv = tmp_path / "repro.csv"
    align.write_table1_reproducibility_csv(results, repro_csv)
    assert repro_csv.exists()

    transitions = {b: align.compute_paired_transitions(results[b].aligned_rows) for b in align.BENCHMARK_ORDER}
    transitions_csv = tmp_path / "transitions.csv"
    align.write_transitions_csv(transitions, transitions_csv)
    with transitions_csv.open(newline="", encoding="utf-8") as fh:
        t_rows = list(csv.DictReader(fh))
    assert len(t_rows) == 3
    assert int(t_rows[0]["aligned_n"]) == 3

    mcnemar = {b: align.compute_mcnemar(b, transitions[b]) for b in align.BENCHMARK_ORDER}
    mcnemar_csv = tmp_path / "mcnemar.csv"
    align.write_mcnemar_csv(mcnemar, mcnemar_csv)
    assert mcnemar_csv.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Fetch script / provenance (1-4)
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_script_uses_pinned_commit_from_graph_distance() -> None:
    import scripts.fetch_text2term_evaluation_outputs as fetch_script  # type: ignore[import-not-found]

    assert fetch_script.PINNED_COMMIT == graph_distance.PINNED_COMMIT
    assert fetch_script.SOURCE_REPOSITORY == graph_distance.SOURCE_REPOSITORY


def test_fetch_script_required_files_have_64_char_sha256() -> None:
    import scripts.fetch_text2term_evaluation_outputs as fetch_script  # type: ignore[import-not-found]

    assert len(fetch_script.REQUIRED_FILES) == 9
    for _upstream, (_local, sha256) in fetch_script.REQUIRED_FILES.items():
        assert len(sha256) == 64
        int(sha256, 16)  # must be valid hex


def test_fetch_script_writes_provenance_manifest(tmp_path: Path) -> None:
    import scripts.fetch_text2term_evaluation_outputs as fetch_script  # type: ignore[import-not-found]

    entries = [{"upstream_path": "output/x.tsv", "local_path": str(tmp_path / "x.tsv"), "sha256": "abc", "status": "fetched"}]
    manifest_path = fetch_script.write_provenance(entries, tmp_path)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pinned_commit"] == graph_distance.PINNED_COMMIT
    assert manifest["repository_url"] == graph_distance.SOURCE_REPOSITORY
    assert manifest["files"] == entries


def test_fetch_script_rejects_sha256_mismatch(tmp_path: Path) -> None:
    import scripts.fetch_text2term_evaluation_outputs as fetch_script  # type: ignore[import-not-found]

    bad_file = tmp_path / "bad.tsv"
    bad_file.write_text("not the expected content", encoding="utf-8")
    with pytest.raises(fetch_script.FetchError):
        fetch_script.fetch_one("output/bad.tsv", "bad.tsv", "0" * 64, tmp_path, force=False)


# ─────────────────────────────────────────────────────────────────────────────
# Safety (41-43)
# ─────────────────────────────────────────────────────────────────────────────


def test_alignment_module_has_no_network_or_mapper_imports() -> None:
    source = Path(align.__file__).read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper(", "PlannedPipeline", "OntologyValidator",
                 "import openai", "import requests", "import httpx", "urllib.request", "SapBert"]
    for token in forbidden:
        assert token not in source


def test_fetch_script_is_the_only_module_using_urllib() -> None:
    align_source = Path(align.__file__).read_text(encoding="utf-8")
    graph_figs_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "llm_ontology_mapper" / "benchmarking" / "figures" / "graph_relationship_comparison.py"
    )
    for source in (align_source, graph_figs_path.read_text(encoding="utf-8")):
        assert "urllib" not in source
        assert "requests." not in source


def test_source_run_directories_never_modified_by_alignment(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, "run",
        unique_queries=[{"query_id": "0", "source_query": "term", "gold_codes": SAME_CODE, "gold_labels": "",
                          "original_row_indices": "0", "original_mapping_pair_count": "1"}],
        expanded=[_blank_expanded("0", "term", SAME_CODE, SAME_CODE, "mapped")],
        predictions_graph={"0": "Same"},
    )
    t2t_path = tmp_path / "results.tsv"
    _write_tsv(t2t_path, _t2t_fields(), [_t2t_row_dict("term", SAME_CODE, SAME_CODE, "Same")])

    before = {f.name: f.read_bytes() for f in run_dir.iterdir()}
    align.align_benchmark("UKBB-EFO", run_dir, t2t_path, _make_published_counts(Same=1))
    after = {f.name: f.read_bytes() for f in run_dir.iterdir()}
    assert before == after
