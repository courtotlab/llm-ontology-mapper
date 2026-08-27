"""
TP-taxonomy automation (Part 15/16 reliability-audit follow-up): end-to-end
proof that --evaluate-existing recomputes fully-automatic TP-taxonomy
Precision/Recall/F1 -- no manual_review_required.csv input, zero mapper/API
calls -- while Top-k/MRR/Recall@GT and graph-distance outputs are unchanged.

The real EFO graph hierarchy is NOT used here: get_graph_index() is
monkeypatched to a fake index with a fixed per-code relationship map, so the
test is deterministic and independent of the actual EFO tree shape (which is
covered separately by tests/benchmarking/test_scenario1_graph_distance.py --
out of scope here, and untouched by this change).

Run with:  pytest tests/benchmarking/test_scenario1_tp_taxonomy_automation.py -v -m unit
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_scenario1_ols_efo.py"
_REPO_DIR = Path(__file__).resolve().parents[2]


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_scenario1_ols_efo_tp_taxonomy_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


class _FakeGraphDistanceResult:
    def __init__(self, predicted_code, gold_codes, graph_relationship, graph_matched_gold_code):
        self.predicted_code = predicted_code
        self.gold_codes = tuple(gold_codes)
        self.graph_relationship = graph_relationship
        self.graph_matched_gold_code = graph_matched_gold_code
        self.graph_shared_parent_code = None
        self.graph_prediction_found = predicted_code is not None
        self.graph_gold_found = True
        self.note = None


class _FakeGraphIndex:
    """Deterministic stand-in for EfoGraphIndex: classify() returns a fixed
    relationship per predicted_code, from a caller-supplied map. Carries the
    same attributes write_graph_reference_metadata() reads."""

    def __init__(self, relationship_by_code: dict[str, str], data_dir: Path):
        self._relationship_by_code = relationship_by_code
        self.edges_path = data_dir / "fake_efo_edges.tsv"
        self.entailed_path = data_dir / "fake_efo_entailed_edges.tsv"
        self.edges_sha256 = "fake-edges-sha256"
        self.entailed_sha256 = "fake-entailed-sha256"

    def classify(self, predicted_code, gold_codes):
        if predicted_code is None:
            return _FakeGraphDistanceResult(None, gold_codes, "Not Applicable", None)
        relationship = self._relationship_by_code.get(predicted_code, "Unrelated")
        matched = gold_codes[0] if gold_codes and relationship != "Unrelated" else None
        return _FakeGraphDistanceResult(predicted_code, gold_codes, relationship, matched)


# ── Fixture: 7 canonical queries covering every taxonomy branch ────────────
#
#   q0  mapped,   rank1 == gold        -> graph "Same"           -> TP-Identical
#   q1  mapped,   rank1 != gold        -> graph "More Specific"  -> TP-Related
#   q2  mapped,   rank1 != gold        -> graph "More General"   -> TP-Related
#   q3  mapped,   rank1 != gold        -> graph "Sibling"        -> TP-Related
#   q4  mapped,   rank1 != gold        -> graph "Unrelated"      -> FP-Error
#   q5  unmapped                        -> FN
#   q6  execution error                 -> FN (locked zero-credit policy)


def _make_dataset_csv(path: Path) -> None:
    rows = [
        ("query same", "gold term 0", "EFO:0000000"),
        ("query more specific", "gold term 1", "EFO:0000001"),
        ("query more general", "gold term 2", "EFO:0000002"),
        ("query sibling", "gold term 3", "EFO:0000003"),
        ("query unrelated", "gold term 4", "EFO:0000004"),
        ("query unmapped", "gold term 5", "EFO:0000005"),
        ("query error", "gold term 6", "EFO:0000006"),
    ]
    lines = ["query,ref_match,ref_match_id"]
    for query, ref_match, ref_match_id in rows:
        lines.append(f"{query},{ref_match},{ref_match_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run_dir(tmp_path: Path, script_module) -> tuple[Path, dict[str, str]]:
    from llm_ontology_mapper.benchmarking.dataset import file_sha256
    from llm_ontology_mapper.benchmarking.scenario1_dataset import (
        audit_dataset,
        build_canonical_queries,
        load_raw_dataset,
    )
    from llm_ontology_mapper.benchmarking.scenario1_metrics import PredictionRecord, score_prediction
    from llm_ontology_mapper.benchmarking.scenario1_output import (
        IncrementalPredictionsCsvWriter,
        build_experiment_config,
        row_to_csv_dict,
        write_dataset_validation_json,
        write_experiment_config,
        write_unique_queries_csv,
    )
    from llm_ontology_mapper.benchmarking.scenario1_runner import RankSlot, SapBertHealth, Scenario1RowResult

    dataset_path = tmp_path / "mini_ols_efo.csv"
    _make_dataset_csv(dataset_path)
    df = load_raw_dataset(dataset_path)
    audit = audit_dataset(df)
    cqs = build_canonical_queries(df)  # query_id 0..6, in file order
    assert len(cqs) == 7

    out_dir = tmp_path / "run"
    out_dir.mkdir()

    health = SapBertHealth(
        raw_response={"status": "ok"}, status="ok", model="fake", loaded_indexes=["EFO"], available_indexes=["EFO"], lazy_load=True
    )
    config = build_experiment_config(
        source_dataset_path=dataset_path,
        source_dataset_sha256=file_sha256(dataset_path),
        raw_row_count=audit.raw_row_count,
        unique_mapping_pair_count=audit.unique_mapping_pair_count,
        unique_query_count=len(cqs),
        provider="openai",
        model="gpt-4.1-mini",
        reasoning_effort=None,
        temperature=None,
        temperature_mode="provider_default",
        seed=42,
        target_ontology="EFO",
        retrieval_mode="local",
        strict_target_ontology=False,
        max_alternatives=4,
        sapbert_url="http://localhost:8765",
        sapbert_health=health,
        repo_dir=_REPO_DIR,
        start_timestamp="2026-08-26T00:00:00Z",
    )
    config["limit"] = None
    config["completed"] = True
    config["rows_completed"] = 7
    write_experiment_config(config, out_dir / "experiment_config.json")
    write_dataset_validation_json(audit, len(cqs), out_dir / "dataset_validation.json")
    write_unique_queries_csv(cqs, out_dir / "unique_queries.csv")

    def _mapped_row(cq, mapped_code: str) -> Scenario1RowResult:
        return Scenario1RowResult(
            query_id=cq.query_id,
            query=cq.source_query,
            gold_codes=cq.gold_codes,
            gold_labels=cq.gold_labels,
            gold_count=len(cq.gold_codes),
            status="mapped",
            mapped_code=mapped_code,
            mapped_term="term",
            mapped_ontology="EFO",
            confidence=0.9,
            ranks=[RankSlot(mapped_code, "term", "EFO")] + [RankSlot() for _ in range(4)],
        )

    def _unmapped_row(cq) -> Scenario1RowResult:
        return Scenario1RowResult(
            query_id=cq.query_id,
            query=cq.source_query,
            gold_codes=cq.gold_codes,
            gold_labels=cq.gold_labels,
            gold_count=len(cq.gold_codes),
            status="unmapped",
            confidence=0.0,
            ranks=[RankSlot() for _ in range(5)],
        )

    def _error_row(cq) -> Scenario1RowResult:
        return Scenario1RowResult(
            query_id=cq.query_id,
            query=cq.source_query,
            gold_codes=cq.gold_codes,
            gold_labels=cq.gold_labels,
            gold_count=len(cq.gold_codes),
            status="error",
            ranks=[RankSlot() for _ in range(5)],
            error_type="PlannedPipelineError",
            error_stage="local_retrieval",
            error_message="simulated failure",
        )

    relationship_by_code = {
        "EFO:0000000": "Same",
        "MONDO:0000001": "More Specific",
        "MONDO:0000002": "More General",
        "MONDO:0000003": "Sibling",
        "MONDO:0000004": "Unrelated",
    }

    rows = [
        _mapped_row(cqs[0], "EFO:0000000"),  # gold match -> Same -> TP-Identical
        _mapped_row(cqs[1], "MONDO:0000001"),  # -> More Specific -> TP-Related
        _mapped_row(cqs[2], "MONDO:0000002"),  # -> More General -> TP-Related
        _mapped_row(cqs[3], "MONDO:0000003"),  # -> Sibling -> TP-Related
        _mapped_row(cqs[4], "MONDO:0000004"),  # -> Unrelated -> FP-Error
        _unmapped_row(cqs[5]),  # -> FN
        _error_row(cqs[6]),  # -> FN (execution error, zero credit)
    ]

    with IncrementalPredictionsCsvWriter(out_dir / "predictions.csv") as writer:
        for row in rows:
            rm = score_prediction(
                PredictionRecord(
                    query_id=row.query_id, query=row.query, gold_codes=tuple(row.gold_codes),
                    status=row.status, ranks=row.rank_codes,
                )
            )
            writer.write_row(row_to_csv_dict(row, row_metrics=rm, graph=None))

    fake_index = _FakeGraphIndex(relationship_by_code, tmp_path)
    return out_dir, {"relationship_by_code": relationship_by_code}, fake_index


def _patch_no_mapper_calls(monkeypatch, script_module) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("--evaluate-existing must never call the mapper/provider/preflight/SapBERT path")

    monkeypatch.setattr(script_module, "build_provider", _boom)
    monkeypatch.setattr(script_module, "build_mapper", _boom)
    monkeypatch.setattr(script_module, "run_preflight", _boom)
    monkeypatch.setattr(script_module, "iter_predictions", _boom)
    monkeypatch.setattr(script_module, "check_sapbert_health", _boom)


def _read_metric_table(output_dir: Path) -> dict[str, dict[str, str]]:
    with (output_dir / "scenario1_metrics.csv").open(newline="", encoding="utf-8") as fh:
        return {row["metric"]: row for row in csv.DictReader(fh)}


# ─────────────────────────────────────────────────────────────────────────────
# 8/9. numeric Precision/Recall/F1 without manual review, zero mapper/API calls
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_existing_produces_numeric_precision_recall_f1_no_manual_review(
    tmp_path: Path, script_module, monkeypatch
) -> None:
    out_dir, _meta, fake_index = _make_run_dir(tmp_path, script_module)
    _patch_no_mapper_calls(monkeypatch, script_module)
    monkeypatch.setattr(script_module, "get_graph_index", lambda *a, **k: fake_index)

    # No manual_review_required.csv exists yet, and --manual-review-csv is no
    # longer even a CLI flag -- nothing to feed metric computation.
    assert not (out_dir / "manual_review_required.csv").exists()

    exit_code = script_module.main(["--evaluate-existing", str(out_dir)])
    assert exit_code == 0  # 9. real evaluate-existing path, zero mapper/API calls (enforced by _boom patches)

    table = _read_metric_table(out_dir)
    for metric in ("Precision", "Recall", "F1"):
        row = table[metric]
        assert row["status"] == "OK"
        assert row["value"] not in ("", "REQUIRES_MANUAL_REVIEW", None)
        float(row["value"])  # numeric, does not raise

    precision = float(table["Precision"]["value"])
    recall = float(table["Recall"]["value"])
    f1 = float(table["F1"]["value"])
    # TP-Identical=1 (q0), TP-Related=3 (q1,q2,q3), FP-Error=1 (q4), FN=2 (q5,q6)
    assert precision == pytest.approx(4 / 5)
    assert recall == pytest.approx(4 / 6)
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert f1 == pytest.approx(expected_f1)

    md_text = (out_dir / "scenario1_metrics.md").read_text(encoding="utf-8")
    assert "REQUIRES_MANUAL_REVIEW" not in md_text
    assert "fully automatic" in md_text.lower()


def test_execution_error_kept_separately_reported_and_zero_credit(tmp_path: Path, script_module, monkeypatch) -> None:
    """Execution errors are (a) reported separately via execution_diagnostics.csv
    with their own error_count, and (b) give zero TP-taxonomy credit (FN) --
    never silently dropped, never TP."""
    out_dir, _meta, fake_index = _make_run_dir(tmp_path, script_module)
    _patch_no_mapper_calls(monkeypatch, script_module)
    monkeypatch.setattr(script_module, "get_graph_index", lambda *a, **k: fake_index)

    script_module.main(["--evaluate-existing", str(out_dir)])

    with (out_dir / "execution_diagnostics.csv").open(newline="", encoding="utf-8") as fh:
        diag = next(csv.DictReader(fh))
    assert diag["total"] == "7"
    assert diag["mapped_count"] == "5"
    assert diag["unmapped_count"] == "1"
    assert diag["error_count"] == "1"


# ─────────────────────────────────────────────────────────────────────────────
# 7. manual_review_required.csv, if present, does not gate the metric
# ─────────────────────────────────────────────────────────────────────────────


def test_manual_review_csv_presence_or_absence_does_not_change_metrics(
    tmp_path: Path, script_module, monkeypatch
) -> None:
    out_dir, _meta, fake_index = _make_run_dir(tmp_path, script_module)
    _patch_no_mapper_calls(monkeypatch, script_module)
    monkeypatch.setattr(script_module, "get_graph_index", lambda *a, **k: fake_index)

    script_module.main(["--evaluate-existing", str(out_dir)])
    table_before = _read_metric_table(out_dir)

    # manual_review_required.csv was generated (diagnostic) -- confirm it
    # lists the graph-related rows, but re-running recompute again (as if a
    # human had "completed" it, without this pipeline ever reading it back)
    # produces bit-identical Precision/Recall/F1.
    assert (out_dir / "manual_review_required.csv").exists()
    with (out_dir / "manual_review_required.csv").open(newline="", encoding="utf-8") as fh:
        review_rows = list(csv.DictReader(fh))
    assert {r["graph_relationship"] for r in review_rows} == {"More Specific", "More General", "Sibling"}

    script_module.main(["--evaluate-existing", str(out_dir)])
    table_after = _read_metric_table(out_dir)

    for metric in ("Precision", "Recall", "F1"):
        assert table_before[metric]["value"] == table_after[metric]["value"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Top-k/MRR/Recall@GT and graph-distance outputs unchanged by this update
# ─────────────────────────────────────────────────────────────────────────────


def test_topk_mrr_recall_at_gt_and_graph_distance_unaffected(tmp_path: Path, script_module, monkeypatch) -> None:
    out_dir, _meta, fake_index = _make_run_dir(tmp_path, script_module)
    _patch_no_mapper_calls(monkeypatch, script_module)
    monkeypatch.setattr(script_module, "get_graph_index", lambda *a, **k: fake_index)

    script_module.main(["--evaluate-existing", str(out_dir)])
    table = _read_metric_table(out_dir)

    # Only q0 has its gold code at rank 1 (exact match); q1-q4 are mapped to
    # a DIFFERENT code than gold (graph-related or unrelated, never exact);
    # q5/q6 have no rank1 at all. So Top-1 = 1/7, MRR = 1/7.
    assert float(table["Top-1"]["value"]) == pytest.approx(1 / 7)
    assert table["Top-1"]["status"] == "OK"
    assert float(table["MRR"]["value"]) == pytest.approx(1 / 7)
    assert table["Recall@GT"]["status"] == "OK"

    # Graph-distance percentages: 1 Same, 1 More Specific, 1 More General,
    # 1 Sibling, 1 Unrelated, 2 "Not Applicable" (unmapped/error) out of 7.
    with (out_dir / "graph_distance_summary.csv").open(newline="", encoding="utf-8") as fh:
        dist = {row["relationship"]: int(row["count"]) for row in csv.DictReader(fh)}
    assert dist["Same"] == 1
    assert dist["More Specific"] == 1
    assert dist["More General"] == 1
    assert dist["Sibling"] == 1
    assert dist["Unrelated"] == 1
    assert dist["Not Applicable"] == 2

    with (out_dir / "graph_distance_rows.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7
