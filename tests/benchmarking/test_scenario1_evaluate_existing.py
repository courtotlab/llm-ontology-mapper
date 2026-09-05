"""
Unit tests for scripts/run_scenario1_ols_efo.py --evaluate-existing: proves
the graph-distance recomputation path makes zero mapper/LLM/SapBERT calls,
and (when present in this checkout) reevaluates the real GPT-5.6 Luna and
GPT-5-mini 10-row smoke outputs with the real EFO v3.62.0 graph reference
data -- no mapping repeated.

Run with:  pytest tests/benchmarking/test_scenario1_evaluate_existing.py -v -m unit
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
_GRAPH_DATA_DIR = _REPO_DIR / "data" / "text2term_evaluation"
_SMOKE_OUTPUTS_DIR = _REPO_DIR / "outputs" / "evaluation" / "scenario1_ols_efo"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_scenario1_ols_efo_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


def _require_graph_data() -> None:
    if not (_GRAPH_DATA_DIR / "efo_edges.tsv").exists():
        pytest.skip("EFO graph reference data not fetched in this checkout")


def _find_existing_smoke_run(model_substring: str) -> Path | None:
    if not _SMOKE_OUTPUTS_DIR.exists():
        return None
    for run_dir in sorted(_SMOKE_OUTPUTS_DIR.iterdir()):
        config_path = run_dir / "experiment_config.json"
        if not config_path.exists():
            continue
        import json

        config = json.loads(config_path.read_text())
        if model_substring in str(config.get("model", "")):
            return run_dir
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 18. --evaluate-existing performs zero mapper/LLM calls
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_existing_never_touches_mapper_or_provider(tmp_path: Path, script_module, monkeypatch) -> None:
    _require_graph_data()

    def _boom(*args, **kwargs):
        raise AssertionError("--evaluate-existing must never call the mapper/provider/preflight path")

    monkeypatch.setattr(script_module, "build_provider", _boom)
    monkeypatch.setattr(script_module, "build_mapper", _boom)
    monkeypatch.setattr(script_module, "run_preflight", _boom)
    monkeypatch.setattr(script_module, "iter_predictions", _boom)
    monkeypatch.setattr(script_module, "check_sapbert_health", _boom)

    output_dir = _make_tiny_run(tmp_path, script_module)
    exit_code = script_module.main(["--evaluate-existing", str(output_dir)])
    assert exit_code == 0


def _make_tiny_run(tmp_path: Path, script_module) -> Path:
    """Build a minimal on-disk Scenario 1 output directory (dataset +
    experiment_config.json + predictions.csv) without any mapper/LLM call,
    reusing the real dataset/output writers."""
    from llm_ontology_mapper.benchmarking.dataset import file_sha256
    from llm_ontology_mapper.benchmarking.scenario1_dataset import (
        audit_dataset,
        build_canonical_queries,
        load_raw_dataset,
    )
    from llm_ontology_mapper.benchmarking.scenario1_graph_distance import get_graph_index
    from llm_ontology_mapper.benchmarking.scenario1_metrics import (
        PredictionRecord,
        score_prediction,
    )
    from llm_ontology_mapper.benchmarking.scenario1_output import (
        IncrementalPredictionsCsvWriter,
        build_experiment_config,
        row_to_csv_dict,
        write_dataset_validation_json,
        write_experiment_config,
        write_unique_queries_csv,
    )
    from llm_ontology_mapper.benchmarking.scenario1_runner import (
        RankSlot,
        SapBertHealth,
        Scenario1RowResult,
    )

    dataset_path = tmp_path / "mini.csv"
    dataset_path.write_text(
        "query,ref_match,ref_match_id\n"
        "chronic lymphocytic leukemia,chronic lymphocytic leukemia,EFO:0000095\n",
        encoding="utf-8",
    )
    df = load_raw_dataset(dataset_path)
    audit = audit_dataset(df)
    cqs = build_canonical_queries(df)

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
        model="gpt-5.6-luna",
        reasoning_effort="low",
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
    config["rows_completed"] = 1
    write_experiment_config(config, out_dir / "experiment_config.json")
    write_dataset_validation_json(audit, len(cqs), out_dir / "dataset_validation.json")
    write_unique_queries_csv(cqs, out_dir / "unique_queries.csv")

    cq = cqs[0]
    row = Scenario1RowResult(
        query_id=cq.query_id,
        query=cq.source_query,
        gold_codes=cq.gold_codes,
        gold_labels=cq.gold_labels,
        gold_count=1,
        status="mapped",
        mapped_code="EFO:0000095",
        mapped_term="chronic lymphocytic leukemia",
        mapped_ontology="EFO",
        confidence=0.99,
        ranks=[RankSlot("EFO:0000095", "chronic lymphocytic leukemia", "EFO")] + [RankSlot() for _ in range(4)],
    )
    graph_index = get_graph_index(_GRAPH_DATA_DIR)
    rm = score_prediction(
        PredictionRecord(
            query_id=row.query_id, query=row.query, gold_codes=tuple(row.gold_codes), status=row.status, ranks=row.rank_codes
        )
    )
    graph = graph_index.classify(row.rank_codes[0], row.gold_codes)
    with IncrementalPredictionsCsvWriter(out_dir / "predictions.csv") as writer:
        writer.write_row(row_to_csv_dict(row, row_metrics=rm, graph=graph))

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 19/20. existing Luna / GPT-5-mini 10-row smoke predictions can be reevaluated
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_substring", ["luna", "5-mini"])
def test_reevaluate_existing_smoke_run_if_present(model_substring: str, script_module, tmp_path: Path) -> None:
    """Operates on a COPY of the real smoke run directory -- must never
    mutate the checked-in-ish outputs/ directory as a side effect of the
    test suite."""
    import shutil

    _require_graph_data()
    source_dir = _find_existing_smoke_run(model_substring)
    if source_dir is None:
        pytest.skip(f"no existing scenario1 smoke run found for model matching {model_substring!r}")

    run_dir = tmp_path / source_dir.name
    shutil.copytree(source_dir, run_dir)

    exit_code = script_module.main(["--evaluate-existing", str(run_dir)])
    assert exit_code == 0

    with (run_dir / "graph_distance_summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    relationships = {r["relationship"] for r in rows}
    # With the real reference data, "NOT_EVALUATED" must no longer appear.
    assert "NOT_EVALUATED" not in relationships
    assert relationships, "expected at least one graph relationship row"
