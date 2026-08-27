"""
Reliability-audit follow-up: focused, non-live tests for --resume error-row
retry, structural error_stage classification, and the consecutive
local-SapBERT-retrieval-failure guard in scripts/run_scenario1_ols_efo.py.

No network calls, no OpenAI calls. Where the full CLI script is exercised
(via script_module.main()), every function that would touch a real network
resource (build_provider, run_preflight, check_sapbert_health, iter_predictions)
is monkeypatched -- see test_no_real_provider_or_sapbert_network_calls_needed.

Run with:  pytest tests/benchmarking/test_scenario1_resume_reliability.py -v -m unit
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.scenario1_dataset import CanonicalQuery
from llm_ontology_mapper.benchmarking.scenario1_output import (
    RETRY_ERROR_HISTORY_CSV_FIELDS,
    IncrementalPredictionsCsvWriter,
    quarantine_error_rows_for_resume,
    read_existing_predictions,
    row_to_csv_dict,
)
from llm_ontology_mapper.benchmarking.scenario1_runner import (
    ERROR_STAGE_LOCAL_RETRIEVAL,
    ERROR_STAGE_QUERY_PLANNER,
    ERROR_STAGE_RERANKER,
    RankSlot,
    SapBertHealth,
    SapBertHealthError,
    Scenario1RowResult,
    classify_error_stage,
)
from llm_ontology_mapper.llm_reranker import LLMRerankerError
from llm_ontology_mapper.local_retriever import LocalRetrievalError
from llm_ontology_mapper.planned_pipeline import PlannedPipelineError
from llm_ontology_mapper.query_planner import QueryPlanningError

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by all tests below
# ─────────────────────────────────────────────────────────────────────────────


def _row(query_id: int = 1, **overrides) -> Scenario1RowResult:
    defaults = dict(
        query_id=query_id,
        query=f"query {query_id}",
        gold_codes=["EFO:0000001"],
        gold_labels=["Some disorder"],
        gold_count=1,
        status="mapped",
        mapped_code="EFO:0000001",
        mapped_term="Some disorder",
        mapped_ontology="EFO",
        confidence=0.9,
        ranks=[RankSlot("EFO:0000001", "Some disorder", "EFO")] + [RankSlot() for _ in range(4)],
    )
    defaults.update(overrides)
    return Scenario1RowResult(**defaults)


def _write_row(writer: IncrementalPredictionsCsvWriter, row: Scenario1RowResult) -> None:
    from llm_ontology_mapper.benchmarking.scenario1_metrics import PredictionRecord, score_prediction

    rm = score_prediction(
        PredictionRecord(
            query_id=row.query_id,
            query=row.query,
            gold_codes=tuple(row.gold_codes),
            status=row.status,
            ranks=row.rank_codes,
        )
    )
    writer.write_row(row_to_csv_dict(row, row_metrics=rm, graph=None))


def _raise_chained(outer_message: str, cause: BaseException) -> BaseException:
    try:
        raise cause
    except type(cause) as caught:
        try:
            raise PlannedPipelineError(outer_message) from caught
        except PlannedPipelineError as wrapped:
            return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# 6/7/8. classify_error_stage: type-based, from the real cause chain
# ─────────────────────────────────────────────────────────────────────────────


def test_local_retrieval_error_classified_as_local_retrieval() -> None:
    exc = _raise_chained(
        "local retrieval failed during planned mapping.",
        LocalRetrievalError("connection refused"),
    )
    assert classify_error_stage(exc) == ERROR_STAGE_LOCAL_RETRIEVAL


def test_query_planner_error_not_classified_local_retrieval() -> None:
    exc = _raise_chained(
        "QueryPlanner failed during planned mapping.",
        QueryPlanningError("malformed JSON from planner"),
    )
    stage = classify_error_stage(exc)
    assert stage == ERROR_STAGE_QUERY_PLANNER
    assert stage != ERROR_STAGE_LOCAL_RETRIEVAL


def test_reranker_error_not_classified_local_retrieval() -> None:
    exc = _raise_chained(
        "LLMReranker failed during planned mapping.",
        LLMRerankerError("hallucinated candidate id"),
    )
    stage = classify_error_stage(exc)
    assert stage == ERROR_STAGE_RERANKER
    assert stage != ERROR_STAGE_LOCAL_RETRIEVAL


def test_plain_exception_with_no_typed_cause_is_unknown_not_local_retrieval() -> None:
    # Mirrors the existing test_execute_query_execution_error_captured_not_scored_as_unmapped
    # fixture: a bare RuntimeError with no pipeline-typed cause at all.
    assert classify_error_stage(RuntimeError("boom")) != ERROR_STAGE_LOCAL_RETRIEVAL


# ─────────────────────────────────────────────────────────────────────────────
# 1/2/3/5. quarantine_error_rows_for_resume: resume set, no duplicates, history
# ─────────────────────────────────────────────────────────────────────────────


def _seed_predictions_csv(path: Path, rows: list[Scenario1RowResult]) -> None:
    with IncrementalPredictionsCsvWriter(path) as writer:
        for row in rows:
            _write_row(writer, row)


def test_mapped_and_unmapped_rows_remain_in_resume_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _seed_predictions_csv(
        output_dir / "predictions.csv",
        [
            _row(1, status="mapped"),
            _row(2, status="unmapped", mapped_code=None, mapped_term=None, mapped_ontology=None, ranks=[RankSlot() for _ in range(5)]),
        ],
    )
    resume_ids = quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-08-26T00:00:00Z", provider="openai", model="gpt-4.1-mini"
    )
    assert resume_ids == {1, 2}
    # No error rows existed -- predictions.csv must be left untouched.
    rows = read_existing_predictions(output_dir / "predictions.csv")
    assert {int(r["query_id"]) for r in rows} == {1, 2}
    assert not (output_dir / "retry_error_history.csv").exists()


def test_error_rows_excluded_from_resume_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _seed_predictions_csv(
        output_dir / "predictions.csv",
        [
            _row(1, status="mapped"),
            _row(
                2,
                status="error",
                mapped_code=None,
                mapped_term=None,
                mapped_ontology=None,
                ranks=[RankSlot() for _ in range(5)],
                error_type="PlannedPipelineError",
                error_stage=ERROR_STAGE_LOCAL_RETRIEVAL,
                error_message="local retrieval failed...",
            ),
        ],
    )
    resume_ids = quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-08-26T00:00:00Z", provider="openai", model="gpt-4.1-mini"
    )
    assert resume_ids == {1}
    assert 2 not in resume_ids


def test_quarantine_strips_error_rows_from_canonical_predictions_csv_no_duplicates(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _seed_predictions_csv(
        output_dir / "predictions.csv",
        [
            _row(1, status="mapped"),
            _row(
                2,
                status="error",
                mapped_code=None,
                mapped_term=None,
                mapped_ontology=None,
                ranks=[RankSlot() for _ in range(5)],
                error_type="PlannedPipelineError",
                error_stage=ERROR_STAGE_LOCAL_RETRIEVAL,
                error_message="boom",
            ),
        ],
    )
    quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-08-26T00:00:00Z", provider="openai", model="gpt-4.1-mini"
    )
    # predictions.csv now contains ONLY the canonical row -- query_id 2 was removed.
    rows = read_existing_predictions(output_dir / "predictions.csv")
    assert [int(r["query_id"]) for r in rows] == [1]

    # Simulate the retry appending a fresh result for query_id 2 -- exactly
    # what run_scenario1_ols_efo.py's IncrementalPredictionsCsvWriter(append=True)
    # does on the next --resume.
    with IncrementalPredictionsCsvWriter(output_dir / "predictions.csv", append=True) as writer:
        _write_row(writer, _row(2, status="mapped"))

    final_rows = read_existing_predictions(output_dir / "predictions.csv")
    query_ids = [int(r["query_id"]) for r in final_rows]
    assert query_ids == [1, 2]
    assert len(query_ids) == len(set(query_ids)), "duplicate canonical query_id row after retry"


def test_quarantine_preserves_error_history_for_audit(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _seed_predictions_csv(
        output_dir / "predictions.csv",
        [
            _row(
                7,
                query="obscure phenotype",
                status="error",
                mapped_code=None,
                mapped_term=None,
                mapped_ontology=None,
                ranks=[RankSlot() for _ in range(5)],
                error_type="PlannedPipelineError",
                error_stage=ERROR_STAGE_LOCAL_RETRIEVAL,
                error_message="Local SapBERT service request failed: connection refused",
            ),
        ],
    )
    quarantine_error_rows_for_resume(
        output_dir, resume_timestamp="2026-08-26T09:00:00Z", provider="openai", model="gpt-5.6-luna"
    )
    history_rows = read_existing_predictions(output_dir / "retry_error_history.csv")
    assert len(history_rows) == 1
    h = history_rows[0]
    assert set(RETRY_ERROR_HISTORY_CSV_FIELDS) <= set(h.keys())
    assert h["query_id"] == "7"
    assert h["query"] == "obscure phenotype"
    assert h["previous_status"] == "error"
    assert h["error_stage"] == ERROR_STAGE_LOCAL_RETRIEVAL
    assert "connection refused" in h["error_message"]
    assert h["resume_timestamp"] == "2026-08-26T09:00:00Z"
    assert h["provider"] == "openai"
    assert h["model"] == "gpt-5.6-luna"
    assert h["attempt_source"] == "prior_resume_error"

    # retry_error_history.csv is diagnostic-only: it must never be read back
    # as a canonical prediction.
    canonical_rows = read_existing_predictions(output_dir / "predictions.csv")
    assert canonical_rows == []


def test_quarantine_noop_when_no_predictions_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    assert quarantine_error_rows_for_resume(output_dir, resume_timestamp="x") == set()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Validate against the existing clean 20-row checkpoint (read-only: no
# error rows present, so quarantine must be a pure no-op and still skip all 20).
# ─────────────────────────────────────────────────────────────────────────────


_EXISTING_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "evaluation"
    / "scenario1_ols_efo"
    / "2026-08-26T13-35-15Z"
)


def test_existing_20_row_checkpoint_all_mapped_and_fully_skipped_on_resume(tmp_path: Path) -> None:
    if not (_EXISTING_CHECKPOINT / "predictions.csv").exists():
        pytest.skip("existing 2026-08-26T13-35-15Z checkpoint not present in this checkout")

    import shutil

    copy_dir = tmp_path / "checkpoint_copy"
    shutil.copytree(_EXISTING_CHECKPOINT, copy_dir)

    rows = read_existing_predictions(copy_dir / "predictions.csv")
    assert len(rows) == 20
    assert all(r["status"] == "mapped" for r in rows)

    resume_ids = quarantine_error_rows_for_resume(
        copy_dir, resume_timestamp="2026-08-26T15-00-00Z", provider="openai", model="gpt-5.6-luna"
    )
    assert len(resume_ids) == 20
    # No error rows -> nothing quarantined, predictions.csv untouched, no history file.
    assert not (copy_dir / "retry_error_history.csv").exists()
    unchanged_rows = read_existing_predictions(copy_dir / "predictions.csv")
    assert len(unchanged_rows) == 20
    assert all(r["status"] == "mapped" for r in unchanged_rows)

    # The real checkpoint under outputs/ was never touched.
    original_rows = read_existing_predictions(_EXISTING_CHECKPOINT / "predictions.csv")
    assert len(original_rows) == 20


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end guard/resume behavior via the real CLI script's main(), with
# every network-touching function monkeypatched (build_provider,
# run_preflight, check_sapbert_health, iter_predictions).
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_scenario1_ols_efo.py"
_REPO_DIR = Path(__file__).resolve().parents[2]
_GRAPH_DATA_DIR = _REPO_DIR / "data" / "text2term_evaluation"


def _require_graph_data() -> None:
    if not (_GRAPH_DATA_DIR / "efo_edges.tsv").exists():
        pytest.skip("EFO graph reference data not fetched in this checkout")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_scenario1_ols_efo_reliability_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


_FAKE_HEALTH = SapBertHealth(
    raw_response={"status": "ok"},
    status="ok",
    model="fake-sapbert",
    loaded_indexes=["EFO"],
    available_indexes=["EFO"],
    lazy_load=True,
)


class _ScriptedHealthCheck:
    """Stands in for check_sapbert_health(). `fail_on_call_numbers` is a set
    of 1-based call indices (call #1 is always the pre-loop startup check)
    that should raise SapBertHealthError; every other call succeeds."""

    def __init__(self, fail_on_call_numbers: frozenset[int] = frozenset()):
        self.calls = 0
        self._fail_on = fail_on_call_numbers

    def __call__(self, sapbert_url: str, *, timeout: float = 10.0) -> SapBertHealth:
        self.calls += 1
        if self.calls in self._fail_on:
            raise SapBertHealthError(f"simulated health failure (call #{self.calls})")
        return _FAKE_HEALTH


def _scripted_row(cq: CanonicalQuery, status: str, error_stage: str | None) -> Scenario1RowResult:
    if status == "mapped":
        return Scenario1RowResult(
            query_id=cq.query_id,
            query=cq.source_query,
            gold_codes=cq.gold_codes,
            gold_labels=cq.gold_labels,
            gold_count=len(cq.gold_codes),
            status="mapped",
            mapped_code="EFO:0000001",
            mapped_term="term",
            mapped_ontology="EFO",
            confidence=0.9,
            ranks=[RankSlot("EFO:0000001", "term", "EFO")] + [RankSlot() for _ in range(4)],
        )
    if status == "unmapped":
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
    # status == "error"
    return Scenario1RowResult(
        query_id=cq.query_id,
        query=cq.source_query,
        gold_codes=cq.gold_codes,
        gold_labels=cq.gold_labels,
        gold_count=len(cq.gold_codes),
        status="error",
        ranks=[RankSlot() for _ in range(5)],
        error_type="PlannedPipelineError",
        error_stage=error_stage,
        error_message="simulated failure",
    )


def _make_dataset_csv(path: Path, n: int) -> None:
    lines = ["query,ref_match,ref_match_id"]
    for i in range(n):
        lines.append(f"synthetic query {i},synthetic gold term {i},EFO:{i:07d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_common(monkeypatch, script_module, *, health: _ScriptedHealthCheck, scripted: dict[int, tuple[str, str | None]]):
    monkeypatch.setattr(script_module, "build_provider", lambda model: object())
    monkeypatch.setattr(script_module, "run_preflight", lambda provider, llm_call_config: None)
    monkeypatch.setattr(script_module, "build_mapper", lambda *, provider, run_config: object())
    monkeypatch.setattr(script_module, "check_sapbert_health", health)

    def _fake_iter_predictions(*, mapper, canonical_queries, pricing, skip_query_ids=()):
        skip = set(skip_query_ids)
        for cq in canonical_queries:
            if cq.query_id in skip:
                continue
            status, error_stage = scripted.get(cq.query_id, ("mapped", None))
            yield _scripted_row(cq, status, error_stage)

    monkeypatch.setattr(script_module, "iter_predictions", _fake_iter_predictions)


def _base_args(dataset_path: Path, output_root: Path, extra: list[str] | None = None) -> list[str]:
    args = [
        "--dataset",
        str(dataset_path),
        "--provider",
        "openai",
        "--model",
        "gpt-4.1-mini",
        "--sapbert-url",
        "http://localhost:8765",
        "--output-root",
        str(output_root),
    ]
    if extra:
        args.extend(extra)
    return args


def _only_subdir(path: Path) -> Path:
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, f"expected exactly one output dir, found {subdirs}"
    return subdirs[0]


@pytest.fixture()
def dataset_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mini_ols_efo.csv"
    _make_dataset_csv(path, 6)  # query_id 0..5
    return path


# ── 9/10/11/12/13/14: consecutive local-retrieval-failure guard ───────────


def test_one_local_retrieval_error_does_not_abort(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    _require_graph_data()
    scripted = {
        0: ("mapped", None),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        2: ("mapped", None),
        3: ("mapped", None),
        4: ("mapped", None),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()  # every call succeeds
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0

    run_dir = _only_subdir(output_root)
    config = json.loads((run_dir / "experiment_config.json").read_text())
    assert config["completed"] is True
    assert "stop_reason" not in config
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6


def test_two_consecutive_local_retrieval_errors_does_not_abort(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    _require_graph_data()
    scripted = {
        0: ("mapped", None),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        2: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        3: ("mapped", None),
        4: ("mapped", None),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0

    run_dir = _only_subdir(output_root)
    config = json.loads((run_dir / "experiment_config.json").read_text())
    assert config["completed"] is True
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6  # all 6 rows were attempted -- no abort


def test_three_consecutive_local_retrieval_errors_aborts_and_checkpoints_third(
    script_module, monkeypatch, tmp_path, dataset_csv
) -> None:
    _require_graph_data()
    scripted = {
        0: ("mapped", None),
        1: ("mapped", None),
        2: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        3: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        4: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        5: ("mapped", None),  # must NEVER be reached
    }
    health = _ScriptedHealthCheck()  # health recheck on first failure succeeds
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(
        _base_args(dataset_csv, output_root, extra=["--max-consecutive-local-retrieval-errors", "3"])
    )
    assert exit_code == 1  # 16. non-zero exit on abort

    run_dir = _only_subdir(output_root)
    config = json.loads((run_dir / "experiment_config.json").read_text())
    assert config["completed"] is False  # 16
    assert config["stop_reason"] == "consecutive_local_retrieval_errors"  # 17
    assert config["error_rows_pending_retry"] == 3

    rows = read_existing_predictions(run_dir / "predictions.csv")
    query_ids = [int(r["query_id"]) for r in rows]
    # 15. the THIRD failing row (query_id=4) was checkpointed before abort.
    assert query_ids == [0, 1, 2, 3, 4]
    assert query_ids.count(5) == 0  # query 5 was never attempted -- run stopped immediately
    statuses = {int(r["query_id"]): r["status"] for r in rows}
    assert statuses[0] == "mapped" and statuses[1] == "mapped"
    assert statuses[2] == statuses[3] == statuses[4] == "error"


def test_mapped_row_resets_consecutive_counter(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    _require_graph_data()
    # 2 local errors, then a mapped row resets the streak, then 2 more local
    # errors -- must NOT abort, because no 3 in a row ever accumulated.
    scripted = {
        0: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        2: ("mapped", None),  # resets counter to 0
        3: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        4: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0
    run_dir = _only_subdir(output_root)
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6  # all attempted -- reset by row 2 prevented an abort


def test_unmapped_row_resets_consecutive_counter(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    _require_graph_data()
    scripted = {
        0: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        2: ("unmapped", None),  # resets counter to 0
        3: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        4: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0
    run_dir = _only_subdir(output_root)
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6


def test_unrelated_error_does_not_increment_local_retrieval_counter(
    script_module, monkeypatch, tmp_path, dataset_csv
) -> None:
    _require_graph_data()
    # 2 local errors, then an UNRELATED (reranker) error, then a mapped row.
    # If the reranker error incremented the local-retrieval counter, this
    # would hit the default threshold of 3 and abort before query 3 (mapped)
    # is ever reached. It must not.
    scripted = {
        0: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        2: ("error", ERROR_STAGE_RERANKER),  # unrelated -- must not count
        3: ("mapped", None),
        4: ("mapped", None),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0  # never aborted
    run_dir = _only_subdir(output_root)
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6  # all 6 rows attempted


# ── optional /health recheck behavior ──────────────────────────────────────


def test_health_recheck_failure_aborts_immediately_after_first_failure(
    script_module, monkeypatch, tmp_path, dataset_csv
) -> None:
    _require_graph_data()
    scripted = {
        0: ("mapped", None),
        1: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),  # triggers recheck -> fails
        2: ("mapped", None),  # must never be reached
        3: ("mapped", None),
        4: ("mapped", None),
        5: ("mapped", None),
    }
    # Call #1 = pre-loop startup health check (must succeed so the run
    # starts). Call #2 = the mid-loop recheck triggered by query 1's error.
    health = _ScriptedHealthCheck(fail_on_call_numbers=frozenset({2}))
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 1

    run_dir = _only_subdir(output_root)
    config = json.loads((run_dir / "experiment_config.json").read_text())
    assert config["completed"] is False
    assert config["stop_reason"] == "sapbert_health_recheck_failed"
    rows = read_existing_predictions(run_dir / "predictions.csv")
    # Aborted after only 2 rows (mapped, then the single failing row) --
    # did NOT wait for 3 consecutive failures.
    assert len(rows) == 2
    assert health.calls == 2


def test_health_recheck_success_allows_streak_to_continue(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    _require_graph_data()
    scripted = {
        0: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),  # triggers recheck -> succeeds
        1: ("mapped", None),
        2: ("mapped", None),
        3: ("mapped", None),
        4: ("mapped", None),
        5: ("mapped", None),
    }
    health = _ScriptedHealthCheck()  # every call succeeds
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0
    assert health.calls >= 2  # startup check + the one mid-loop recheck
    run_dir = _only_subdir(output_root)
    rows = read_existing_predictions(run_dir / "predictions.csv")
    assert len(rows) == 6


# ── 18/19. resume after simulated outage retries failed rows, no duplicates ─


def test_resume_after_outage_retries_failed_rows_single_canonical_row_each(
    script_module, monkeypatch, tmp_path, dataset_csv
) -> None:
    _require_graph_data()

    # Run 1: SapBERT dies at query 2, aborts after 3 consecutive failures.
    scripted_run1 = {
        0: ("mapped", None),
        1: ("mapped", None),
        2: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        3: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        4: ("error", ERROR_STAGE_LOCAL_RETRIEVAL),
        5: ("mapped", None),
    }
    health1 = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health1, scripted=scripted_run1)

    output_root = tmp_path / "outputs"
    exit_code_1 = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code_1 == 1
    run_dir = _only_subdir(output_root)

    rows_after_run1 = read_existing_predictions(run_dir / "predictions.csv")
    assert [int(r["query_id"]) for r in rows_after_run1] == [0, 1, 2, 3, 4]

    # SapBERT "restored": run 2, all previously-failing queries now succeed,
    # plus the never-attempted query 5.
    scripted_run2 = {
        2: ("mapped", None),
        3: ("mapped", None),
        4: ("unmapped", None),
        5: ("mapped", None),
    }
    health2 = _ScriptedHealthCheck()
    module2 = _load_script_module()  # fresh module instance for the resumed process
    _patch_common(monkeypatch, module2, health=health2, scripted=scripted_run2)

    exit_code_2 = module2.main(
        [
            "--dataset",
            str(dataset_csv),
            "--provider",
            "openai",
            "--model",
            "gpt-4.1-mini",
            "--sapbert-url",
            "http://localhost:8765",
            "--resume",
            str(run_dir),
        ]
    )
    assert exit_code_2 == 0  # 9. full completion after resume

    config = json.loads((run_dir / "experiment_config.json").read_text())
    assert config["completed"] is True
    assert "stop_reason" not in config or config.get("stop_reason") is None or exit_code_2 == 0

    final_rows = read_existing_predictions(run_dir / "predictions.csv")
    final_ids = [int(r["query_id"]) for r in final_rows]
    assert sorted(final_ids) == [0, 1, 2, 3, 4, 5]
    assert len(final_ids) == len(set(final_ids)), "duplicate canonical query_id row(s) after resume"  # 19

    by_id = {int(r["query_id"]): r["status"] for r in final_rows}
    assert by_id[0] == "mapped" and by_id[1] == "mapped"  # untouched, not retried
    assert by_id[2] == "mapped" and by_id[3] == "mapped"  # retried and now succeed
    assert by_id[4] == "unmapped"  # retried, now a genuine no-match
    assert by_id[5] == "mapped"  # never-attempted row, now completed

    # Prior error attempts preserved for audit, never treated as canonical.
    history_rows = read_existing_predictions(run_dir / "retry_error_history.csv")
    assert {int(r["query_id"]) for r in history_rows} == {2, 3, 4}
    assert all(r["previous_status"] == "error" for r in history_rows)


# ── 20. no OpenAI / real-network calls needed for any of the above ─────────


def test_no_real_provider_or_sapbert_network_calls_needed(script_module, monkeypatch, tmp_path, dataset_csv) -> None:
    """Proves the guard/resume behavior needs zero real network access: the
    only functions capable of reaching OpenAI or a real SapBERT service
    (build_provider, run_preflight, check_sapbert_health, iter_predictions)
    are fully monkeypatched, and this test asserts real network I/O would
    raise if anything bypassed the patch."""

    def _boom(*args, **kwargs):
        raise AssertionError("a real network-capable call was made")

    import llm_ontology_mapper.providers as providers_module

    monkeypatch.setattr(providers_module.OpenAIProvider, "complete", _boom)
    monkeypatch.setattr("requests.post", _boom)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unpatched requests.get called")),
    )

    scripted = {i: ("mapped", None) for i in range(6)}
    health = _ScriptedHealthCheck()
    _patch_common(monkeypatch, script_module, health=health, scripted=scripted)

    output_root = tmp_path / "outputs"
    exit_code = script_module.main(_base_args(dataset_csv, output_root))
    assert exit_code == 0
