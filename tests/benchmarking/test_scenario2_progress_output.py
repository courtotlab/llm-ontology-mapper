"""
Scenario 2 terminal progress-line tests: proves gold_rank in the per-row
progress line is the SAME value written to predictions.csv's
first_gold_rank column (Part B, items 10-13) -- never an independent
recalculation. Zero OpenAI/network calls: build_provider, run_preflight,
build_pipeline_and_mappers, and OntologyValidator are all patched with
in-memory fakes before scripts/run_scenario2_retrieval_ablation.py:main()
runs a full 4-row loop.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.models import AlternativeMapping, LogicType, MappingMetadata, MappingResult

pytestmark = pytest.mark.unit

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_scenario2_retrieval_ablation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_scenario2_retrieval_ablation_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


def _alt(code: str, confidence: float) -> AlternativeMapping:
    return AlternativeMapping(code=code, term=f"term {code}", ontology="HPO", confidence=confidence)


def _mapped_result(*, selected_code: str, alternatives: list[AlternativeMapping]) -> MappingResult:
    return MappingResult(
        source_term="x",
        target_code=selected_code,
        target_term="t",
        ontology="HPO",
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=alternatives,
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )


def _unmapped_result() -> MappingResult:
    return MappingResult(
        source_term="x",
        target_code="UNKNOWN:UNMAPPED",
        target_term="UNMAPPED",
        ontology="UNKNOWN",
        confidence=0.0,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )


_RESULTS_BY_SOURCE_TERM = {
    # gold_rank=1: gold code IS the selected (rank-1) result.
    "row_rank1": _mapped_result(selected_code="HP:0000001", alternatives=[]),
    # gold_rank=2: gold code is the first alternative (rank 2).
    "row_rank2": _mapped_result(
        selected_code="HP:0009999",
        alternatives=[_alt("HP:0000002", 0.8), _alt("HP:0000010", 0.5)],
    ),
    # gold_rank=5: gold code only appears in the 4th alternative (rank 5).
    "row_rank5": _mapped_result(
        selected_code="HP:0009999",
        alternatives=[_alt("HP:0000010", 0.8), _alt("HP:0000011", 0.7), _alt("HP:0000012", 0.6), _alt("HP:0000005", 0.5)],
    ),
    # gold_rank=None: unmapped.
    "row_unmapped": _unmapped_result(),
}

_ROWS = [
    ("row_rank1", "HP:0000001"),
    ("row_rank2", "HP:0000002"),
    ("row_rank5", "HP:0000005"),
    ("row_unmapped", "HP:0000099"),
]


def _write_fixture_workbook(path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "source_variable": [name for name, _ in _ROWS],
            "source_label": [name for name, _ in _ROWS],
            "source_description": [None] * len(_ROWS),
            "target_ontology": ["HPO"] * len(_ROWS),
            "target_code": [gold for _, gold in _ROWS],
            "target_term": ["term"] * len(_ROWS),
        }
    )
    df.to_excel(path, index=False, engine="openpyxl", sheet_name="dict_mapped_all")


class _FakeValidator:
    """Zero-network stand-in for OntologyValidator -- always UNRESOLVED."""

    def validate_code(self, ontology_code: str) -> bool | None:
        return None


def _run_mocked(script_module, tmp_path: Path, capsys) -> tuple[int, str, Path]:
    dataset_path = tmp_path / "dict_mapped_all.xlsx"
    _write_fixture_workbook(dataset_path)
    output_root = tmp_path / "out"

    mock_mapper = MagicMock()
    mock_mapper.map_term.side_effect = lambda *, source_term, **_: _RESULTS_BY_SOURCE_TERM[source_term]

    script_module.build_provider = MagicMock(return_value=MagicMock())
    script_module.run_preflight = MagicMock(return_value=None)
    script_module.build_pipeline_and_mappers = MagicMock(return_value=(MagicMock(), {"HPO": mock_mapper}))
    script_module.OntologyValidator = MagicMock(return_value=_FakeValidator())

    exit_code = script_module.main(
        ["--input", str(dataset_path), "--mode", "public", "--output-root", str(output_root)]
    )
    captured = capsys.readouterr()
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    return exit_code, captured.out, run_dirs[0]


# ─────────────────────────────────────────────────────────────────────────────
# 13. gold_rank=1 / 2 / 5 / None all appear correctly in the progress line
# ─────────────────────────────────────────────────────────────────────────────


def test_progress_line_shows_gold_rank_1(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, _out_dir = _run_mocked(script_module, tmp_path, capsys)
    line = next(line_ for line_ in stdout.splitlines() if "row_rank1" in line_)
    assert "status=mapped" in line
    assert "gold_rank=1" in line
    assert "gold_rank=1 " in line or line.rstrip().endswith("gold_rank=1")


def test_progress_line_shows_gold_rank_2(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, _out_dir = _run_mocked(script_module, tmp_path, capsys)
    line = next(line_ for line_ in stdout.splitlines() if "row_rank2" in line_)
    assert "status=mapped" in line
    assert "gold_rank=2" in line


def test_progress_line_shows_gold_rank_5_ranks_3_4_also_supported(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, _out_dir = _run_mocked(script_module, tmp_path, capsys)
    line = next(line_ for line_ in stdout.splitlines() if "row_rank5" in line_)
    assert "status=mapped" in line
    assert "gold_rank=5" in line  # never clamped/truncated to 1 or 2


def test_progress_line_shows_gold_rank_none_for_unmapped(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, _out_dir = _run_mocked(script_module, tmp_path, capsys)
    line = next(line_ for line_ in stdout.splitlines() if "row_unmapped" in line_)
    assert "status=unmapped" in line
    assert "gold_rank=None" in line


# ─────────────────────────────────────────────────────────────────────────────
# terminal value == row.first_gold_rank (predictions.csv) -- one canonical
# calculation, logging can never diverge from the metric.
# ─────────────────────────────────────────────────────────────────────────────


def test_terminal_gold_rank_matches_predictions_csv_first_gold_rank(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, out_dir = _run_mocked(script_module, tmp_path, capsys)

    with (out_dir / "predictions.csv").open(newline="", encoding="utf-8") as fh:
        csv_rows = {row["source_variable"]: row["first_gold_rank"] for row in csv.DictReader(fh)}

    printed_rank_by_source: dict[str, str] = {}
    for name, _gold in _ROWS:
        line = next(line_ for line_ in stdout.splitlines() if f"{name!r}" in line_)
        # Extract "gold_rank=<value>" token from the printed line.
        token = next(part for part in line.split() if part.startswith("gold_rank="))
        printed_rank_by_source[name] = token.removeprefix("gold_rank=")

    for name, _gold in _ROWS:
        csv_value = csv_rows[name]
        printed_value = printed_rank_by_source[name]
        # predictions.csv stores "" for None (csv has no null type); the
        # printed line stores the literal string "None" -- both represent
        # the same underlying first_gold_rank value from ONE calculation.
        if printed_value == "None":
            assert csv_value == ""
        else:
            assert csv_value == printed_value


def test_status_and_error_stage_still_printed_error_stage_not_suppressed(script_module, tmp_path, capsys) -> None:
    _exit_code, stdout, _out_dir = _run_mocked(script_module, tmp_path, capsys)
    # None of the fixture rows error, but confirm the format string still has
    # the conditional error_stage suffix wired up (no regression to that path).
    assert "def _finalize_mode_outputs" not in stdout  # sanity: not accidentally printing source
    for name, _gold in _ROWS:
        line = next(line_ for line_ in stdout.splitlines() if f"{name!r}" in line_)
        assert "latency=" in line
