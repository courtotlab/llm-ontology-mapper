"""
Unit tests for scripts/run_model_benchmark.py: proves one invocation performs
exactly two repetitions for one model and never loops over other
model_registry entries. Network/provider calls are mocked -- this also
exercises the full CLI wiring (dataset -> run loop -> output writing) as a
mocked smoke test without spending API money.

Run with:  pytest tests/benchmarking/test_cli_single_model.py -v -m unit
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_ontology_mapper.benchmarking.dataset import BenchmarkRow
from llm_ontology_mapper.benchmarking.runner import AlternativeSlot, RowResult

pytestmark = pytest.mark.unit

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_model_benchmark.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_model_benchmark_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script_module():
    return _load_script_module()


def _fake_row(*, input_row: int, model: str, run_number: int) -> RowResult:
    return RowResult(
        input_row=input_row,
        source_variable=f"var{input_row}",
        source_label="label",
        source_description=None,
        target_ontology="HPO",
        gold_code_raw="HP:0000001",
        gold_codes_normalized=["HP:0000001"],
        gold_target_term="term",
        mapped_status="mapped",
        mapped_code="HP:0000001",
        mapped_code_normalized="HP:0000001",
        mapped_term="term",
        mapped_ontology="HPO",
        confidence=0.9,
        logic_type="rag",
        alternatives=[AlternativeSlot(None, None, None, None) for _ in range(4)],
        gold_rank=1,
        top1_correct=True,
        top5_hit=True,
        tp=1.0,
        fp=0.0,
        fn=0.0,
        tn=0.0,
        end_to_end_seconds=0.01,
        model=model,
        requested_reasoning_effort="N/A",
        temperature=0.0,
        seed=42,
        retrieval_mode="public",
        max_alternatives=4,
        run_number=run_number,
    )


def test_one_invocation_runs_exactly_two_reps_for_one_model_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_module: Any
) -> None:
    model = "gpt-4.1-mini"
    dataset = [
        BenchmarkRow(
            input_row=1,
            source_variable="var1",
            source_label="label",
            source_description=None,
            target_ontology="HPO",
            gold_code_raw="HP:0000001",
            gold_codes=["HP:0000001"],
            gold_target_term="term",
        )
    ]

    monkeypatch.setattr(script_module, "load_dataset", lambda path: dataset)

    build_provider_calls: list[str] = []

    def _fake_build_provider(model_name: str) -> Any:
        build_provider_calls.append(model_name)
        fake = type("FakeProvider", (), {"model": model_name})()
        return fake

    monkeypatch.setattr(script_module, "build_provider", _fake_build_provider)
    monkeypatch.setattr(script_module, "run_preflight", lambda provider, cfg: None)
    monkeypatch.setattr(
        script_module,
        "build_pipeline_and_mappers",
        lambda **kwargs: (MagicMockPipeline(), {"HPO": object()}),
    )

    iter_run_rows_calls: list[int] = []

    def _fake_iter_run_rows(*, mappers, dataset, run_number, run_config, pricing):
        iter_run_rows_calls.append(run_number)
        assert run_config.model_config.model == model
        for row in dataset:
            yield _fake_row(input_row=row.input_row, model=model, run_number=run_number)

    monkeypatch.setattr(script_module, "iter_run_rows", _fake_iter_run_rows)

    exit_code = script_module.main(
        [
            "--input",
            "dict_mapped_all.xlsx",
            "--model",
            model,
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert iter_run_rows_calls == [1, 2]  # exactly two reps
    assert build_provider_calls == [model]  # never touched another registry entry

    output_dirs = list((tmp_path / model).glob("*"))
    assert len(output_dirs) == 1
    run_dir = output_dirs[0]
    assert (run_dir / "benchmark_config.json").exists()
    assert (run_dir / "run_1_raw.csv").exists()
    assert (run_dir / "run_2_raw.csv").exists()
    assert (run_dir / "run_1_summary.csv").exists()
    assert (run_dir / "run_2_summary.csv").exists()
    assert (run_dir / "model_summary.csv").exists()
    assert (run_dir / "reproducibility_summary.csv").exists()


def test_cli_rejects_unsupported_model(script_module: Any) -> None:
    with pytest.raises(SystemExit):
        script_module.parse_args(["--input", "x.xlsx", "--model", "not-a-real-model"])


def test_temperature_flag_defaults_to_provider_default_none(script_module: Any) -> None:
    """The normal command (no --temperature flag) must default to
    provider-default temperature -- users should not need an extra flag to
    run a supported model like gpt-5.6-luna."""
    args = script_module.parse_args(["--input", "x.xlsx", "--model", "gpt-5.6-luna"])
    assert args.temperature is None


def test_temperature_flag_explicit_override_still_parses(script_module: Any) -> None:
    args = script_module.parse_args(
        ["--input", "x.xlsx", "--model", "gpt-5.6-luna", "--temperature", "0.0"]
    )
    assert args.temperature == 0.0


def test_main_uses_provider_default_temperature_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_module: Any
) -> None:
    """End-to-end (mocked) proof that the normal CLI invocation with no
    --temperature flag builds a run_config/llm_call_config with temperature
    omitted, and that gpt-4.1-mini's reasoning_effort stays omitted too."""
    model = "gpt-4.1-mini"
    dataset = [
        BenchmarkRow(
            input_row=1,
            source_variable="var1",
            source_label="label",
            source_description=None,
            target_ontology="HPO",
            gold_code_raw="HP:0000001",
            gold_codes=["HP:0000001"],
            gold_target_term="term",
        )
    ]

    monkeypatch.setattr(script_module, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(
        script_module, "build_provider", lambda model_name: type("FakeProvider", (), {"model": model_name})()
    )

    seen_configs: list[Any] = []

    def _fake_run_preflight(provider: Any, llm_call_config: Any) -> None:
        seen_configs.append(llm_call_config)

    monkeypatch.setattr(script_module, "run_preflight", _fake_run_preflight)
    monkeypatch.setattr(
        script_module,
        "build_pipeline_and_mappers",
        lambda **kwargs: (MagicMockPipeline(), {"HPO": object()}),
    )

    def _fake_iter_run_rows(*, mappers, dataset, run_number, run_config, pricing):
        assert run_config.temperature is None
        for row in dataset:
            yield _fake_row(input_row=row.input_row, model=model, run_number=run_number)

    monkeypatch.setattr(script_module, "iter_run_rows", _fake_iter_run_rows)

    exit_code = script_module.main(
        ["--input", "dict_mapped_all.xlsx", "--model", model, "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert len(seen_configs) == 1
    assert seen_configs[0].temperature is None
    assert seen_configs[0].force_temperature is True
    assert seen_configs[0].reasoning_effort is None  # gpt-4.1-mini omits reasoning_effort

    config_path = next((tmp_path / model).glob("*/benchmark_config.json"))
    written = json.loads(config_path.read_text())
    assert written["temperature"] is None
    assert written["temperature_mode"] == "provider_default"
    assert written["reasoning_effort"] == "N/A"


class MagicMockPipeline:
    """Placeholder pipeline object -- build_pipeline_and_mappers is mocked so
    the real PlannedPipeline is never constructed in this CLI-wiring test."""
