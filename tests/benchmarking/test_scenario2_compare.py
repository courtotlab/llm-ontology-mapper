"""
Scenario 2 cross-mode comparison tests (Part 30, items 46-51).

Filesystem-only (tmp_path) -- no network, no mapper, no LLM calls. This is
exactly what --compare must guarantee: these tests prove the code path never
imports/constructs a provider or mapper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_ontology_mapper.benchmarking.scenario2_compare import (
    CompareDatasetMismatchError,
    build_paired_predictions,
    load_and_validate_runs,
)
from llm_ontology_mapper.benchmarking.scenario2_output import (
    PREDICTIONS_CSV_FIELDS,
    CompareConfigMismatchError,
    validate_compare_configs,
)
from llm_ontology_mapper.benchmarking.scenario2_reliability_plot import plot_reliability_diagram

pytestmark = pytest.mark.unit

_BASE_CONFIG = {
    "source_dataset_sha256": "abc123",
    "dataset_row_count": 3,
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "reasoning_effort": "low",
    "temperature": None,
    "temperature_mode": "provider_default",
    "seed": 42,
    "max_alternatives": 4,
    "strict_target_ontology": False,
}

_ROWS = [
    {"row_id": 1, "source_variable": "age", "source_label": "Age", "source_description": "",
     "target_ontology": "HPO", "gold_codes": "HP:0001507"},
    {"row_id": 2, "source_variable": "wt", "source_label": "Weight", "source_description": "",
     "target_ontology": "MONDO", "gold_codes": "MONDO:0000001"},
    {"row_id": 3, "source_variable": "dx", "source_label": "Diagnosis", "source_description": "",
     "target_ontology": "LOINC", "gold_codes": "LOINC:1234-5"},
]


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PREDICTIONS_CSV_FIELDS})


def _make_run_dir(tmp_path: Path, mode: str, *, rows=None, config_overrides=None) -> Path:
    out = tmp_path / mode
    out.mkdir()
    config = dict(_BASE_CONFIG)
    config["retrieval_mode"] = mode
    if config_overrides:
        config.update(config_overrides)
    (out / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_predictions(out / "predictions.csv", rows if rows is not None else _ROWS)
    return out


def _prediction_row(row_id, *, code, correct, confidence, status="mapped", grounded=True, valid="VALID"):
    return {
        "row_id": row_id,
        "source_variable": _ROWS[row_id - 1]["source_variable"],
        "source_label": _ROWS[row_id - 1]["source_label"],
        "source_description": "",
        "target_ontology": _ROWS[row_id - 1]["target_ontology"],
        "gold_codes": _ROWS[row_id - 1]["gold_codes"],
        "status": status,
        "mapped_code_normalized": code,
        "confidence": confidence,
        "semantic_correctness": str(correct),
        "is_grounded": str(grounded),
        "validation_status": valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 46. dataset mismatch rejected
# ─────────────────────────────────────────────────────────────────────────────


def test_dataset_row_id_mismatch_rejected(tmp_path: Path) -> None:
    public_dir = _make_run_dir(tmp_path, "public")
    local_dir = _make_run_dir(tmp_path, "local")
    disabled_rows = [dict(r) for r in _ROWS]
    disabled_rows[0]["row_id"] = 99  # different row_id set
    disabled_dir = _make_run_dir(tmp_path, "disabled", rows=disabled_rows)

    with pytest.raises(CompareDatasetMismatchError):
        load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


def test_dataset_gold_code_mismatch_rejected(tmp_path: Path) -> None:
    public_dir = _make_run_dir(tmp_path, "public")
    local_dir = _make_run_dir(tmp_path, "local")
    mismatched_rows = [dict(r) for r in _ROWS]
    mismatched_rows[0]["gold_codes"] = "HP:9999999"  # different gold for the same row_id
    disabled_dir = _make_run_dir(tmp_path, "disabled", rows=mismatched_rows)

    with pytest.raises(CompareDatasetMismatchError):
        load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 47. config mismatch rejected
# ─────────────────────────────────────────────────────────────────────────────


def test_model_mismatch_rejected(tmp_path: Path) -> None:
    public_dir = _make_run_dir(tmp_path, "public")
    local_dir = _make_run_dir(tmp_path, "local")
    disabled_dir = _make_run_dir(tmp_path, "disabled", config_overrides={"model": "gpt-5.4-mini"})

    with pytest.raises(CompareConfigMismatchError):
        load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)


def test_temperature_mismatch_rejected(tmp_path: Path) -> None:
    public_dir = _make_run_dir(tmp_path, "public")
    local_dir = _make_run_dir(tmp_path, "local", config_overrides={"temperature": 0.5, "temperature_mode": "explicit"})
    disabled_dir = _make_run_dir(tmp_path, "disabled")

    with pytest.raises(CompareConfigMismatchError):
        validate_compare_configs(
            {
                "public": json.loads((public_dir / "experiment_config.json").read_text()),
                "local": json.loads((local_dir / "experiment_config.json").read_text()),
                "disabled": json.loads((disabled_dir / "experiment_config.json").read_text()),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# 48. retrieval_mode difference allowed (this is the whole point of the ablation)
# ─────────────────────────────────────────────────────────────────────────────


def test_retrieval_mode_difference_is_allowed(tmp_path: Path) -> None:
    public_dir = _make_run_dir(tmp_path, "public")
    local_dir = _make_run_dir(tmp_path, "local", config_overrides={"sapbert_url": "http://localhost:8765"})
    disabled_dir = _make_run_dir(tmp_path, "disabled")

    runs = load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    assert runs["public"].config["retrieval_mode"] == "public"
    assert runs["local"].config["retrieval_mode"] == "local"
    assert runs["disabled"].config["retrieval_mode"] == "disabled"


# ─────────────────────────────────────────────────────────────────────────────
# 49. row pairing correct
# ─────────────────────────────────────────────────────────────────────────────


def test_paired_predictions_row_pairing(tmp_path: Path) -> None:
    public_rows = [
        _prediction_row(1, code="HP:0001507", correct=True, confidence=0.9),
        _prediction_row(2, code="MONDO:0000002", correct=False, confidence=0.6),
        _prediction_row(3, code="LOINC:1234-5", correct=True, confidence=0.8),
    ]
    local_rows = [
        _prediction_row(1, code="HP:0001507", correct=True, confidence=0.85),
        _prediction_row(2, code="MONDO:0000001", correct=True, confidence=0.7),
        _prediction_row(3, code="LOINC:9999-9", correct=False, confidence=0.4),
    ]
    disabled_rows = [
        _prediction_row(1, code="HP:0001507", correct=True, confidence=0.5, grounded=False),
        _prediction_row(2, code="MONDO:0000002", correct=False, confidence=0.3, grounded=False),
        _prediction_row(3, code=None, correct=False, confidence=0.2, status="unmapped", grounded=False),
    ]
    public_dir = _make_run_dir(tmp_path, "public", rows=public_rows)
    local_dir = _make_run_dir(tmp_path, "local", rows=local_rows, config_overrides={"sapbert_url": "http://x"})
    disabled_dir = _make_run_dir(tmp_path, "disabled", rows=disabled_rows)

    runs = load_and_validate_runs(public_dir=public_dir, local_dir=local_dir, disabled_dir=disabled_dir)
    paired = build_paired_predictions(runs)
    assert len(paired) == 3

    row1 = next(r for r in paired if r["row_id"] == 1)
    assert row1["public_code"] == "HP:0001507"
    assert row1["local_code"] == "HP:0001507"
    assert row1["disabled_code"] == "HP:0001507"
    assert row1["public_local_same_code"] is True
    assert row1["public_disabled_same_code"] is True

    row2 = next(r for r in paired if r["row_id"] == 2)
    assert row2["public_correct"] is False
    assert row2["local_correct"] is True
    assert row2["public_local_same_code"] is False

    row3 = next(r for r in paired if r["row_id"] == 3)
    assert row3["disabled_status"] == "unmapped"
    assert row3["disabled_code"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 50. reliability bins combine correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_reliability_diagram_combines_three_modes(tmp_path: Path) -> None:
    bins_by_mode = {
        "public": [
            {"bin_lower": "0.8", "bin_upper": "0.9", "mean_confidence": "0.85", "empirical_accuracy": "0.9", "count": "10"},
        ],
        "local": [
            {"bin_lower": "0.8", "bin_upper": "0.9", "mean_confidence": "0.82", "empirical_accuracy": "0.7", "count": "8"},
        ],
        "disabled": [
            {"bin_lower": "0.8", "bin_upper": "0.9", "mean_confidence": "0.81", "empirical_accuracy": "0.5", "count": "5"},
        ],
    }
    out_png = tmp_path / "reliability_diagram.png"
    out_svg = tmp_path / "reliability_diagram.svg"
    out_pdf = tmp_path / "reliability_diagram.pdf"
    plot_reliability_diagram(bins_by_mode, output_png=out_png, output_svg=out_svg, output_pdf=out_pdf)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_svg.exists() and out_svg.stat().st_size > 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# 51. compare mode makes zero mapping/LLM calls -- the compare module never
# imports a provider/mapper/OpenAI client.
# ─────────────────────────────────────────────────────────────────────────────


def test_compare_module_has_no_llm_or_mapper_imports() -> None:
    import llm_ontology_mapper.benchmarking.scenario2_compare as compare_module

    source = Path(compare_module.__file__).read_text(encoding="utf-8")
    forbidden = ["OpenAIProvider", "OntologyMapper", "PlannedPipeline", "import openai"]
    for token in forbidden:
        assert token not in source, f"{token!r} must never appear in the zero-LLM-call compare module"
