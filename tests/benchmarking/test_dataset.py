"""
Unit tests for llm_ontology_mapper.benchmarking.dataset.

Uses a small in-memory xlsx fixture written to a temp dir -- no network.

Run with:  pytest tests/benchmarking/test_dataset.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_ontology_mapper.benchmarking.dataset import (
    BenchmarkDatasetError,
    blank_to_none,
    file_sha256,
    load_dataset,
)

pytestmark = pytest.mark.unit


def _write_xlsx(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")


def test_blank_source_description_becomes_none() -> None:
    assert blank_to_none(float("nan")) is None
    assert blank_to_none(None) is None
    assert blank_to_none("") is None
    assert blank_to_none("   ") is None
    assert blank_to_none("nan") is None


def test_blank_to_none_preserves_real_text() -> None:
    assert blank_to_none("  headache disorder  ") == "headache disorder"


def test_load_dataset_parses_rows(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "dict_mapped_all.xlsx"
    _write_xlsx(
        xlsx_path,
        [
            {
                "source_variable": "sinus_pain",
                "source_label": "Sinus pain/congestion",
                "source_description": float("nan"),
                "target_ontology": "HPO",
                "target_code": "HP:0000245 | HP:0001742",
                "target_term": "Abnormal paranasal sinus morphology | Nasal congestion",
            },
            {
                "source_variable": "nosebleed",
                "source_label": "Nosebleed",
                "source_description": "bleeding from the nose",
                "target_ontology": "HPO",
                "target_code": "HP:0000421",
                "target_term": "Epistaxis",
            },
        ],
    )

    rows = load_dataset(xlsx_path)
    assert len(rows) == 2

    row1 = rows[0]
    assert row1.input_row == 1
    assert row1.source_variable == "sinus_pain"
    assert row1.source_description is None  # blank cell, not "nan" string
    assert row1.target_ontology == "HPO"
    assert row1.gold_code_raw == "HP:0000245 | HP:0001742"
    assert row1.gold_codes == ["HP:0000245", "HP:0001742"]

    row2 = rows[1]
    assert row2.input_row == 2
    assert row2.source_description == "bleeding from the nose"
    assert row2.gold_codes == ["HP:0000421"]


def test_load_dataset_missing_required_column_raises(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "bad.xlsx"
    _write_xlsx(
        xlsx_path,
        [{"source_variable": "x", "target_ontology": "HPO", "target_code": "HP:1"}],
    )
    with pytest.raises(BenchmarkDatasetError, match="missing required columns"):
        load_dataset(xlsx_path)


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkDatasetError, match="not found"):
        load_dataset(tmp_path / "does_not_exist.xlsx")


def test_load_dataset_blank_target_code_raises(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "blank_code.xlsx"
    _write_xlsx(
        xlsx_path,
        [
            {
                "source_variable": "x",
                "source_label": "x",
                "source_description": float("nan"),
                "target_ontology": "HPO",
                "target_code": float("nan"),
                "target_term": "x",
            }
        ],
    )
    with pytest.raises(BenchmarkDatasetError, match="target_code"):
        load_dataset(xlsx_path)


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "dict_mapped_all.xlsx"
    _write_xlsx(xlsx_path, [{"source_variable": "x", "source_label": "x",
                              "source_description": "x", "target_ontology": "HPO",
                              "target_code": "HP:1", "target_term": "x"}])
    h1 = file_sha256(xlsx_path)
    h2 = file_sha256(xlsx_path)
    assert h1 == h2
    assert len(h1) == 64
