"""
Benchmark dataset loading for dict_mapped_all.xlsx.

Treats the workbook schema as the benchmark contract:
    source_variable -> source_term
    source_label    -> source_label
    source_description -> source_description (blank -> None, never "nan")
    target_ontology -> hard target-ontology constraint
    target_code     -> gold code(s), '|'-separated
    target_term     -> gold term (reference only; scoring uses target_code)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from llm_ontology_mapper.benchmarking.scoring import parse_gold_codes

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_variable",
    "source_label",
    "source_description",
    "target_ontology",
    "target_code",
    "target_term",
)


@dataclass(frozen=True)
class BenchmarkRow:
    """One parsed row of the benchmark dataset contract."""

    input_row: int  # 1-based, stable across runs -- used to join run1/run2
    source_variable: str
    source_label: str | None
    source_description: str | None
    target_ontology: str
    gold_code_raw: str
    gold_codes: list[str]
    gold_target_term: str | None


class BenchmarkDatasetError(ValueError):
    """Raised when the input workbook does not satisfy the benchmark schema contract."""


def blank_to_none(value: Any) -> str | None:
    """Normalize a spreadsheet cell to a prompt-safe string or None.

    Blank cells, pandas NaN, and the literal strings 'nan'/'none'/'null'
    (case-insensitive, from prior lossy round-trips) all become None -- never
    the string "nan".
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def load_dataset(path: str | Path) -> list[BenchmarkRow]:
    """Load and parse dict_mapped_all.xlsx into BenchmarkRow records.

    Raises BenchmarkDatasetError if required columns are missing or a
    required cell (source_variable, target_ontology, target_code) is blank.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise BenchmarkDatasetError(f"Benchmark input file not found: {resolved}")

    df = pd.read_excel(resolved, engine="openpyxl")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BenchmarkDatasetError(
            f"{resolved} is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}."
        )

    rows: list[BenchmarkRow] = []
    for i, record in enumerate(df.to_dict(orient="records"), start=1):
        source_variable = blank_to_none(record.get("source_variable"))
        if source_variable is None:
            raise BenchmarkDatasetError(f"Row {i}: source_variable is required and cannot be blank")

        target_ontology = blank_to_none(record.get("target_ontology"))
        if target_ontology is None:
            raise BenchmarkDatasetError(f"Row {i}: target_ontology is required and cannot be blank")

        gold_code_raw = blank_to_none(record.get("target_code"))
        if gold_code_raw is None:
            raise BenchmarkDatasetError(f"Row {i}: target_code is required and cannot be blank")

        rows.append(
            BenchmarkRow(
                input_row=i,
                source_variable=source_variable,
                source_label=blank_to_none(record.get("source_label")),
                source_description=blank_to_none(record.get("source_description")),
                target_ontology=target_ontology,
                gold_code_raw=gold_code_raw,
                gold_codes=parse_gold_codes(gold_code_raw),
                gold_target_term=blank_to_none(record.get("target_term")),
            )
        )

    return rows


def file_sha256(path: str | Path) -> str:
    """Hash the input workbook so benchmark_config.json can pin the exact input."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
