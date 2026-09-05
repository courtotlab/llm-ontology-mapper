"""
Scenario 2 dataset tests (Part 30, items 1-7).

Loads the ACTUAL dict_mapped_all.xlsx workbook at the repo root -- no network,
no mapper, no LLM calls -- and asserts the exact figures the Scenario 2 spec
expects, plus that scenario2_runner forwards source fields to the mapper
identically to the preceding model-selection benchmark (runner.py).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.benchmarking.dataset import file_sha256, load_dataset
from llm_ontology_mapper.benchmarking.model_registry import get_model_config
from llm_ontology_mapper.benchmarking.pricing import get_pricing
from llm_ontology_mapper.benchmarking.scenario2_dataset import audit_dataset
from llm_ontology_mapper.benchmarking.scenario2_runner import Scenario2RunConfig, execute_row
from llm_ontology_mapper.models import LogicType, MappingMetadata, MappingResult

REPO_DIR = Path(__file__).resolve().parents[2]
WORKBOOK = REPO_DIR / "dict_mapped_all.xlsx"
EXPECTED_SHA256 = "91980c4df28781e5ef8d33d614c4f768966fa88b3db24996f3da38fc01bbddfd"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not WORKBOOK.exists(), reason="dict_mapped_all.xlsx not present at repo root"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. workbook loads 218 canonical rows
# ─────────────────────────────────────────────────────────────────────────────


def test_workbook_loads_218_rows() -> None:
    rows = load_dataset(WORKBOOK)
    assert len(rows) == 218


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHA validation
# ─────────────────────────────────────────────────────────────────────────────


def test_workbook_sha256_matches_expected() -> None:
    assert file_sha256(WORKBOOK) == EXPECTED_SHA256


# ─────────────────────────────────────────────────────────────────────────────
# 3. target ontology distribution
# ─────────────────────────────────────────────────────────────────────────────


def test_target_ontology_distribution() -> None:
    rows = load_dataset(WORKBOOK)
    audit = audit_dataset(rows)
    assert audit.ontology_distribution == {
        "HPO": 64,
        "MONDO": 49,
        "LOINC": 47,
        "ICD10": 16,
        "SNOMED": 15,
        "NCIT": 14,
        "RxNorm": 13,
    }
    assert audit.namespaces_consistent


# ─────────────────────────────────────────────────────────────────────────────
# 4/5. multi-gold splitting: 212 single-gold, 6 double-gold, max 2
# ─────────────────────────────────────────────────────────────────────────────


def test_gold_cardinality_distribution() -> None:
    rows = load_dataset(WORKBOOK)
    audit = audit_dataset(rows)
    assert audit.gold_cardinality_distribution == {1: 212, 2: 6}
    assert audit.max_gold_codes_per_row == 2


def test_six_multi_gold_rows_have_two_distinct_codes() -> None:
    rows = load_dataset(WORKBOOK)
    multi_gold = [r for r in rows if len(r.gold_codes) == 2]
    assert len(multi_gold) == 6
    for row in multi_gold:
        assert len(set(row.gold_codes)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. source_description blanks preserved (38 populated / 180 blank)
# ─────────────────────────────────────────────────────────────────────────────


def test_source_description_split() -> None:
    rows = load_dataset(WORKBOOK)
    audit = audit_dataset(rows)
    assert audit.source_description_populated == 38
    assert audit.source_description_blank == 180
    assert audit.source_description_populated + audit.source_description_blank == len(rows)


def test_source_description_never_synthesized_from_label() -> None:
    rows = load_dataset(WORKBOOK)
    for row in rows:
        if row.source_description is None:
            continue
        # A populated description must be its own text, not a copy of source_label.
        assert row.source_description != row.source_label or row.source_label is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. same dataset-to-mapper semantics as the existing model benchmark
# (runner.execute_row calls mapper.map_term(source_term=row.source_variable,
# source_label=row.source_label, source_description=row.source_description)
# with no strict_target_ontology override -- scenario2_runner.execute_row
# must forward the IDENTICAL kwargs for the identical row).
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_row_forwards_source_fields_identically_to_model_benchmark() -> None:
    rows = load_dataset(WORKBOOK)
    row = next(r for r in rows if r.source_description is not None)

    mapper = MagicMock()
    mapper.map_term.return_value = MappingResult(
        source_term=row.source_variable,
        target_code=row.gold_codes[0],
        target_term="term",
        ontology=row.target_ontology,
        confidence=0.9,
        logic_type=LogicType.RAG,
        alternatives=[],
        metadata=MappingMetadata(model="m", provider="p", rag_debug=None),
    )

    model_cfg = get_model_config("gpt-5.6-luna")
    run_config = Scenario2RunConfig(model_config=model_cfg, retrieval_mode="public")
    pricing = get_pricing("gpt-5.6-luna")

    execute_row(mapper=mapper, row=row, run_config=run_config, pricing=pricing)

    _, kwargs = mapper.map_term.call_args
    assert kwargs["source_term"] == row.source_variable
    assert kwargs["source_label"] == row.source_label
    assert kwargs["source_description"] == row.source_description
    assert "strict_target_ontology" not in kwargs
