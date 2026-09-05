"""Direct live smoke script for OpenAI agentic ontology mapping.

Run with:
    uv run python tests/live/agentic_openai_smoke.py
"""

from __future__ import annotations

import json
import os

from llm_ontology_mapper.agentic_mapper import AgenticMapper
from llm_ontology_mapper.providers import OpenAIProvider
from llm_ontology_mapper.search_tools import SearchTools


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================
# Change the model, mapping input, and expected output here.
# Keep API keys and passwords out of this file.

OPENAI_MODEL = "gpt-4.1-mini"

SOURCE_TERM = "cough"
SOURCE_LABEL = "Does the patient have a cough?"
TARGET_ONTOLOGY = "HPO"
CLINICAL_AREA = "phenotype"

EXPECT_PREFIX = "HP:"
EXPECT_ONTOLOGY = "HPO"
EXPECT_LOGIC_TYPE = "agentic"
REQUIRE_MAPPED = True


# =============================================================================
# REQUIRED ENVIRONMENT VARIABLES
# =============================================================================
# OpenAI requires:
#   export OPENAI_API_KEY="..."
#
# A LOINC case additionally requires:
#   export LOINC_USERNAME="..."
#   export LOINC_PASSWORD="..."


def _optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _logic_value(result: object) -> str:
    logic_type = getattr(result, "logic_type")
    return str(getattr(logic_type, "value", logic_type))


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Run: export OPENAI_API_KEY='...'"
        )

    source_label = _optional(SOURCE_LABEL)
    target_ontology = _optional(TARGET_ONTOLOGY)
    clinical_area = _optional(CLINICAL_AREA)
    if (target_ontology or "").upper() == "LOINC" and (
        not os.environ.get("LOINC_USERNAME")
        or not os.environ.get("LOINC_PASSWORD")
    ):
        raise RuntimeError(
            "LOINC_USERNAME and LOINC_PASSWORD are required for LOINC smoke tests."
        )

    print(json.dumps({
        "provider": "openai",
        "model": OPENAI_MODEL,
        "source_term": SOURCE_TERM,
        "source_label": source_label,
        "target_ontology": target_ontology,
        "clinical_area": clinical_area,
    }, indent=2))

    # Actual live mapper call.
    mapper = AgenticMapper(
        provider=OpenAIProvider(model=OPENAI_MODEL, api_key=api_key),
        search_tools=SearchTools(api_timeout=15, request_delay=0.1),
    )
    result = mapper.map(
        source_term=SOURCE_TERM,
        source_label=source_label,
        target_ontology=target_ontology,
        clinical_area=clinical_area,
    )

    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    print(json.dumps(payload, indent=2, default=str))

    # Output validation and smoke checks.
    assert result is not None, "Mapper returned no result"
    assert result.target_code.strip(), "target_code is empty"
    assert result.target_term.strip(), "target_term is empty"
    if REQUIRE_MAPPED:
        assert result.target_code != "UNKNOWN:UNMAPPED", "Result was not mapped"
    if EXPECT_PREFIX:
        assert result.target_code.startswith(EXPECT_PREFIX), (
            f"Expected code prefix {EXPECT_PREFIX!r}, got {result.target_code!r}"
        )
    if EXPECT_ONTOLOGY:
        assert result.ontology == EXPECT_ONTOLOGY, (
            f"Expected ontology {EXPECT_ONTOLOGY!r}, got {result.ontology!r}"
        )
    if EXPECT_LOGIC_TYPE:
        assert _logic_value(result) == EXPECT_LOGIC_TYPE, (
            f"Expected logic type {EXPECT_LOGIC_TYPE!r}, got {_logic_value(result)!r}"
        )
    print("PASS")


if __name__ == "__main__":
    main()
