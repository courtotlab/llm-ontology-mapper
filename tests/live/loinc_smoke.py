"""Direct live smoke script for the LOINC Search API integration.

This bypasses the LLM and tests only SearchTools.search_loinc().

Run with:
    uv run python tests/live/loinc_smoke.py
"""

from __future__ import annotations

import json
import os

from llm_ontology_mapper.search_tools import SearchTools


# =============================================================================
# EDIT THIS SECTION FOR LOCAL SMOKE TESTING
# =============================================================================
# Change the query, result count, and expected code here.
# Set EXPECT_CODE = "" to accept any returned LOINC code.

QUERY = "systolic blood pressure"
TOP_K = 5
EXPECT_CODE = "LOINC:8480-6"
REQUIRE_RESULTS = True


# =============================================================================
# REQUIRED ENVIRONMENT VARIABLES
# =============================================================================
# LOINC Search API access requires:
#   export LOINC_USERNAME="..."
#   export LOINC_PASSWORD="..."


def main() -> None:
    if (
        not os.environ.get("LOINC_USERNAME")
        or not os.environ.get("LOINC_PASSWORD")
    ):
        raise RuntimeError(
            "LOINC_USERNAME and LOINC_PASSWORD are required for LOINC smoke tests."
        )

    print(json.dumps({
        "integration": "LOINC Search API",
        "query": QUERY,
        "top_k": TOP_K,
        "expected_code": EXPECT_CODE or None,
    }, indent=2))

    # Actual live SearchTools call.
    results = SearchTools(api_timeout=15, request_delay=0.1).search_loinc(
        QUERY,
        top_k=TOP_K,
    )
    print(json.dumps(results, indent=2, default=str))

    # Output validation and smoke checks.
    if REQUIRE_RESULTS:
        assert results, "LOINC search returned no candidates"
    if results:
        assert any(result.get("ontology") == "LOINC" for result in results), (
            "No returned candidate has ontology='LOINC'"
        )
    if EXPECT_CODE:
        assert any(result.get("code") == EXPECT_CODE for result in results), (
            f"Expected LOINC code {EXPECT_CODE!r} was not returned"
        )
    print("PASS")


if __name__ == "__main__":
    main()
