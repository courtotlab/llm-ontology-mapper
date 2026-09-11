"""
Unit tests for tests/live/_planned_smoke_helpers.py's print_trace_summary.

Locks down the presentation fix for the "outer vs nested" is_grounded /
grounding_source ambiguity: when public retrieval returns candidates but the
LLM reranker abstains (an UNKNOWN:UNMAPPED result), the two concepts must be
printed under distinct, unambiguous keys rather than the same bare
"is_grounded"/"grounding_source" name used for both the result-level and
retrieval-level values.

No network access; result objects are plain SimpleNamespace fakes shaped
like the real MappingResult/MappingMetadata/RAGDebugInfo attribute access
print_trace_summary relies on (getattr-based, duck-typed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_helpers() -> Any:
    live_dir = Path(__file__).parent / "live"
    sys.path.insert(0, str(live_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "_planned_smoke_helpers", live_dir / "_planned_smoke_helpers.py"
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(live_dir))


helpers = _load_helpers()


def _fake_result(pipeline_info: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            rag_debug=SimpleNamespace(candidates_retrieved=[pipeline_info])
        )
    )


def _abstention_pipeline_info() -> dict[str, Any]:
    """Shape mirrors MappingResultBuilder._build_metadata's pipeline_info for
    the metformin-style scenario: retrieval succeeded (6 candidates,
    RetrievalTrace.is_grounded=True, grounding_source=public_api) but the
    reranker abstained (RerankDecision.is_grounded=False,
    grounding_source=none)."""
    return {
        "retrieval_mode": "public",
        "is_grounded": False,
        "grounding_source": "none",
        "policy": "production_grounded",
        "candidate_count": 6,
        "query_plan": {"normalized_term": "metformin"},
        "retrieval_trace": {
            "is_grounded": True,
            "grounding_source": "public_api",
            "route_calls": [{"route": "public_api"}] * 3,
            "raw_candidate_count": 6,
            "merged_candidate_count": 6,
            "selected_candidate_code": None,
            "errors": [],
            "query_plan": {"normalized_term": "metformin"},
        },
    }


def test_print_trace_summary_labels_result_and_retrieval_grounding_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _fake_result(_abstention_pipeline_info())

    helpers.print_trace_summary(result)

    printed = json.loads(capsys.readouterr().out)
    summary = printed["trace_summary"]

    # The two concepts disagree here (retrieval succeeded, reranker
    # abstained) and must be printed under distinct keys, not one shared
    # "is_grounded"/"grounding_source" pair.
    assert summary["result_is_grounded"] is False
    assert summary["result_grounding_source"] == "none"
    assert summary["retrieval_is_grounded"] is True
    assert summary["retrieval_grounding_source"] == "public_api"

    # The old ambiguous bare keys must not reappear.
    assert "is_grounded" not in summary
    assert "grounding_source" not in summary

    # Other retrieval-trace fields are still surfaced.
    assert summary["route_call_count"] == 3
    assert summary["raw_candidate_count"] == 6
    assert summary["merged_candidate_count"] == 6
    assert summary["selected_candidate_code"] is None
    assert summary["errors"] == []


def test_print_trace_summary_agrees_when_result_is_selected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Contrast case: when a candidate is actually selected, both
    representations agree (both True / both public_api)."""
    info = _abstention_pipeline_info()
    info["is_grounded"] = True
    info["grounding_source"] = "public_api"
    info["retrieval_trace"]["selected_candidate_code"] = "LOINC:6809-2"
    result = _fake_result(info)

    helpers.print_trace_summary(result)

    summary = json.loads(capsys.readouterr().out)["trace_summary"]
    assert summary["result_is_grounded"] is True
    assert summary["result_grounding_source"] == "public_api"
    assert summary["retrieval_is_grounded"] is True
    assert summary["retrieval_grounding_source"] == "public_api"
    assert summary["selected_candidate_code"] == "LOINC:6809-2"


def test_print_trace_summary_no_pipeline_metadata_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(metadata=None)

    helpers.print_trace_summary(result)

    assert capsys.readouterr().out == ""
