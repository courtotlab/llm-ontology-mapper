"""
Unit tests for llm_ontology_mapper.benchmarking.model_registry.

Run with:  pytest tests/benchmarking/test_model_registry.py -v -m unit
"""

from __future__ import annotations

import pytest

from llm_ontology_mapper.benchmarking.model_registry import ALLOWED_MODELS, get_model_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.4-mini", "gpt-5-mini"])
def test_supported_gpt5_models_request_low_reasoning(model: str) -> None:
    cfg = get_model_config(model)
    assert cfg.is_reasoning is True
    assert cfg.reasoning_effort == "low"


def test_gpt_41_mini_omits_reasoning_effort() -> None:
    cfg = get_model_config("gpt-4.1-mini")
    assert cfg.is_reasoning is False
    assert cfg.reasoning_effort is None


def test_unsupported_model_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark model"):
        get_model_config("gpt-3.5-turbo")


def test_registry_has_exactly_the_four_reviewed_models() -> None:
    assert set(ALLOWED_MODELS) == {"gpt-5.6-luna", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini"}
