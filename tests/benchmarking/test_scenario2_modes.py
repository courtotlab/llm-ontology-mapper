"""
Scenario 2 mode-forwarding tests (Part 30, items 8-11).

No network, no real LLM/OpenAI/SapBERT calls -- PlannedPipeline and its
retrievers/OntologyMapper are patched so we can inspect exactly what
build_pipeline_and_mappers() constructs for each retrieval_mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.benchmarking.model_registry import get_model_config
from llm_ontology_mapper.benchmarking.scenario2_runner import (
    Scenario2RunConfig,
    build_pipeline_and_mappers,
)

pytestmark = pytest.mark.unit


def _run_config(mode: str, **overrides) -> Scenario2RunConfig:
    defaults = dict(
        model_config=get_model_config("gpt-5.6-luna"),
        retrieval_mode=mode,
        temperature=None,
        seed=42,
        max_alternatives=4,
    )
    if mode == "local":
        defaults["sapbert_url"] = "http://localhost:8765"
    defaults.update(overrides)
    return Scenario2RunConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 8. public mode forwarded
# ─────────────────────────────────────────────────────────────────────────────


def test_public_mode_constructs_public_retriever_and_no_local_retriever() -> None:
    with (
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PublicOntologyRetriever") as mock_public,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.LocalSemanticRetriever") as mock_local,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PlannedPipeline") as mock_pipeline,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.OntologyMapper") as mock_mapper_cls,
    ):
        provider = MagicMock()
        run_config = _run_config("public")
        build_pipeline_and_mappers(provider=provider, run_config=run_config, target_ontologies=["HPO"])

        mock_public.assert_called_once()
        mock_local.assert_not_called()
        _, pipeline_kwargs = mock_pipeline.call_args
        assert pipeline_kwargs["public_retriever"] is mock_public.return_value
        assert "local_retriever" not in pipeline_kwargs
        _, mapper_kwargs = mock_mapper_cls.call_args
        assert mapper_kwargs["retrieval_mode"] == "public"


# ─────────────────────────────────────────────────────────────────────────────
# 9. local mode forwarded
# ─────────────────────────────────────────────────────────────────────────────


def test_local_mode_constructs_local_retriever_with_sapbert_url_and_no_public_retriever() -> None:
    with (
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PublicOntologyRetriever") as mock_public,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.LocalSemanticRetriever") as mock_local,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PlannedPipeline") as mock_pipeline,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.OntologyMapper") as mock_mapper_cls,
    ):
        provider = MagicMock()
        run_config = _run_config("local", sapbert_url="http://localhost:9999")
        build_pipeline_and_mappers(provider=provider, run_config=run_config, target_ontologies=["HPO"])

        mock_local.assert_called_once_with(sapbert_url="http://localhost:9999")
        mock_public.assert_not_called()
        _, pipeline_kwargs = mock_pipeline.call_args
        assert pipeline_kwargs["local_retriever"] is mock_local.return_value
        assert "public_retriever" not in pipeline_kwargs
        _, mapper_kwargs = mock_mapper_cls.call_args
        assert mapper_kwargs["retrieval_mode"] == "local"


def test_local_mode_requires_sapbert_url() -> None:
    with pytest.raises(ValueError, match="sapbert_url"):
        Scenario2RunConfig(model_config=get_model_config("gpt-5.6-luna"), retrieval_mode="local", sapbert_url=None)


# ─────────────────────────────────────────────────────────────────────────────
# 10. disabled mode forwarded
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_mode_constructs_neither_retriever() -> None:
    with (
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PublicOntologyRetriever") as mock_public,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.LocalSemanticRetriever") as mock_local,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PlannedPipeline") as mock_pipeline,
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.OntologyMapper") as mock_mapper_cls,
    ):
        provider = MagicMock()
        run_config = _run_config("disabled")
        build_pipeline_and_mappers(provider=provider, run_config=run_config, target_ontologies=["HPO"])

        mock_public.assert_not_called()
        mock_local.assert_not_called()
        _, pipeline_kwargs = mock_pipeline.call_args
        assert "public_retriever" not in pipeline_kwargs
        assert "local_retriever" not in pipeline_kwargs
        _, mapper_kwargs = mock_mapper_cls.call_args
        assert mapper_kwargs["retrieval_mode"] == "disabled"


def test_invalid_retrieval_mode_rejected() -> None:
    with pytest.raises(ValueError, match="retrieval_mode"):
        Scenario2RunConfig(model_config=get_model_config("gpt-5.6-luna"), retrieval_mode="both")


# ─────────────────────────────────────────────────────────────────────────────
# 11. all non-mode config identical across the three modes
# ─────────────────────────────────────────────────────────────────────────────


def test_llm_call_config_identical_across_modes_except_mode_itself() -> None:
    public_cfg = _run_config("public")
    local_cfg = _run_config("local")
    disabled_cfg = _run_config("disabled")

    public_llm = public_cfg.to_llm_call_config()
    local_llm = local_cfg.to_llm_call_config()
    disabled_llm = disabled_cfg.to_llm_call_config()

    assert public_llm == local_llm == disabled_llm

    for cfg in (public_cfg, local_cfg, disabled_cfg):
        assert cfg.model_config.model == "gpt-5.6-luna"
        assert cfg.model_config.reasoning_effort == "low"
        assert cfg.temperature is None
        assert cfg.seed == 42
        assert cfg.max_alternatives == 4


def test_mapper_construction_identical_except_retrieval_mode() -> None:
    with (
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PublicOntologyRetriever"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.LocalSemanticRetriever"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.PlannedPipeline"),
        patch("llm_ontology_mapper.benchmarking.scenario2_runner.OntologyMapper") as mock_mapper_cls,
    ):
        provider = MagicMock()
        for mode in ("public", "local", "disabled"):
            mock_mapper_cls.reset_mock()
            run_config = _run_config(mode)
            build_pipeline_and_mappers(provider=provider, run_config=run_config, target_ontologies=["HPO"])
            _, kwargs = mock_mapper_cls.call_args
            assert kwargs["ontologies"] == ["HPO"]
            assert kwargs["use_planned_pipeline"] is True
            assert kwargs["max_alternatives"] == 4
            assert kwargs["rag_top_k"] == run_config.max_results_per_query
            assert kwargs["max_candidates"] == run_config.max_candidates
