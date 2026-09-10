"""
llm-ontology-mapper
~~~~~~~~~~~~~~~~~~~
LLM-powered ontology mapping via the seven-stage planned pipeline
(QueryPlanner -> RetrievalRouter -> retriever -> CandidateNormalizer ->
CandidateMerger -> LLMReranker -> MappingResultBuilder). This is the only
supported mapping architecture.

Public API surface
──────────────────
Import only from this module.  Internal submodules are subject to change.

Quick start::

    from llm_ontology_mapper import OntologyMapper, MappingResult

    mapper = OntologyMapper(provider="openai", model="gpt-4o")
    result: MappingResult = mapper.map_term("cough", source_label="Do you have a cough?")

    print(result.target_code)    # 'HP:0012735'
    print(result.confidence)     # 0.93
    print(result.logic_type)     # LogicType.RAG
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

# ── Phase 4B — candidate merger ───────────────────────────────────────────────
from .candidate_merger import CandidateMergeError, CandidateMerger

# ── Phase 4A — candidate normalizer ───────────────────────────────────────────
from .candidate_normalizer import CandidateNormalizationError, CandidateNormalizer

# ── Phase 6 — disabled LLM-only mapping path ──────────────────────────────────
from .disabled_mapping import DisabledMappingError, DisabledMappingRunner

# ── Phase 5A — grounded LLM reranker ──────────────────────────────────────────
from .llm_reranker import LLMReranker, LLMRerankerError

# ── Phase 8 — local semantic retriever wrapper ────────────────────────────────
from .local_retriever import LocalRetrievalError, LocalSemanticRetriever, SapBERTClient

# ── Core mapper (imported here so users don't need to know the submodule) ──────
from .mapper import OntologyMapper

# ── Phase 5B — mapping result builder ─────────────────────────────────────────
from .mapping_result_builder import MappingResultBuilder, MappingResultBuilderError
from .models import (
    AlternativeMapping,
    GroundingSource,
    LogicType,
    MappingBatch,
    MappingMetadata,
    MappingResult,
    NormalizedCandidate,
    OntologyPrefix,
    QueryPlan,
    RAGDebugInfo,
    RerankAlternative,
    RerankDecision,
    RetrievalMode,
    RetrievalRoutePlan,
    RetrievalTrace,
)

# ── NER extractor (optional — pulls in scispacy when used) ─────────────────────
from .ner_extractor import NERQueryExtractor

# ── Phase 9 — planned pipeline orchestrator ──────────────────────────────────
from .planned_pipeline import PlannedPipeline, PlannedPipelineError

# ── Provider layer (lazy — SDK imports happen inside the classes) ──────────────
from .providers import (
    AnthropicProvider,
    BaseLLMProvider,
    ChatMessage,
    CompletionResponse,
    LLMProviderFactory,
    OllamaProvider,
    OpenAIProvider,
)

# ── Phase 7 — public ontology retriever wrapper ────────────────────────────────
from .public_retriever import PublicOntologyRetriever, PublicRetrievalError

# ── Phase 2 — LLM-assisted query planner ──────────────────────────────────────
from .query_planner import QueryPlanner, QueryPlanningError

# ── Phase 3 — retrieval router ────────────────────────────────────────────────
from .retrieval_router import RetrievalRouter

# ── Public retrieval-source adapters/registry (extensibility CASE 1 / CASE 2) ──
from .retrieval_sources import (
    RetrievalConfigError,
    RetrievalSource,
    register_source,
    unregister_source,
)

# ── Validator (optional — needs requests for live API calls) ───────────────────
from .validator import OntologyValidator

# ── Version ───────────────────────────────────────────────────────────────────
try:
    __version__: str = version("llm-ontology-mapper")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # running from source without install

# ── Optional public exports ───────────────────────────────────────────────────
# Evaluator symbols need pandas, which belongs to the optional 'eval' extra.
# Keep them available through package-level imports without making every import
# of llm_ontology_mapper require pandas.
_EVALUATOR_EXPORTS = {
    "EvaluationDetail",
    "EvaluationMetrics",
    "EvaluationReport",
    "MatchType",
    "OntologyBreakdown",
    "OntologyMappingEvaluator",
}


def __getattr__(name: str) -> Any:
    """Lazily expose optional evaluator exports."""
    if name in _EVALUATOR_EXPORTS:
        try:
            module = import_module(".evaluator", __name__)
        except ModuleNotFoundError as exc:
            if exc.name == "pandas":
                raise ModuleNotFoundError(
                    "Evaluator exports require pandas. Install the eval extra with "
                    "`uv sync --extra eval` or `pip install 'llm-ontology-mapper[eval]'`."
                ) from exc
            raise
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EVALUATOR_EXPORTS)

__all__ = [
    # Version
    "__version__",
    # Data models
    "MappingResult",
    "MappingBatch",
    "MappingMetadata",
    "AlternativeMapping",
    "RAGDebugInfo",
    "LogicType",
    "OntologyPrefix",
    # Provider layer
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LLMProviderFactory",
    "ChatMessage",
    "CompletionResponse",
    # Core services
    "OntologyMapper",
    "OntologyValidator",
    # Evaluator
    "OntologyMappingEvaluator",
    "MatchType",
    "EvaluationDetail",
    "OntologyBreakdown",
    "EvaluationMetrics",
    "EvaluationReport",
    "NERQueryExtractor",
    # Phase 1 pipeline models
    "RetrievalMode",
    "GroundingSource",
    "QueryPlan",
    "NormalizedCandidate",
    "RetrievalTrace",
    "RerankAlternative",
    "RerankDecision",
    # Phase 2 — LLM-assisted query planner
    "QueryPlanner",
    "QueryPlanningError",
    # Phase 3 — retrieval router
    "RetrievalRouter",
    "RetrievalRoutePlan",
    # Phase 4A — candidate normalizer
    "CandidateNormalizer",
    "CandidateNormalizationError",
    # Phase 4B — candidate merger
    "CandidateMerger",
    "CandidateMergeError",
    # Phase 5A — grounded LLM reranker
    "LLMReranker",
    "LLMRerankerError",
    # Phase 5B — mapping result builder
    "MappingResultBuilder",
    "MappingResultBuilderError",
    # Phase 6 — disabled LLM-only mapping path
    "DisabledMappingRunner",
    "DisabledMappingError",
    # Phase 7 — public ontology retriever wrapper
    "PublicOntologyRetriever",
    "PublicRetrievalError",
    # Public retrieval-source adapters/registry
    "RetrievalSource",
    "RetrievalConfigError",
    "register_source",
    "unregister_source",
    # Phase 8 — local semantic retriever wrapper
    "LocalSemanticRetriever",
    "LocalRetrievalError",
    "SapBERTClient",
    # Phase 9 — planned pipeline orchestrator
    "PlannedPipeline",
    "PlannedPipelineError",
]
