"""
llm-ontology-mapper
~~~~~~~~~~~~~~~~~~~
LLM-powered ontology mapping with RAG grounding and multi-provider support.

Public API surface
──────────────────
Import only from this module.  Internal submodules are subject to change.

Quick start::

    from llm_ontology_mapper import OntologyMapper, MappingResult

    mapper = OntologyMapper(provider="openai", model="gpt-4o", use_rag=True)
    result: MappingResult = mapper.map_term("cough", source_label="Do you have a cough?")

    print(result.curie)          # 'HP:0012735'
    print(result.confidence)     # 0.93
    print(result.logic_type)     # LogicType.RAG
"""

from importlib.metadata import PackageNotFoundError, version

# ── Version ───────────────────────────────────────────────────────────────────
try:
    __version__: str = version("llm-ontology-mapper")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # running from source without install

# ── Public data contracts (always importable — no heavy deps) ─────────────────
# ── Evaluator (needs pandas; install the 'eval' extra) ─────────────────────────
from .evaluator import (
    EvaluationDetail,
    EvaluationMetrics,
    EvaluationReport,
    MatchType,
    OntologyBreakdown,
    OntologyMappingEvaluator,
)

# ── Core mapper (imported here so users don't need to know the submodule) ──────
from .mapper import OntologyMapper
from .models import (
    AlternativeMapping,
    LogicType,
    MappingBatch,
    MappingMetadata,
    MappingResult,
    OntologyPrefix,
    RAGDebugInfo,
)

# ── NER extractor (optional — pulls in scispacy when used) ─────────────────────
from .ner_extractor import NERQueryExtractor

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

# ── Retriever (optional — only materialises HTTP dep when used) ────────────────
from .retriever import OntologyRetriever

# ── Agentic engine ────────────────────────────────────────────────────────────
from .agentic_mapper import AgenticMapper, normalize_code

# ── Validator (optional — needs requests for live API calls) ───────────────────
from .validator import OntologyValidator

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
    "OntologyRetriever",
    "OntologyValidator",
    # Evaluator
    "OntologyMappingEvaluator",
    "MatchType",
    "EvaluationDetail",
    "OntologyBreakdown",
    "EvaluationMetrics",
    "EvaluationReport",
    "NERQueryExtractor",
    # Agentic engine
    "AgenticMapper",
    "normalize_code",
]

