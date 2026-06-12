"""
Public data contracts for llm-ontology-mapper.

All classes here are pure Pydantic models — zero dependency on LLM providers,
retriever HTTP clients, or PCGL-specific logic.  Any downstream consumer
(PCGL Data Mapper, a REST API, a Jupyter notebook) can import and use these
schemas without pulling in the heavier optional dependencies.

Schema design principles:
  • Immutable by default (model_config frozen=True on leaf result types)
  • Explicit Literal types instead of bare strings for categorical fields
  • All optional fields carry sensible defaults (never None in unexpected places)
  • Serialises losslessly to JSON via model.model_dump(mode="json")
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class LogicType(str, Enum):
    """How the final mapping code was determined."""

    # LLM generated a code directly (no retrieval grounding)
    LLM = "llm"

    # A RAG candidate was selected; LLM validated the choice
    RAG = "rag"

    # RAG returned a high-confidence hit that was auto-accepted (no LLM call)
    DIRECT = "direct"

    # Hybrid: RAG shortlisted candidates; LLM re-ranked / reconciled
    HYBRID = "hybrid"

    # Agentic: tool-calling agent loop drove the mapping
    AGENTIC = "agentic"


class OntologyPrefix(str, Enum):
    """Supported ontology namespaces."""

    HPO    = "HPO"      # Human Phenotype Ontology  → HP:XXXXXXX
    MONDO  = "MONDO"    # Mondo Disease Ontology     → MONDO:XXXXXXX
    NCIT   = "NCIT"     # NCI Thesaurus              → NCIT:CXXXXXX
    LOINC  = "LOINC"    # LOINC                      → bare numeric code
    UO     = "UO"       # Units of Measurement       → UO:XXXXXXX
    ICD10  = "ICD10"    # ICD-10-CM                  → alpha-numeric
    CHEBI  = "CHEBI"    # Chemical Entities          → CHEBI:XXXXXXX
    SNOMED = "SNOMED"   # SNOMED CT                  → numeric SCTID
    RXNORM = "RxNorm"   # RxNorm                     → numeric RxCUI
    OTHER  = "OTHER"    # Any ontology not in the list above


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────


class AlternativeMapping(BaseModel):
    """A runner-up ontology code suggested by the LLM or retriever."""

    code: str = Field(..., description="Ontology CURIE or bare code, e.g. 'HP:0002110'")
    term: str = Field(..., description="Human-readable label for the code")
    ontology: str = Field(..., description="Ontology prefix (HPO, MONDO, …)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score in [0, 1]"
    )
    source: Literal["llm", "rag", "direct", "agentic"] = Field(
        "llm",
        description="How this alternative was generated"
    )

    model_config = {"frozen": True}


class RAGDebugInfo(BaseModel):
    """
    Diagnostic payload attached when RAG is active.

    Intentionally verbose — strip this field before storing in production
    databases or returning to end-users.
    """

    query_sent: str = Field(..., description="Exact query string forwarded to the retriever")
    candidates_retrieved: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw candidate list returned by the retriever (code, term, score)"
    )
    top_k: int = Field(..., description="Number of candidates requested")
    auto_accepted: bool = Field(
        False,
        description="True when the top candidate exceeded auto-accept threshold"
    )
    auto_accept_threshold: float = Field(0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class MappingMetadata(BaseModel):
    """
    Provenance and runtime metadata attached to every MappingResult.

    Designed for audit trails and reproducibility.
    """

    model: str = Field(..., description="LLM model identifier used (e.g. 'gpt-4o')")
    provider: str = Field(..., description="LLM provider (openai / anthropic / ollama / github / azure)")
    latency_ms: float | None = Field(None, description="Wall-clock time for the mapping call in ms")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the mapping was produced"
    )
    prompt_tokens: int | None = Field(None, description="Prompt token count (when available)")
    completion_tokens: int | None = Field(None, description="Completion token count (when available)")
    rag_debug: RAGDebugInfo | None = Field(
        None,
        description="RAG diagnostic info; None when RAG was not used"
    )

    model_config = {"frozen": True}


# ─────────────────────────────────────────────────────────────────────────────
# Primary public contract
# ─────────────────────────────────────────────────────────────────────────────


class MappingResult(BaseModel):
    """
    Canonical output of a single ontology mapping operation.

    This is the **only** type that crosses the package boundary.
    Every consumer (PCGL Data Mapper, REST endpoint, evaluation harness)
    must be able to work with this model without importing anything else
    from llm_ontology_mapper.

    Field alignment with legacy OntologyMapping dataclass
    ──────────────────────────────────────────────────────
      OntologyMapping.source_field   → MappingResult.source_term
      OntologyMapping.ontology_code  → MappingResult.target_code
      OntologyMapping.ontology_term  → MappingResult.target_term
      OntologyMapping.ontology_source→ MappingResult.ontology
      OntologyMapping.confidence     → MappingResult.confidence
      (new)                          → MappingResult.logic_type
    """

    # ── Input side ────────────────────────────────────────────────────────────

    source_term: str = Field(
        ...,
        description="Original field name or clinical label from the source data dictionary"
    )
    source_label: str | None = Field(
        None,
        description="Human-readable label / question text for the source field"
    )
    source_type: str | None = Field(
        None,
        description="Data type hint from the source schema (radio, text, integer, …)"
    )

    # ── Output side ───────────────────────────────────────────────────────────

    target_code: str = Field(
        ...,
        description="Ontology CURIE, e.g. 'HP:0002110', 'MONDO:0005180'. Bare codes are auto-normalised to PREFIX:code."
    )
    target_term: str = Field(
        ...,
        description="Official label for the mapped ontology code"
    )
    ontology: str = Field(
        ...,
        description="Ontology namespace (HPO, MONDO, NCIT, LOINC, …)"
    )

    # ── Quality indicators ────────────────────────────────────────────────────

    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Aggregate confidence score in [0, 1]"
    )
    logic_type: LogicType = Field(
        ...,
        description="Strategy that produced this mapping (llm / rag / direct / hybrid)"
    )

    # ── Supporting detail ─────────────────────────────────────────────────────

    alternatives: list[AlternativeMapping] = Field(
        default_factory=list,
        description="Ordered list of runner-up mappings (descending confidence)"
    )
    notes: str | None = Field(
        None,
        description="Free-text notes from the LLM (caveats, ambiguity warnings, …)"
    )
    metadata: MappingMetadata | None = Field(
        None,
        description="Provenance — model, provider, latency, RAG debug info"
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("target_code")
    @classmethod
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target_code must not be blank")
        return v.strip()

    @field_validator("ontology")
    @classmethod
    def normalise_ontology_prefix(cls, v: str) -> str:
        return v.upper().strip()

    @model_validator(mode="after")
    def normalise_and_sort(self) -> MappingResult:
        """Normalise target_code to CURIE and sort alternatives by confidence."""
        if ":" not in self.target_code:
            self.target_code = f"{self.ontology}:{self.target_code}"
        self.alternatives.sort(key=lambda a: a.confidence, reverse=True)
        return self

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def is_high_confidence(self) -> bool:
        """Convenience flag: confidence >= 0.8."""
        return self.confidence >= 0.8

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Serialise to the flat dict format used by the legacy OntologyMapping.to_dict().

        Use this in the bridge adapter (core/llm/_bridge.py) so that existing
        PCGL mapper code that unpacks these dicts continues to work unchanged.
        """
        return {
            "source_field":        self.source_term,
            "source_label":        self.source_label,
            "source_type":         self.source_type,
            "code":                self.target_code,
            "term":                self.target_term,
            "ontology":            self.ontology,
            "confidence":          self.confidence,
            "alternatives":        [a.model_dump() for a in self.alternatives],
            "notes":               self.notes,
        }

    model_config = {
        "frozen": False,          # allow metadata to be attached after construction
        "populate_by_name": True, # accept both alias and field name during construction
        "json_schema_extra": {
            "examples": [
                {
                    "source_term":  "cough",
                    "source_label": "Do you have a cough?",
                    "source_type":  "radio",
                    "target_code":  "HP:0012735",
                    "target_term":  "Cough",
                    "ontology":     "HPO",
                    "confidence":   0.92,
                    "logic_type":   "rag",
                    "alternatives": [],
                    "notes":        None,
                    "metadata":     None,
                }
            ]
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch container
# ─────────────────────────────────────────────────────────────────────────────


class MappingBatch(BaseModel):
    """Container for a set of MappingResult objects from a data dictionary run."""

    study_id: str | None = Field(None, description="Source study identifier")
    entity_type: str | None = Field(None, description="Entity type (phenotype, diagnosis, …)")
    results: list[MappingResult] = Field(default_factory=list)
    produced_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def high_confidence(self) -> list[MappingResult]:
        return [r for r in self.results if r.is_high_confidence]

    @property
    def needs_review(self) -> list[MappingResult]:
        return [r for r in self.results if not r.is_high_confidence]

    def to_csv_records(self) -> list[dict[str, Any]]:
        """Flat list of dicts suitable for pandas.DataFrame() or csv.DictWriter."""
        return [r.to_legacy_dict() for r in self.results]
