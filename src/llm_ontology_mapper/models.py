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

from llm_ontology_mapper.ontology_identity import (
    canonical_ontology,
    normalize_code_for_ontology,
    validate_candidate_identity,
)

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

    HPO = "HPO"  # Human Phenotype Ontology  → HP:XXXXXXX
    MONDO = "MONDO"  # Mondo Disease Ontology     → MONDO:XXXXXXX
    NCIT = "NCIT"  # NCI Thesaurus              → NCIT:CXXXXXX
    LOINC = "LOINC"  # LOINC                      → bare numeric code
    UO = "UO"  # Units of Measurement       → UO:XXXXXXX
    ICD10 = "ICD10"  # ICD-10-CM                  → alpha-numeric
    CHEBI = "CHEBI"  # Chemical Entities          → CHEBI:XXXXXXX
    SNOMED = "SNOMED"  # SNOMED CT                  → numeric SCTID
    RXNORM = "RxNorm"  # RxNorm                     → numeric RxCUI
    OTHER = "OTHER"  # Any ontology not in the list above


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────


class AlternativeMapping(BaseModel):
    """A runner-up ontology code suggested by the LLM or retriever."""

    code: str = Field(..., description="Ontology CURIE or bare code, e.g. 'HP:0002110'")
    term: str = Field(..., description="Human-readable label for the code")
    ontology: str = Field(..., description="Ontology prefix (HPO, MONDO, …)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    source: Literal["llm", "rag", "direct", "agentic"] = Field(
        "llm", description="How this alternative was generated"
    )
    explanation: str | None = Field(
        None,
        description="Plain-language explanation of when this alternative may fit",
    )

    @model_validator(mode="after")
    def validate_code_namespace(self) -> AlternativeMapping:
        validation = validate_candidate_identity(ontology=self.ontology, code=self.code)
        if not validation.valid:
            raise ValueError(validation.reason or "alternative ontology/code mismatch")
        return self

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
        description="Raw candidate list returned by the retriever (code, term, score)",
    )
    top_k: int = Field(..., description="Number of candidates requested")
    auto_accepted: bool = Field(
        False, description="True when the top candidate exceeded auto-accept threshold"
    )
    auto_accept_threshold: float = Field(0.0, ge=0.0, le=1.0)
    pipeline_timings: dict[str, float] | None = Field(
        default=None,
        description="Major planned-pipeline stage timings in milliseconds, when available",
    )
    pipeline_usage: dict[str, Any] | None = Field(
        default=None,
        description=(
            "LLM token usage per planned-pipeline stage (query_planner, "
            "llm_reranker), when available"
        ),
    )
    retrieval_diagnostics: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Aggregate public-retrieval retry/error telemetry for this mapping "
            "(retrieval_request_count, retrieval_retry_count, "
            "retrieval_recovered_error_count, retrieval_final_error_count, "
            "retrieval_error_sources, retrieval_error_types), when available. "
            "Diagnostic metadata only -- never affects scoring or mapped_status."
        ),
    )

    model_config = {"frozen": True}


class MappingMetadata(BaseModel):
    """
    Provenance and runtime metadata attached to every MappingResult.

    Designed for audit trails and reproducibility.
    """

    model: str = Field(..., description="LLM model identifier used (e.g. 'gpt-4o')")
    provider: str = Field(
        ..., description="LLM provider (openai / anthropic / ollama / github / azure)"
    )
    latency_ms: float | None = Field(None, description="Wall-clock time for the mapping call in ms")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the mapping was produced",
    )
    prompt_tokens: int | None = Field(None, description="Prompt token count (when available)")
    completion_tokens: int | None = Field(
        None, description="Completion token count (when available)"
    )
    rag_debug: RAGDebugInfo | None = Field(
        None, description="RAG diagnostic info; None when RAG was not used"
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
        ..., description="Original field name or clinical label from the source data dictionary"
    )
    source_label: str | None = Field(
        None, description="Human-readable label / question text for the source field"
    )
    source_type: str | None = Field(
        None, description="Data type hint from the source schema (radio, text, integer, …)"
    )

    # ── Output side ───────────────────────────────────────────────────────────

    target_code: str = Field(
        ...,
        description="Ontology CURIE, e.g. 'HP:0002110', 'MONDO:0005180'. Bare codes are auto-normalised to PREFIX:code.",
    )
    target_term: str = Field(..., description="Official label for the mapped ontology code")
    ontology: str = Field(..., description="Ontology namespace (HPO, MONDO, NCIT, LOINC, …)")

    # ── Quality indicators ────────────────────────────────────────────────────

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate confidence score in [0, 1]"
    )
    logic_type: LogicType = Field(
        ..., description="Strategy that produced this mapping (llm / rag / direct / hybrid)"
    )

    # ── Supporting detail ─────────────────────────────────────────────────────

    alternatives: list[AlternativeMapping] = Field(
        default_factory=list,
        description="Ordered list of runner-up mappings (descending confidence)",
    )
    notes: str | None = Field(
        None, description="Free-text notes from the LLM (caveats, ambiguity warnings, …)"
    )
    metadata: MappingMetadata | None = Field(
        None, description="Provenance — model, provider, latency, RAG debug info"
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
            self.target_code = normalize_code_for_ontology(self.target_code, self.ontology)
        validation = validate_candidate_identity(ontology=self.ontology, code=self.target_code)
        if not validation.valid:
            raise ValueError(validation.reason or "target ontology/code mismatch")
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
            "source_field": self.source_term,
            "source_label": self.source_label,
            "source_type": self.source_type,
            "code": self.target_code,
            "term": self.target_term,
            "ontology": self.ontology,
            "confidence": self.confidence,
            "alternatives": [a.model_dump() for a in self.alternatives],
            "notes": self.notes,
        }

    model_config = {
        "frozen": False,  # allow metadata to be attached after construction
        "populate_by_name": True,  # accept both alias and field name during construction
        "json_schema_extra": {
            "examples": [
                {
                    "source_term": "cough",
                    "source_label": "Do you have a cough?",
                    "source_type": "radio",
                    "target_code": "HP:0012735",
                    "target_term": "Cough",
                    "ontology": "HPO",
                    "confidence": 0.92,
                    "logic_type": "rag",
                    "alternatives": [],
                    "notes": None,
                    "metadata": None,
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
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def high_confidence(self) -> list[MappingResult]:
        return [r for r in self.results if r.is_high_confidence]

    @property
    def needs_review(self) -> list[MappingResult]:
        return [r for r in self.results if not r.is_high_confidence]

    def to_csv_records(self) -> list[dict[str, Any]]:
        """Flat list of dicts suitable for pandas.DataFrame() or csv.DictWriter."""
        return [r.to_legacy_dict() for r in self.results]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 pipeline enumerations  (planned architecture contracts — no runtime
# behaviour change; these define future pipeline data contracts only)
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalMode(str, Enum):
    """User-facing Layer 2 retrieval mode.

    Exactly three values are supported; there is no user-facing 'both' mode.
    """

    PUBLIC = "public"  # public ontology database retrieval
    LOCAL = "local"  # local semantic retrieval (e.g. SapBERT / FAISS)
    DISABLED = "disabled"  # no retrieval; LLM-only / ungrounded


class GroundingSource(str, Enum):
    """Where retrieval grounding candidates originated."""

    PUBLIC_API = "public_api"  # public ontology APIs (OLS, LOINC, RxNav, …)
    LOCAL_SAPBERT = "local_sapbert"  # local SapBERT / FAISS index
    NONE = "none"  # no grounding (disabled mode or unmapped)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 pipeline models  (planned architecture contracts)
# ─────────────────────────────────────────────────────────────────────────────


class QueryPlan(BaseModel):
    """Planned interpretation of a source term before retrieval or LLM-only mapping."""

    original_term: str = Field(..., description="Raw input term, e.g. 'sys_bp'")
    original_label: str | None = Field(None, description="Optional human-readable label")
    source_description: str | None = Field(
        None,
        description="Optional caller-supplied description of the source field",
    )
    source_type: str | None = Field(
        None,
        description="Optional source schema/data type hint such as integer, decimal, or text",
    )
    normalized_term: str | None = Field(None, description="Basic normalized form of the term")
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Queries to send to retrieval, e.g. ['systolic blood pressure']",
    )
    inferred_meaning: str | None = Field(None, description="Plain-language meaning of the field")
    semantic_type: str | None = Field(
        None,
        description="Type such as measurement, phenotype, diagnosis, drug, procedure",
    )
    candidate_ontologies: list[str] = Field(
        default_factory=list,
        description="Candidate ontology families inferred by the planner",
    )
    preferred_ontology: str | None = Field(
        None,
        description="Preferred ontology when planner confidence is high",
    )
    allowed_target_ontologies: list[str] | None = Field(
        None,
        description="Caller-selected ontology allow-list; None means unrestricted",
    )
    retrieval_mode: RetrievalMode = Field(
        RetrievalMode.PUBLIC,
        description="User-facing Layer 2 mode: public, local, or disabled",
    )
    target_ontology_constraint: str | None = Field(
        None,
        description="Hard target-ontology constraint supplied by the caller",
    )
    retrieval_disabled_reason: str | None = Field(
        None,
        description="Why retrieval was disabled, when applicable",
    )
    reasoning: str | None = Field(None, description="Short explanation of the plan")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Planner confidence in the interpretation, in [0, 1]",
    )

    @field_validator("original_term")
    @classmethod
    def original_term_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("original_term must not be blank")
        return v.strip()

    @field_validator("allowed_target_ontologies")
    @classmethod
    def normalise_allowed_target_ontologies(
        cls,
        v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for ontology in v:
            text = str(ontology or "").upper().strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized or None

    # ── Derived routing helpers ───────────────────────────────────────────────

    @property
    def route_public_apis(self) -> bool:
        """True only when retrieval_mode is public."""
        return self.retrieval_mode == RetrievalMode.PUBLIC

    @property
    def route_local(self) -> bool:
        """True only when retrieval_mode is local."""
        return self.retrieval_mode == RetrievalMode.LOCAL

    @property
    def retrieval_disabled(self) -> bool:
        """True only when retrieval_mode is disabled."""
        return self.retrieval_mode == RetrievalMode.DISABLED

    model_config = {"frozen": True}


class NormalizedCandidate(BaseModel):
    """A retrieved candidate from a grounded retrieval mode in a common schema.

    Disabled mode must not produce NormalizedCandidate records; retrieval_mode
    is therefore restricted to public or local.
    """

    code: str = Field(..., description="Canonical CURIE or normalized code")
    term: str = Field(..., description="Official candidate label")
    ontology: str = Field(..., description="Ontology family such as LOINC, HPO, MONDO")
    definition: str | None = Field(None, description="Definition or description if available")
    source: str = Field(..., description="Source system such as OLS, LOINC-Search-API, SapBERT")
    matched_query: str = Field(..., description="Expanded query that produced this candidate")
    retrieval_mode: RetrievalMode = Field(
        ...,
        description="Grounded retrieval mode that produced the candidate; must be public or local",
    )
    raw_score: float | None = Field(
        None,
        description="Source-native score; any float or None (not normalized across sources)",
    )
    normalized_score: float | None = Field(
        None,
        description="Calibrated score in [0, 1] when provided",
    )
    provenance: dict[str, Any] | list[dict[str, Any]] | None = Field(
        None,
        description="Route, endpoint, query, and raw payload references",
    )
    retrieved_from_ontologies: list[str] = Field(
        default_factory=list,
        description=(
            "Ontology search spaces that retrieved this candidate. This is distinct "
            "from ontology, which is the candidate's native code namespace."
        ),
    )

    @field_validator("code", "term", "source", "matched_query")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be blank")
        return v.strip()

    @field_validator("ontology")
    @classmethod
    def normalise_ontology(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ontology must not be blank")
        return v.upper().strip()

    @field_validator("retrieved_from_ontologies")
    @classmethod
    def normalise_retrieved_from_ontologies(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for ontology in v:
            key = canonical_ontology(ontology) or str(ontology or "").upper().strip()
            if key and key not in seen:
                normalized.append(key)
                seen.add(key)
        return normalized

    @field_validator("retrieval_mode")
    @classmethod
    def retrieval_mode_not_disabled(cls, v: RetrievalMode) -> RetrievalMode:
        if v == RetrievalMode.DISABLED:
            raise ValueError(
                "NormalizedCandidate retrieval_mode cannot be disabled; "
                "disabled mode does not produce retrieval candidates"
            )
        return v

    @field_validator("normalized_score")
    @classmethod
    def normalized_score_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("normalized_score must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_code_namespace(self) -> NormalizedCandidate:
        validation = validate_candidate_identity(
            ontology=self.ontology,
            code=self.code,
            iri=(
                self.provenance.get("raw_candidate", {}).get("iri")
                if isinstance(self.provenance, dict)
                and isinstance(self.provenance.get("raw_candidate"), dict)
                else None
            ),
        )
        if not validation.valid:
            raise ValueError(validation.reason or "candidate ontology/code mismatch")
        return self

    model_config = {"frozen": True}


class RerankAlternative(BaseModel):
    """Structured alternative selected by the reranker from retrieved candidates."""

    candidate_id: str = Field(..., description="Candidate id such as C1")
    code: str = Field(..., description="Exact ontology code from the candidate")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Final reranker confidence for this alternative in [0, 1], on the "
            "same scale as RerankDecision.confidence"
        ),
    )
    explanation: str = Field(
        ...,
        description="Why this retrieved candidate could be a good alternative",
    )

    @field_validator("candidate_id", "code", "explanation")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be blank")
        return v.strip()

    model_config = {"frozen": True}


class RetrievalTrace(BaseModel):
    """Evidence trail for retrieval, or explicit record that retrieval was skipped.

    For disabled mode, retrieval_skipped is auto-set to True during construction.
    """

    query_plan: QueryPlan | None = Field(
        None,
        description="Plan used for retrieval or disabled mapping",
    )
    retrieval_mode: RetrievalMode = Field(..., description="User-facing Layer 2 mode")
    is_grounded: bool = Field(
        ...,
        description="True for candidate-grounded modes; false for disabled",
    )
    grounding_source: GroundingSource = Field(..., description="Grounding source category")
    retrieval_skipped: bool = Field(
        False,
        description="True when retrieval_mode is disabled; auto-set during construction",
    )
    retrieval_disabled_reason: str | None = Field(
        None,
        description="Reason retrieval was skipped, if applicable",
    )
    route_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Routes called, queries sent, and ontology constraints",
    )
    raw_candidate_count: int = Field(
        0,
        ge=0,
        description="Number of raw candidates before deduplication",
    )
    merged_candidate_count: int = Field(
        0,
        ge=0,
        description="Number of candidates after deduplication",
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Non-fatal route errors encountered during retrieval",
    )
    selected_candidate_code: str | None = Field(
        None,
        description="Final selected candidate code, if available",
    )

    @model_validator(mode="before")
    @classmethod
    def auto_set_retrieval_skipped(cls, data: Any) -> Any:
        """Auto-set retrieval_skipped=True when retrieval_mode is disabled."""
        if isinstance(data, dict) and data.get("retrieval_mode") == "disabled":
            data.setdefault("retrieval_skipped", True)
        return data

    @model_validator(mode="after")
    def validate_disabled_consistency(self) -> RetrievalTrace:
        if self.retrieval_mode == RetrievalMode.DISABLED:
            if self.is_grounded:
                raise ValueError(
                    "disabled retrieval_mode cannot be grounded; is_grounded must be False"
                )
            if self.grounding_source != GroundingSource.NONE:
                raise ValueError("disabled retrieval_mode requires grounding_source=none")
        return self

    model_config = {"frozen": True}


class RerankDecision(BaseModel):
    """Final decision before converting to MappingResult.

    For public/local production mode, an ungrounded result requires either
    is_unmapped=True or a non-production policy (e.g. 'research_debug').
    """

    selected_code: str | None = Field(
        None,
        description="Candidate code selected by reranker; None when unmapped",
    )
    selected_candidate_id: str | None = Field(
        None,
        description="Stable candidate identifier, if introduced",
    )
    is_unmapped: bool = Field(False, description="Whether no mapping was selected")
    is_grounded: bool = Field(
        ...,
        description="True only if selected_code was chosen from retrieved candidates",
    )
    grounding_source: GroundingSource = Field(..., description="Where grounding came from")
    retrieval_mode: RetrievalMode = Field(..., description="Layer 2 mode used")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Decision confidence in [0, 1]",
    )
    reasoning: str | None = Field(
        None,
        description="Clinical or data-manager-facing explanation",
    )
    alternative_codes: list[str] = Field(
        default_factory=list,
        description="Candidate alternatives by code; may be empty in disabled mode",
    )
    alternatives: list[RerankAlternative] = Field(
        default_factory=list,
        description="Structured candidate alternatives with reviewer-facing explanations",
    )
    policy: str = Field(
        "production_grounded",
        description="production_grounded, disabled_llm_only, or research_debug",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> RerankDecision:
        if self.retrieval_mode == RetrievalMode.DISABLED:
            if self.is_grounded:
                raise ValueError(
                    "disabled retrieval_mode cannot be grounded; is_grounded must be False"
                )
            if self.grounding_source != GroundingSource.NONE:
                raise ValueError("disabled retrieval_mode requires grounding_source=none")
        if (
            not self.is_grounded
            and self.retrieval_mode != RetrievalMode.DISABLED
            and not self.is_unmapped
            and self.policy == "production_grounded"
        ):
            raise ValueError(
                "ungrounded result in public/local production mode requires "
                "is_unmapped=True or a non-production policy (research_debug)"
            )
        return self

    model_config = {"frozen": True}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 pipeline models  (planned architecture contracts)
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalRoutePlan(BaseModel):
    """Output of RetrievalRouter: a planned description of what retrieval should happen.

    No retrieval has occurred at this point.  The plan is consumed by the
    concrete retriever in the next pipeline stage.

    Disabled mode is always ungrounded (is_grounded_mode=False, route_calls=[]).
    retrieval_skipped is auto-set to True when retrieval_mode is disabled.
    """

    retrieval_mode: RetrievalMode = Field(..., description="User-facing Layer 2 mode")
    is_grounded_mode: bool = Field(
        ...,
        description="True for public/local grounded modes; False for disabled",
    )
    retrieval_skipped: bool = Field(
        False,
        description="True when retrieval_mode is disabled; auto-set during construction",
    )
    grounding_source: GroundingSource = Field(
        ...,
        description="Grounding source category for this route",
    )
    queries: list[str] = Field(
        default_factory=list,
        description="Query strings to send to the retriever",
    )
    target_ontology_constraint: str | None = Field(
        None,
        description="Hard target-ontology constraint passed through from caller",
    )
    allowed_target_ontologies: list[str] | None = Field(
        None,
        description="Caller-selected ontology allow-list; None means unrestricted",
    )
    candidate_ontologies: list[str] = Field(
        default_factory=list,
        description="Ontology families the retriever should search",
    )
    route_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Planned retrieval call descriptions (not actual HTTP calls)",
    )
    retrieval_disabled_reason: str | None = Field(
        None,
        description="Why retrieval was skipped, if applicable",
    )

    @model_validator(mode="before")
    @classmethod
    def auto_set_retrieval_skipped(cls, data: Any) -> Any:
        """Auto-set retrieval_skipped=True when retrieval_mode is disabled."""
        if isinstance(data, dict) and data.get("retrieval_mode") == "disabled":
            data.setdefault("retrieval_skipped", True)
        return data

    @field_validator("allowed_target_ontologies")
    @classmethod
    def normalise_allowed_target_ontologies(
        cls,
        v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for ontology in v:
            text = str(ontology or "").upper().strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized or None

    @model_validator(mode="after")
    def validate_disabled_consistency(self) -> RetrievalRoutePlan:
        if self.retrieval_mode == RetrievalMode.DISABLED:
            if self.is_grounded_mode:
                raise ValueError(
                    "disabled retrieval_mode cannot be grounded; is_grounded_mode must be False"
                )
            if self.grounding_source != GroundingSource.NONE:
                raise ValueError("disabled retrieval_mode requires grounding_source=none")
        return self

    model_config = {"frozen": True}
