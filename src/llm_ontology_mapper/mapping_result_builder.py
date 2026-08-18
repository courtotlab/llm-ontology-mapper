"""
MappingResultBuilder — converts pipeline decisions into MappingResult.

Accepts a QueryPlan, RerankDecision, NormalizedCandidate list, and optional
RetrievalTrace; produces the canonical MappingResult consumed by Bridge and
downstream systems.

Pure and side-effect free.  No LLM calls, no retrieval, no external services.

Phase 5B of the planned grounded mapping pipeline.

Metadata strategy
─────────────────
MappingMetadata requires model/provider (LLM-specific fields) that the builder
does not possess.  Pipeline-specific metadata (retrieval_mode, is_grounded,
grounding_source, policy, query_plan summary, retrieval_trace, candidate
provenance) is stored inside metadata.rag_debug.candidates_retrieved[0] as an
arbitrary dict.  This re-uses existing model fields without any schema change
and keeps MappingResult serialization intact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_ontology_mapper.models import (
    AlternativeMapping,
    GroundingSource,
    LogicType,
    MappingMetadata,
    MappingResult,
    NormalizedCandidate,
    QueryPlan,
    RAGDebugInfo,
    RerankAlternative,
    RerankDecision,
    RetrievalMode,
    RetrievalTrace,
)
from llm_ontology_mapper.ontology_identity import (
    canonical_ontology,
    validate_candidate_identity,
)
from llm_ontology_mapper.target_eligibility import candidate_allowed_for_targets

# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class MappingResultBuilderError(Exception):
    """
    Raised when pipeline inputs are inconsistent and a MappingResult cannot
    be built.

    Causes include: disabled mode, selected_code absent from candidates,
    target_ontology_constraint mismatch, ungrounded selected decision,
    allowed_target_ontologies mismatch, and empty candidates for a non-unmapped
    decision.
    """


# ─────────────────────────────────────────────────────────────────────────────
# MappingResultBuilder
# ─────────────────────────────────────────────────────────────────────────────

# Pre-normalised UNMAPPED sentinels — match the agentic_mapper convention.
_UNMAPPED_CODE = "UNKNOWN:UNMAPPED"
_UNMAPPED_TERM = "UNMAPPED"
_UNMAPPED_ONTOLOGY = "UNKNOWN"


class MappingResultBuilder:
    """
    Convert a grounded pipeline decision into a MappingResult.

    Handles two outcomes:
      - Selected: RerankDecision chose one of the NormalizedCandidates.
      - Unmapped: RerankDecision found no suitable match (is_unmapped=True).

    Disabled mode raises MappingResultBuilderError because the disabled
    LLM-only mapping path is a separate future phase.

    Usage::

        builder = MappingResultBuilder()
        result = builder.build(
            query_plan=plan,
            rerank_decision=decision,
            candidates=merged_candidates,
        )
    """

    def build(
        self,
        *,
        query_plan: QueryPlan,
        rerank_decision: RerankDecision,
        candidates: Sequence[NormalizedCandidate],
        retrieval_trace: RetrievalTrace | None = None,
        source_type: str | None = None,
        max_alternatives: int = 5,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        """
        Build a MappingResult from a grounded pipeline decision.

        Args:
            query_plan:       The QueryPlan used to drive retrieval.
            rerank_decision:  The LLMReranker's output decision.
            candidates:       The normalized and merged candidate list that was
                              passed to the reranker.
            retrieval_trace:  Optional retrieval trace for debugging/audit.
            source_type:      Optional source data type hint (radio, text, …).
            strict_target_ontology: When True, require canonical native-ontology
                              membership only for the selected mapping and every
                              alternative; disables the EFO imported-term
                              provenance exception.

        Returns:
            A MappingResult.

        Raises:
            MappingResultBuilderError: If inputs are inconsistent or disabled
                mode is detected.
        """
        mode = rerank_decision.retrieval_mode

        if mode == RetrievalMode.DISABLED:
            raise MappingResultBuilderError(
                "MappingResultBuilder handles grounded public/local decisions only. "
                "Disabled mode (LLM-only mapping) is a separate future phase."
            )

        if rerank_decision.is_unmapped:
            return self._build_unmapped(
                query_plan=query_plan,
                rerank_decision=rerank_decision,
                candidates=candidates,
                retrieval_trace=retrieval_trace,
                source_type=source_type,
                strict_target_ontology=strict_target_ontology,
            )

        return self._build_selected(
            query_plan=query_plan,
            rerank_decision=rerank_decision,
            candidates=candidates,
            retrieval_trace=retrieval_trace,
            source_type=source_type,
            max_alternatives=max_alternatives,
            strict_target_ontology=strict_target_ontology,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_selected(
        self,
        *,
        query_plan: QueryPlan,
        rerank_decision: RerankDecision,
        candidates: Sequence[NormalizedCandidate],
        retrieval_trace: RetrievalTrace | None,
        source_type: str | None,
        max_alternatives: int,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        # Validate grounding
        if not rerank_decision.is_grounded:
            raise MappingResultBuilderError(
                "rerank_decision.is_grounded=False for a non-unmapped decision; "
                "public/local production-grounded results must be grounded."
            )

        # Validate candidates available
        if not candidates:
            raise MappingResultBuilderError(
                "candidates is empty but rerank_decision.is_unmapped=False; "
                "cannot build a selected result without candidate records."
            )

        # Locate the selected candidate
        selected_code = rerank_decision.selected_code
        selected_candidate = _find_by_code(candidates, selected_code)
        if selected_candidate is None:
            raise MappingResultBuilderError(
                f"rerank_decision.selected_code={selected_code!r} is not present "
                f"in the provided candidates list; "
                f"the selected code must come from the candidate set."
            )
        selected_validation = validate_candidate_identity(
            ontology=selected_candidate.ontology,
            code=selected_candidate.code,
        )
        if not selected_validation.valid:
            return self._build_unmapped(
                query_plan=query_plan,
                rerank_decision=_unmapped_decision_from(
                    rerank_decision,
                    "Selected candidate ontology/code identity is inconsistent.",
                ),
                candidates=candidates,
                retrieval_trace=retrieval_trace,
                source_type=source_type,
                strict_target_ontology=strict_target_ontology,
            )

        # Validate target_ontology_constraint
        if query_plan.target_ontology_constraint and not candidate_allowed_for_targets(
            selected_candidate,
            [query_plan.target_ontology_constraint],
            strict_target_ontology=strict_target_ontology,
        ):
            raise MappingResultBuilderError(
                f"Selected candidate ontology={selected_candidate.ontology!r} "
                f"does not match "
                f"target_ontology_constraint={query_plan.target_ontology_constraint!r}."
            )

        allowed_ontologies = _normalize_allowed_target_ontologies(
            query_plan.allowed_target_ontologies
        )
        if allowed_ontologies is not None and not candidate_allowed_for_targets(
            selected_candidate,
            allowed_ontologies,
            strict_target_ontology=strict_target_ontology,
        ):
            return self._build_unmapped(
                query_plan=query_plan,
                rerank_decision=_unmapped_decision_from(
                    rerank_decision,
                    "Selected candidate ontology is outside allowed_target_ontologies.",
                ),
                candidates=candidates,
                retrieval_trace=retrieval_trace,
                source_type=source_type,
                strict_target_ontology=strict_target_ontology,
            )

        alternatives = _build_alternatives(
            structured_alternatives=rerank_decision.alternatives,
            candidates=candidates,
            exclude_candidate=selected_candidate,
            selected_confidence=rerank_decision.confidence,
            allowed_ontologies=allowed_ontologies,
            max_alternatives=max_alternatives,
            strict_target_ontology=strict_target_ontology,
        )

        metadata = _build_metadata(
            rerank_decision=rerank_decision,
            query_plan=query_plan,
            candidates=candidates,
            selected_candidate=selected_candidate,
            retrieval_trace=retrieval_trace,
            strict_target_ontology=strict_target_ontology,
        )

        return MappingResult(
            source_term=query_plan.original_term,
            source_label=query_plan.original_label,
            source_type=source_type,
            target_code=selected_candidate.code,
            target_term=selected_candidate.term,
            ontology=selected_candidate.ontology,
            confidence=rerank_decision.confidence,
            logic_type=LogicType.RAG,
            alternatives=alternatives,
            notes=rerank_decision.reasoning,
            metadata=metadata,
        )

    def _build_unmapped(
        self,
        *,
        query_plan: QueryPlan,
        rerank_decision: RerankDecision,
        candidates: Sequence[NormalizedCandidate],
        retrieval_trace: RetrievalTrace | None,
        source_type: str | None,
        strict_target_ontology: bool = False,
    ) -> MappingResult:
        metadata = _build_metadata(
            rerank_decision=rerank_decision,
            query_plan=query_plan,
            candidates=candidates,
            selected_candidate=None,
            retrieval_trace=retrieval_trace,
            strict_target_ontology=strict_target_ontology,
        )

        return MappingResult(
            source_term=query_plan.original_term,
            source_label=query_plan.original_label,
            source_type=source_type,
            target_code=_UNMAPPED_CODE,
            target_term=_UNMAPPED_TERM,
            ontology=_UNMAPPED_ONTOLOGY,
            confidence=rerank_decision.confidence,
            logic_type=LogicType.RAG,
            alternatives=[],
            notes=rerank_decision.reasoning,
            metadata=metadata,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_by_code(
    candidates: Sequence[NormalizedCandidate],
    code: str | None,
) -> NormalizedCandidate | None:
    """Return the first candidate whose code matches exactly, or None."""
    if not code:
        return None
    for c in candidates:
        if c.code == code:
            return c
    return None


def _candidate_identity(candidate: NormalizedCandidate) -> tuple[str, str]:
    """Stable identity for comparing retrieved candidates across score stages."""
    return (candidate.ontology.upper().strip(), candidate.code.upper().strip())


def _build_alternatives(
    *,
    structured_alternatives: Sequence[RerankAlternative],
    candidates: Sequence[NormalizedCandidate],
    exclude_candidate: NormalizedCandidate,
    selected_confidence: float,
    allowed_ontologies: set[str] | None,
    max_alternatives: int = 5,
    strict_target_ontology: bool = False,
) -> list[AlternativeMapping]:
    """
    Build grounded AlternativeMapping objects from structured reranker alternatives.

    At the MappingResult boundary, AlternativeMapping.confidence has the same
    meaning as MappingResult.confidence: final LLM reranker confidence. Retrieval
    scores are intentionally not used here.
    """
    code_map: dict[str, NormalizedCandidate] = {c.code: c for c in candidates}
    result: list[AlternativeMapping] = []
    seen_identities: set[tuple[str, str]] = set()
    exclude_identity = _candidate_identity(exclude_candidate)

    def _append(
        *,
        code: str,
        confidence: float,
        explanation: str,
    ) -> None:
        if len(result) >= max_alternatives:
            return
        if confidence > selected_confidence:
            raise MappingResultBuilderError(
                "Alternative reranker confidence cannot be greater than selected "
                "result confidence at the MappingResult boundary."
            )
        candidate = code_map.get(code)
        if candidate is None:
            return
        validation = validate_candidate_identity(ontology=candidate.ontology, code=candidate.code)
        if not validation.valid:
            return
        if allowed_ontologies is not None and not candidate_allowed_for_targets(
            candidate,
            allowed_ontologies,
            strict_target_ontology=strict_target_ontology,
        ):
            return
        identity = _candidate_identity(candidate)
        if identity == exclude_identity or identity in seen_identities:
            return
        result.append(
            AlternativeMapping(
                code=candidate.code,
                term=candidate.term,
                ontology=candidate.ontology,
                confidence=confidence,
                source="rag",
                explanation=explanation,
            )
        )
        seen_identities.add(identity)

    for alt in sorted(structured_alternatives, key=lambda item: item.confidence, reverse=True):
        _append(
            code=alt.code,
            confidence=alt.confidence,
            explanation=alt.explanation,
        )

    return result


def _build_metadata(
    *,
    rerank_decision: RerankDecision,
    query_plan: QueryPlan,
    candidates: Sequence[NormalizedCandidate],
    selected_candidate: NormalizedCandidate | None,
    retrieval_trace: RetrievalTrace | None,
    strict_target_ontology: bool = False,
) -> MappingMetadata:
    """
    Pack pipeline metadata into a MappingMetadata object.

    Pipeline-specific fields (not native to MappingMetadata) are stored in
    rag_debug.candidates_retrieved[0] as a plain dict so that the existing
    MappingResult schema is not changed.
    """
    pipeline_info: dict[str, Any] = {
        "retrieval_mode": rerank_decision.retrieval_mode.value,
        "is_grounded": rerank_decision.is_grounded,
        "grounding_source": rerank_decision.grounding_source.value,
        "policy": rerank_decision.policy,
        "candidate_count": len(candidates),
        "strict_target_ontology": strict_target_ontology,
        "query_plan": {
            "original_term": query_plan.original_term,
            "inferred_meaning": query_plan.inferred_meaning,
            "semantic_type": query_plan.semantic_type,
            "expanded_queries": list(query_plan.expanded_queries),
            "target_ontology_constraint": query_plan.target_ontology_constraint,
            "allowed_target_ontologies": query_plan.allowed_target_ontologies,
            "preferred_ontology": query_plan.preferred_ontology,
            "reasoning": query_plan.reasoning,
        },
    }

    if query_plan.target_ontology_constraint:
        pipeline_info["target_ontology_constraint"] = query_plan.target_ontology_constraint
    if query_plan.allowed_target_ontologies is not None:
        pipeline_info["allowed_target_ontologies"] = list(query_plan.allowed_target_ontologies)

    if retrieval_trace is not None:
        pipeline_info["retrieval_trace"] = retrieval_trace.model_dump(mode="json")

    if selected_candidate is not None and selected_candidate.provenance is not None:
        pipeline_info["selected_candidate_provenance"] = selected_candidate.provenance

    allowed_ontologies = _normalize_allowed_target_ontologies(query_plan.allowed_target_ontologies)
    pipeline_info["confidence_contract"] = (
        "MappingResult.confidence and alternatives[].confidence are final "
        "LLM reranker confidences on the same [0, 1] scale. Retrieval scores "
        "are retained separately as retrieval provenance."
    )
    pipeline_info["candidate_score_provenance"] = [
        _candidate_score_provenance(candidate) for candidate in candidates
    ]
    pipeline_info["final_ranking_trace"] = _final_ranking_trace(
        rerank_decision=rerank_decision,
        candidates=candidates,
        selected_candidate=selected_candidate,
        allowed_ontologies=allowed_ontologies,
        strict_target_ontology=strict_target_ontology,
    )

    rag_debug = RAGDebugInfo(
        query_sent=query_plan.original_term,
        candidates_retrieved=[pipeline_info],
        top_k=len(candidates),
        auto_accepted=False,
        auto_accept_threshold=0.0,
    )

    return MappingMetadata(
        model="pipeline",
        provider="grounded_pipeline",
        latency_ms=None,
        prompt_tokens=None,
        completion_tokens=None,
        rag_debug=rag_debug,
    )


def _candidate_score_provenance(candidate: NormalizedCandidate) -> dict[str, Any]:
    return {
        "code": candidate.code,
        "ontology": candidate.ontology,
        "retrieved_from_ontologies": list(candidate.retrieved_from_ontologies),
        "term": candidate.term,
        "raw_retrieval_score": candidate.raw_score,
        "normalized_retrieval_score": candidate.normalized_score,
        "retrieval_mode": candidate.retrieval_mode.value,
        "source": candidate.source,
        "matched_query": candidate.matched_query,
    }


def _final_ranking_trace(
    *,
    rerank_decision: RerankDecision,
    candidates: Sequence[NormalizedCandidate],
    selected_candidate: NormalizedCandidate | None,
    allowed_ontologies: set[str] | None,
    strict_target_ontology: bool = False,
) -> list[dict[str, Any]]:
    code_map: dict[str, NormalizedCandidate] = {
        candidate.code: candidate for candidate in candidates
    }
    trace: list[dict[str, Any]] = []

    if selected_candidate is not None:
        trace.append(
            {
                "rank": 1,
                "candidate_id": rerank_decision.selected_candidate_id,
                "code": selected_candidate.code,
                "ontology": selected_candidate.ontology,
                "reranker_score": rerank_decision.confidence,
                "final_confidence": rerank_decision.confidence,
                "confidence_source": "llm_reranker",
                "raw_retrieval_score": selected_candidate.raw_score,
                "normalized_retrieval_score": selected_candidate.normalized_score,
                "selected": True,
            }
        )

    for rank, alt in enumerate(
        sorted(rerank_decision.alternatives, key=lambda item: item.confidence, reverse=True),
        start=2,
    ):
        candidate = code_map.get(alt.code)
        if allowed_ontologies is not None and (
            candidate is None
            or not candidate_allowed_for_targets(
                candidate, allowed_ontologies, strict_target_ontology=strict_target_ontology
            )
        ):
            continue
        trace.append(
            {
                "rank": rank,
                "candidate_id": alt.candidate_id,
                "code": alt.code,
                "ontology": candidate.ontology if candidate is not None else None,
                "reranker_score": alt.confidence,
                "final_confidence": alt.confidence,
                "confidence_source": "llm_reranker",
                "raw_retrieval_score": candidate.raw_score if candidate is not None else None,
                "normalized_retrieval_score": (
                    candidate.normalized_score if candidate is not None else None
                ),
                "selected": False,
            }
        )

    return trace


def _normalize_allowed_target_ontologies(
    ontologies: list[str] | None,
) -> set[str] | None:
    if ontologies is None:
        return None
    allowed = {
        canonical_ontology(ontology) or str(ontology or "").upper().strip()
        for ontology in ontologies
        if str(ontology or "").strip()
    }
    return allowed or None


def _same_ontology(left: str, right: str) -> bool:
    left_key = canonical_ontology(left) or str(left or "").upper().strip()
    right_key = canonical_ontology(right) or str(right or "").upper().strip()
    return left_key == right_key


def _ontology_in_set(ontology: str, allowed: set[str]) -> bool:
    key = canonical_ontology(ontology) or str(ontology or "").upper().strip()
    return key in allowed


def _unmapped_decision_from(
    decision: RerankDecision,
    reason: str,
) -> RerankDecision:
    reasoning = reason
    if decision.reasoning:
        reasoning = f"{reason} {decision.reasoning}"
    return RerankDecision(
        selected_code=None,
        selected_candidate_id=None,
        is_unmapped=True,
        is_grounded=False,
        grounding_source=GroundingSource.NONE,
        retrieval_mode=decision.retrieval_mode,
        confidence=0.0,
        reasoning=reasoning,
        alternative_codes=[],
        alternatives=[],
        policy=decision.policy,
    )
