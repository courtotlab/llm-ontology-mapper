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

from typing import Any, Sequence

from llm_ontology_mapper.llm_reranker import fallback_alternative_explanation
from llm_ontology_mapper.models import (
    AlternativeMapping,
    LogicType,
    MappingMetadata,
    MappingResult,
    NormalizedCandidate,
    QueryPlan,
    RAGDebugInfo,
    RerankDecision,
    RerankAlternative,
    RetrievalMode,
    RetrievalTrace,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class MappingResultBuilderError(Exception):
    """
    Raised when pipeline inputs are inconsistent and a MappingResult cannot
    be built.

    Causes include: disabled mode, selected_code absent from candidates,
    target_ontology_constraint mismatch, ungrounded selected decision,
    and empty candidates for a non-unmapped decision.
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
            )

        return self._build_selected(
            query_plan=query_plan,
            rerank_decision=rerank_decision,
            candidates=candidates,
            retrieval_trace=retrieval_trace,
            source_type=source_type,
            max_alternatives=max_alternatives,
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
    ) -> MappingResult:
        # Validate grounding
        if not rerank_decision.is_grounded:
            raise MappingResultBuilderError(
                f"rerank_decision.is_grounded=False for a non-unmapped decision; "
                f"public/local production-grounded results must be grounded."
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

        # Validate target_ontology_constraint
        if query_plan.target_ontology_constraint:
            target_upper = query_plan.target_ontology_constraint.upper()
            if selected_candidate.ontology != target_upper:
                raise MappingResultBuilderError(
                    f"Selected candidate ontology={selected_candidate.ontology!r} "
                    f"does not match target_ontology_constraint={target_upper!r}."
                )

        alternatives = _build_alternatives(
            structured_alternatives=rerank_decision.alternatives,
            alternative_codes=rerank_decision.alternative_codes,
            candidates=candidates,
            exclude_code=selected_code,
            max_alternatives=max_alternatives,
        )

        metadata = _build_metadata(
            rerank_decision=rerank_decision,
            query_plan=query_plan,
            candidates=candidates,
            selected_candidate=selected_candidate,
            retrieval_trace=retrieval_trace,
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
    ) -> MappingResult:
        metadata = _build_metadata(
            rerank_decision=rerank_decision,
            query_plan=query_plan,
            candidates=candidates,
            selected_candidate=None,
            retrieval_trace=retrieval_trace,
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


def _build_alternatives(
    *,
    structured_alternatives: Sequence[RerankAlternative],
    alternative_codes: list[str],
    candidates: Sequence[NormalizedCandidate],
    exclude_code: str | None,
    max_alternatives: int = 5,
) -> list[AlternativeMapping]:
    """
    Build grounded AlternativeMapping objects from structured alternatives,
    legacy alternative_codes, or retrieval-ranked fallback candidates.
    """
    code_map: dict[str, NormalizedCandidate] = {c.code: c for c in candidates}
    result: list[AlternativeMapping] = []
    seen_codes: set[str] = set()

    def _append(
        *,
        code: str,
        confidence: float,
        explanation: str,
    ) -> None:
        if len(result) >= max_alternatives:
            return
        if code == exclude_code or code in seen_codes:
            return
        candidate = code_map.get(code)
        if candidate is None:
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
        seen_codes.add(code)

    for alt in structured_alternatives:
        _append(
            code=alt.code,
            confidence=alt.confidence,
            explanation=alt.explanation,
        )

    for code in alternative_codes:
        candidate = code_map.get(code)
        if candidate is None:
            continue
        _append(
            code=code,
            confidence=_candidate_confidence(candidate),
            explanation=fallback_alternative_explanation(candidate),
        )

    for candidate in candidates:
        _append(
            code=candidate.code,
            confidence=_candidate_confidence(candidate),
            explanation=fallback_alternative_explanation(candidate),
        )

        if len(result) >= max_alternatives:
            break

    return result


def _candidate_confidence(candidate: NormalizedCandidate) -> float:
    """Best available confidence value for a candidate, clamped to [0, 1]."""
    if candidate.normalized_score is not None:
        return candidate.normalized_score
    rs = candidate.raw_score
    if rs is not None and 0.0 <= rs <= 1.0:
        return rs
    return 0.5


def _build_metadata(
    *,
    rerank_decision: RerankDecision,
    query_plan: QueryPlan,
    candidates: Sequence[NormalizedCandidate],
    selected_candidate: NormalizedCandidate | None,
    retrieval_trace: RetrievalTrace | None,
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
        "query_plan": {
            "original_term": query_plan.original_term,
            "inferred_meaning": query_plan.inferred_meaning,
            "semantic_type": query_plan.semantic_type,
            "expanded_queries": list(query_plan.expanded_queries),
            "target_ontology_constraint": query_plan.target_ontology_constraint,
            "preferred_ontology": query_plan.preferred_ontology,
            "reasoning": query_plan.reasoning,
        },
    }

    if query_plan.target_ontology_constraint:
        pipeline_info["target_ontology_constraint"] = query_plan.target_ontology_constraint

    if retrieval_trace is not None:
        pipeline_info["retrieval_trace"] = retrieval_trace.model_dump(mode="json")

    if selected_candidate is not None and selected_candidate.provenance is not None:
        pipeline_info["selected_candidate_provenance"] = selected_candidate.provenance

    rag_debug = RAGDebugInfo(
        query_sent=query_plan.original_term,
        candidates_retrieved=[pipeline_info],
        top_k=len(candidates),
    )

    return MappingMetadata(
        model="pipeline",
        provider="grounded_pipeline",
        rag_debug=rag_debug,
    )
