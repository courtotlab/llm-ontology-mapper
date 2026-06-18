"""
DisabledMappingRunner — LLM-only mapping for retrieval_mode="disabled".

When Layer 2 retrieval is intentionally disabled the pipeline skips:
  - public API retrieval
  - local/SapBERT retrieval
  - CandidateNormalizer
  - CandidateMerger
  - LLMReranker
  - MappingResultBuilder (grounded public/local path)

Instead, this component calls the LLM once with the QueryPlan context and builds
a MappingResult that is explicitly ungrounded.

Hard constraints enforced by Python (LLM cannot override):
  - Only QueryPlan with retrieval_mode=DISABLED is accepted.
  - target_ontology_constraint is enforced: the returned ontology must match if
    one is present, unless the result is UNKNOWN/UNMAPPED.
  - Confidence must be in [0, 1]; values outside this range raise an error.

The result is always marked:
  - logic_type = LogicType.LLM
  - is_grounded = False
  - grounding_source = none
  - policy = disabled_llm_only
  - retrieval_skipped = True

Phase 6 of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import (
    GroundingSource,
    LogicType,
    MappingMetadata,
    MappingResult,
    QueryPlan,
    RAGDebugInfo,
    RetrievalMode,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "assets" / "prompts"

# Pre-normalised UNMAPPED sentinels — match the agentic_mapper and
# MappingResultBuilder convention.
_UNMAPPED_CODE = "UNKNOWN:UNMAPPED"
_UNMAPPED_TERM = "UNMAPPED"
_UNMAPPED_ONTOLOGY = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class DisabledMappingError(Exception):
    """
    Raised when the disabled-mode mapping path cannot produce a valid result.

    Causes include: non-disabled retrieval_mode, malformed LLM JSON, confidence
    outside [0, 1], and target_ontology_constraint violations.
    """


# ─────────────────────────────────────────────────────────────────────────────
# DisabledMappingRunner
# ─────────────────────────────────────────────────────────────────────────────


class DisabledMappingRunner:
    """
    LLM-only mapping for disabled retrieval mode.

    Accepts a QueryPlan with retrieval_mode=DISABLED, calls the configured LLM
    provider once, and returns a MappingResult that is explicitly ungrounded.

    No retrieval, no candidate normalization, no reranking.

    Usage::

        runner = DisabledMappingRunner(provider)
        result = runner.map(query_plan)
        assert result.logic_type == LogicType.LLM
        assert not result.metadata.rag_debug.candidates_retrieved[0]["is_grounded"]
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        prompt_template_path: str | None = None,
    ) -> None:
        self._provider = provider
        path = (
            Path(prompt_template_path)
            if prompt_template_path
            else _PROMPT_DIR / "disabled_mapping_prompt.txt"
        )
        self._prompt_template: str = path.read_text(encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────

    def map(
        self,
        query_plan: QueryPlan,
        *,
        source_type: str | None = None,
    ) -> MappingResult:
        """
        Produce an LLM-only MappingResult for disabled retrieval mode.

        Args:
            query_plan:   Must have retrieval_mode=DISABLED.
            source_type:  Optional source data type hint (radio, text, …).

        Returns:
            A MappingResult with logic_type=LLM and is_grounded=False.

        Raises:
            DisabledMappingError: non-disabled retrieval_mode, malformed JSON,
                confidence outside [0, 1], or target_ontology_constraint
                violation.
        """
        if query_plan.retrieval_mode != RetrievalMode.DISABLED:
            raise DisabledMappingError(
                f"DisabledMappingRunner only handles retrieval_mode=disabled. "
                f"Got retrieval_mode={query_plan.retrieval_mode.value!r}. "
                f"Use LLMReranker + MappingResultBuilder for public/local modes."
            )

        messages = self._build_messages(query_plan)

        logger.debug(
            "DisabledMappingRunner.map: source_term=%r target_ontology_constraint=%r",
            query_plan.original_term,
            query_plan.target_ontology_constraint,
        )

        response = self._provider.complete(messages, temperature=0.1, max_tokens=512)

        raw = _strip_fences(response.content)
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DisabledMappingError(
                f"LLM returned malformed JSON for "
                f"source_term={query_plan.original_term!r}: {exc}\n"
                f"Response (first 500 chars): {response.content[:500]!r}"
            ) from exc

        return self._build_result(data, query_plan, source_type, response.model)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(self, query_plan: QueryPlan) -> list[ChatMessage]:
        query_context = _build_query_context(query_plan)
        target_constraint_section = _build_target_constraint_section(query_plan)

        content = self._prompt_template.format(
            query_context=query_context,
            target_constraint_section=target_constraint_section,
        )

        return [
            ChatMessage(
                role="system",
                content=(
                    "You are a biomedical ontology mapping assistant. "
                    "Retrieval is disabled. Return only valid JSON."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]

    def _build_result(
        self,
        data: dict[str, Any],
        query_plan: QueryPlan,
        source_type: str | None,
        response_model: str,
    ) -> MappingResult:
        # Extract fields with safe coercion
        raw_code: str = str(data.get("target_code") or "").strip()
        raw_term: str = str(data.get("target_term") or "").strip()
        raw_ontology: str = str(data.get("ontology") or "").strip().upper()

        # Detect unmapped convention
        is_unmapped = _is_unmapped(raw_code, raw_term, raw_ontology)

        # Validate confidence — strict, matching LLMReranker behavior
        try:
            raw_conf = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            raw_conf = 0.0
        if not (0.0 <= raw_conf <= 1.0):
            raise DisabledMappingError(
                f"LLM returned confidence={raw_conf!r} which is outside [0.0, 1.0]; "
                f"confidence must be a float in the valid range."
            )

        notes: str | None = str(data.get("notes") or "").strip() or None

        if is_unmapped:
            target_code = _UNMAPPED_CODE
            target_term = _UNMAPPED_TERM
            ontology = _UNMAPPED_ONTOLOGY
        else:
            # Enforce target_ontology_constraint (Python-side; LLM cannot override)
            if query_plan.target_ontology_constraint:
                target_upper = query_plan.target_ontology_constraint.upper()
                if raw_ontology != target_upper:
                    raise DisabledMappingError(
                        f"LLM returned ontology={raw_ontology!r} but "
                        f"target_ontology_constraint={target_upper!r}; "
                        f"the disabled-mode LLM must not return codes outside the "
                        f"constrained ontology."
                    )
            target_code = raw_code
            target_term = raw_term
            ontology = raw_ontology

        metadata = _build_metadata(
            query_plan=query_plan,
            response_model=response_model,
            provider_name=self._provider.provider_name,
        )

        return MappingResult(
            source_term=query_plan.original_term,
            source_label=query_plan.original_label,
            source_type=source_type,
            target_code=target_code,
            target_term=target_term,
            ontology=ontology,
            confidence=raw_conf,
            logic_type=LogicType.LLM,
            alternatives=[],
            notes=notes,
            metadata=metadata,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _is_unmapped(code: str, term: str, ontology: str) -> bool:
    """
    True when the LLM signalled that no mapping was found.

    Accepts both the explicit UNKNOWN:UNMAPPED convention and empty/missing values.
    """
    if not code or not term or not ontology:
        return True
    if ontology.upper() == "UNKNOWN":
        return True
    if code.upper() in ("UNMAPPED", "UNKNOWN:UNMAPPED"):
        return True
    return False


def _build_query_context(plan: QueryPlan) -> str:
    """Render QueryPlan fields as a human-readable context block."""
    lines = [f"- Source term: {plan.original_term}"]
    if plan.original_label:
        lines.append(f"- Label: {plan.original_label}")
    if plan.normalized_term:
        lines.append(f"- Normalized term: {plan.normalized_term}")
    if plan.expanded_queries:
        joined = ", ".join(f"'{q}'" for q in plan.expanded_queries)
        lines.append(f"- Expanded queries: {joined}")
    if plan.inferred_meaning:
        lines.append(f"- Inferred meaning: {plan.inferred_meaning}")
    if plan.semantic_type:
        lines.append(f"- Semantic type: {plan.semantic_type}")
    if plan.candidate_ontologies:
        lines.append(f"- Candidate ontologies: {', '.join(plan.candidate_ontologies)}")
    if plan.preferred_ontology:
        lines.append(f"- Preferred ontology: {plan.preferred_ontology}")
    if plan.retrieval_disabled_reason:
        lines.append(f"- Retrieval disabled reason: {plan.retrieval_disabled_reason}")
    if plan.reasoning:
        lines.append(f"- Plan reasoning: {plan.reasoning}")
    return "\n".join(lines)


def _build_target_constraint_section(plan: QueryPlan) -> str:
    """Return the target-ontology constraint block for the prompt, or empty string."""
    if not plan.target_ontology_constraint:
        return ""
    constraint = plan.target_ontology_constraint.upper()
    return (
        f"## Target ontology constraint (HARD)\n"
        f"You MUST only return a code from the {constraint} ontology. "
        f"If you cannot map to {constraint}, return UNKNOWN:UNMAPPED.\n\n"
    )


def _build_metadata(
    *,
    query_plan: QueryPlan,
    response_model: str,
    provider_name: str,
) -> MappingMetadata:
    """
    Pack disabled-mode pipeline metadata into a MappingMetadata object.

    Follows the same rag_debug.candidates_retrieved[0] convention as
    MappingResultBuilder so that downstream consumers can inspect pipeline
    metadata via a consistent access pattern.
    """
    disabled_info: dict[str, Any] = {
        "retrieval_mode": RetrievalMode.DISABLED.value,
        "is_grounded": False,
        "grounding_source": GroundingSource.NONE.value,
        "policy": "disabled_llm_only",
        "retrieval_skipped": True,
        "retrieval_disabled_reason": (
            query_plan.retrieval_disabled_reason or "retrieval_mode=disabled"
        ),
        "candidate_count": 0,
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
        disabled_info["target_ontology_constraint"] = (
            query_plan.target_ontology_constraint
        )

    rag_debug = RAGDebugInfo(
        query_sent=query_plan.original_term,
        candidates_retrieved=[disabled_info],
        top_k=0,
    )

    return MappingMetadata(
        model=response_model,
        provider=provider_name,
        rag_debug=rag_debug,
    )


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
