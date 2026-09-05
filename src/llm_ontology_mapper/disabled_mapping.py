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
    one is present, unless the result is UNKNOWN/UNMAPPED. Applies to the
    selected mapping and to every alternative.
  - allowed_target_ontologies is enforced: the returned ontology must be in the
    allow-list when present, unless the result is UNKNOWN/UNMAPPED. Applies to
    the selected mapping and to every alternative.
  - Confidence must be in [0, 1]; values outside this range raise an error for
    the selected mapping. An out-of-range or higher-than-selected alternative
    confidence causes that one alternative to be dropped, not the whole result.
  - Up to max_alternatives LLM-suggested alternatives may be attached
    (rank 2..max_alternatives+1); the LLM is never required to fill all of
    them, and an abstained (UNKNOWN/UNMAPPED) result always has zero
    alternatives. Alternatives are LLM-only suggestions -- no retrieval, no
    candidate cross-checking -- so they remain ungrounded like the selected
    mapping.

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
    AlternativeMapping,
    GroundingSource,
    LogicType,
    MappingMetadata,
    MappingResult,
    QueryPlan,
    RAGDebugInfo,
    RetrievalMode,
)
from llm_ontology_mapper.ontology_identity import canonical_ontology, normalize_code_for_ontology
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
        max_alternatives: int = 5,
    ) -> MappingResult:
        """
        Produce an LLM-only MappingResult for disabled retrieval mode.

        Args:
            query_plan:   Must have retrieval_mode=DISABLED.
            source_type:  Optional source data type hint (radio, text, …).
            max_alternatives: Upper bound on ranked alternatives the LLM may
                return alongside the selected mapping (0 disables alternatives
                entirely). Same knob as MappingResultBuilder.build's
                max_alternatives for public/local modes.

        Returns:
            A MappingResult with logic_type=LLM and is_grounded=False.

        Raises:
            DisabledMappingError: non-disabled retrieval_mode, malformed JSON,
                confidence outside [0, 1], negative max_alternatives, or
                target_ontology_constraint violation.
        """
        if query_plan.retrieval_mode != RetrievalMode.DISABLED:
            raise DisabledMappingError(
                f"DisabledMappingRunner only handles retrieval_mode=disabled. "
                f"Got retrieval_mode={query_plan.retrieval_mode.value!r}. "
                f"Use LLMReranker + MappingResultBuilder for public/local modes."
            )
        if max_alternatives < 0:
            raise DisabledMappingError(
                f"max_alternatives must be >= 0, got {max_alternatives!r}."
            )

        messages = self._build_messages(query_plan, max_alternatives)

        logger.debug(
            "DisabledMappingRunner.map: source_term=%r target_ontology_constraint=%r "
            "max_alternatives=%d",
            query_plan.original_term,
            query_plan.target_ontology_constraint,
            max_alternatives,
        )

        response = self._provider.complete(
            messages, temperature=0.1, max_tokens=_response_max_tokens(max_alternatives)
        )

        raw = _strip_fences(response.content)
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DisabledMappingError(
                f"LLM returned malformed JSON for "
                f"source_term={query_plan.original_term!r}: {exc}\n"
                f"Response (first 500 chars): {response.content[:500]!r}"
            ) from exc

        return self._build_result(
            data, query_plan, source_type, response.model, max_alternatives
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(
        self, query_plan: QueryPlan, max_alternatives: int
    ) -> list[ChatMessage]:
        query_context = _build_query_context(query_plan)
        target_constraint_section = _build_target_constraint_section(query_plan)
        alternatives_section = _build_alternatives_section(max_alternatives)

        content = self._prompt_template.format(
            query_context=query_context,
            target_constraint_section=target_constraint_section,
            alternatives_section=alternatives_section,
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
        max_alternatives: int,
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
        alternatives: list[AlternativeMapping] = []

        if is_unmapped:
            # Abstention: no selected mapping and no ranked alternatives, even
            # when the LLM output an "alternatives" array anyway.
            target_code = _UNMAPPED_CODE
            target_term = _UNMAPPED_TERM
            ontology = _UNMAPPED_ONTOLOGY
        else:
            # Enforce target_ontology_constraint (Python-side; LLM cannot override).
            # Uses the shared canonical-ontology alias machinery so equivalent
            # spellings (e.g. HP/HPO, SNOMED/SNOMED-CT, RXNORM/RxNorm) match.
            if query_plan.target_ontology_constraint and not _ontology_matches_constraint(
                raw_ontology, query_plan.target_ontology_constraint
            ):
                raise DisabledMappingError(
                    f"LLM returned ontology={raw_ontology!r} but "
                    f"target_ontology_constraint={query_plan.target_ontology_constraint.upper()!r}; "
                    f"the disabled-mode LLM must not return codes outside the "
                    f"constrained ontology."
                )
            allowed_ontologies = _normalize_allowed_target_ontologies(
                query_plan.allowed_target_ontologies
            )
            if allowed_ontologies is not None and (
                canonical_ontology(raw_ontology) or raw_ontology
            ) not in allowed_ontologies:
                target_code = _UNMAPPED_CODE
                target_term = _UNMAPPED_TERM
                ontology = _UNMAPPED_ONTOLOGY
                notes = _append_note(
                    notes,
                    "Disabled-mode LLM output used an ontology outside allowed_target_ontologies.",
                )
            else:
                target_code = raw_code
                target_term = raw_term
                ontology = raw_ontology
                alternatives = _parse_alternatives(
                    data=data,
                    query_plan=query_plan,
                    selected_code=target_code,
                    selected_ontology=ontology,
                    selected_confidence=raw_conf,
                    max_alternatives=max_alternatives,
                )

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
            alternatives=alternatives,
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
    return code.upper() in ("UNMAPPED", "UNKNOWN:UNMAPPED")


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
    if plan.allowed_target_ontologies:
        lines.append(f"- Allowed target ontologies: {', '.join(plan.allowed_target_ontologies)}")
    if plan.retrieval_disabled_reason:
        lines.append(f"- Retrieval disabled reason: {plan.retrieval_disabled_reason}")
    if plan.reasoning:
        lines.append(f"- Plan reasoning: {plan.reasoning}")
    return "\n".join(lines)


def _build_target_constraint_section(plan: QueryPlan) -> str:
    """Return the target-ontology constraint block for the prompt, or empty string."""
    if not plan.target_ontology_constraint and not plan.allowed_target_ontologies:
        return ""
    if plan.target_ontology_constraint:
        constraint = plan.target_ontology_constraint.upper()
        return (
            f"## Target ontology constraint (HARD)\n"
            f"You MUST only return a code from the {constraint} ontology. "
            f"If you cannot map to {constraint}, return UNKNOWN:UNMAPPED.\n\n"
        )

    allowed = ", ".join(plan.allowed_target_ontologies or [])
    return (
        f"## Target ontology allow-list (HARD)\n"
        f"You MUST only return a code from one of these ontologies: {allowed}. "
        f"If you cannot map to one of them, return UNKNOWN:UNMAPPED.\n\n"
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
            "allowed_target_ontologies": query_plan.allowed_target_ontologies,
            "preferred_ontology": query_plan.preferred_ontology,
            "reasoning": query_plan.reasoning,
        },
    }

    if query_plan.target_ontology_constraint:
        disabled_info["target_ontology_constraint"] = query_plan.target_ontology_constraint
    if query_plan.allowed_target_ontologies is not None:
        disabled_info["allowed_target_ontologies"] = list(query_plan.allowed_target_ontologies)

    rag_debug = RAGDebugInfo(
        query_sent=query_plan.original_term,
        candidates_retrieved=[disabled_info],
        top_k=0,
        auto_accepted=False,
        auto_accept_threshold=0.0,
    )

    return MappingMetadata(
        model=response_model,
        provider=provider_name,
        latency_ms=None,
        prompt_tokens=None,
        completion_tokens=None,
        rag_debug=rag_debug,
    )


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


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


def _ontology_matches_constraint(raw_ontology: str, constraint: str) -> bool:
    """True when raw_ontology and constraint resolve to the same canonical
    ontology, using the shared alias machinery (HP/HPO, SNOMED/SNOMED-CT,
    RXNORM/RxNorm, …) instead of a bespoke prefix comparison."""
    raw_canonical = canonical_ontology(raw_ontology) or raw_ontology.upper().strip()
    constraint_canonical = canonical_ontology(constraint) or constraint.upper().strip()
    return raw_canonical == constraint_canonical


def _append_note(existing: str | None, note: str) -> str:
    if not existing:
        return note
    return f"{note} {existing}"


def _build_alternatives_section(max_alternatives: int) -> str:
    """Return the alternatives-instruction block for the prompt."""
    if max_alternatives <= 0:
        return (
            "## Alternatives\n"
            'Do not return alternatives. "alternatives" MUST be an empty array: [].\n\n'
        )
    return (
        "## Alternatives\n"
        f"- You MAY also return up to {max_alternatives} alternative mapping(s) in "
        '"alternatives" if there are other genuinely plausible interpretations of the '
        "source term.\n"
        f"- Only include an alternative when it is a real candidate worth a reviewer's "
        f"attention — never invent filler alternatives merely to reach {max_alternatives}.\n"
        "- It is completely normal and expected to return zero alternatives when no other "
        "mapping is plausible.\n"
        "- Rank alternatives from strongest to weakest by confidence.\n"
        "- Do not repeat the selected target_code as an alternative.\n"
        "- Every alternative must be a distinct ontology code.\n"
        "- Each alternative must satisfy the same target ontology constraint as the "
        "selected mapping (if one is shown above).\n"
        "- Each alternative object must have exactly these fields: code, term, ontology, "
        "confidence (same [0, 1] scale as the selected mapping's confidence).\n"
        '- If you return UNKNOWN:UNMAPPED as the selected mapping, "alternatives" MUST be '
        "an empty array: [].\n\n"
    )


def _response_max_tokens(max_alternatives: int) -> int:
    """Scale the completion budget with how many alternatives may be requested."""
    return 512 + max(0, max_alternatives) * 128


def _parse_alternatives(
    *,
    data: dict[str, Any],
    query_plan: QueryPlan,
    selected_code: str,
    selected_ontology: str,
    selected_confidence: float,
    max_alternatives: int,
) -> list[AlternativeMapping]:
    """Parse LLM-suggested alternatives for disabled mode.

    No candidates exist in disabled mode (LLM-only, ungrounded), so unlike
    LLMReranker's structured alternatives this validates each item directly
    against the shared target-ontology constraint and code/ontology identity
    rules -- no candidate list to cross-check against. Violating items are
    dropped individually; they never invalidate the selected result.
    """
    if max_alternatives <= 0:
        return []

    raw_items = data.get("alternatives") or []
    if not isinstance(raw_items, list):
        return []

    target_constraint = query_plan.target_ontology_constraint
    allowed_ontologies = _normalize_allowed_target_ontologies(
        query_plan.allowed_target_ontologies
    )

    normalized_selected_code = normalize_code_for_ontology(selected_code, selected_ontology)
    seen_identities: set[tuple[str, str]] = {
        (
            canonical_ontology(selected_ontology) or selected_ontology.upper().strip(),
            normalized_selected_code.upper().strip(),
        )
    }

    alternatives: list[AlternativeMapping] = []
    for item in raw_items:
        if len(alternatives) >= max_alternatives:
            break
        if not isinstance(item, dict):
            continue

        raw_code = str(item.get("code") or "").strip()
        raw_term = str(item.get("term") or "").strip()
        raw_ontology = str(item.get("ontology") or "").strip().upper()
        if not raw_code or not raw_term or not raw_ontology:
            continue

        if target_constraint and not _ontology_matches_constraint(raw_ontology, target_constraint):
            continue
        if allowed_ontologies is not None and (
            canonical_ontology(raw_ontology) or raw_ontology
        ) not in allowed_ontologies:
            continue

        raw_confidence_value = item.get("confidence")
        if raw_confidence_value is None:
            continue
        try:
            raw_confidence = float(raw_confidence_value)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= raw_confidence <= 1.0):
            continue
        if raw_confidence > selected_confidence:
            # Alternatives must not outrank the selected mapping's confidence;
            # drop rather than raise so one bad alternative doesn't cost the
            # otherwise-valid selected result.
            continue

        normalized_code = normalize_code_for_ontology(raw_code, raw_ontology)
        identity = (
            canonical_ontology(raw_ontology) or raw_ontology,
            normalized_code.upper().strip(),
        )
        if identity in seen_identities:
            continue

        try:
            alternative = AlternativeMapping(
                code=normalized_code,
                term=raw_term,
                ontology=raw_ontology,
                confidence=raw_confidence,
                source="llm",
                explanation=None,
            )
        except ValueError:
            # code/ontology identity mismatch (e.g. HPO ontology with a
            # non-HP code) -- drop this one alternative, keep the rest.
            continue

        alternatives.append(alternative)
        seen_identities.add(identity)

    return alternatives
