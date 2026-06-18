"""
LLMReranker — candidate-grounded reranking via an LLM provider.

Receives a QueryPlan and a list of already-normalized NormalizedCandidate
objects, asks the LLM to select the best candidate or return UNMAPPED, and
returns a RerankDecision.

Hard constraints enforced by Python (LLM cannot override):
  - selected_candidate_id must be in the provided candidate list.
  - selected_code must match the code of the selected candidate.
  - selected_code must be in the candidate code set.
  - alternative_codes must be a subset of the candidate code set.
  - target_ontology_constraint is enforced: selected candidate's ontology
    must match when the constraint is present.
  - disabled mode raises immediately — it does not produce candidates.

Phase 5A of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from llm_ontology_mapper.models import (
    GroundingSource,
    NormalizedCandidate,
    QueryPlan,
    RerankDecision,
    RetrievalMode,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "assets" / "prompts"


# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class LLMRerankerError(Exception):
    """
    Raised when the reranker cannot produce a valid RerankDecision.

    Causes include: disabled mode input, malformed JSON from the LLM,
    hallucinated candidate IDs or codes, code-ID mismatches, alternative
    codes outside the candidate list, target ontology violations, and
    confidence values outside [0, 1].
    """


# ─────────────────────────────────────────────────────────────────────────────
# LLMReranker
# ─────────────────────────────────────────────────────────────────────────────


class LLMReranker:
    """
    Candidate-grounded LLM reranker.

    Calls a configured LLM provider to select the best NormalizedCandidate
    for a given QueryPlan, or to return UNMAPPED when no candidate is correct.

    Supports exactly two grounded retrieval modes: public and local.
    Raises LLMRerankerError for disabled-mode input.

    Empty candidate lists are handled without calling the provider: an
    UNMAPPED RerankDecision is returned immediately with is_unmapped=True.

    Usage::

        reranker = LLMReranker(provider)
        decision = reranker.rerank(query_plan, merged_candidates)
        if decision.is_unmapped:
            print("No suitable candidate found.")
        else:
            print(decision.selected_code, decision.confidence)
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
            else _PROMPT_DIR / "llm_reranker_prompt.txt"
        )
        self._prompt_template: str = path.read_text(encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query_plan: QueryPlan,
        candidates: Sequence[NormalizedCandidate],
    ) -> RerankDecision:
        """
        Select the best candidate for the source term or return UNMAPPED.

        Args:
            query_plan:  The planned interpretation of the source term.
            candidates:  Already-normalized and merged candidate objects.

        Returns:
            RerankDecision with is_grounded=True and a selected code, or
            is_grounded=False and is_unmapped=True when no match is found
            (including the empty-candidates case, which skips the LLM call).

        Raises:
            LLMRerankerError: disabled mode, malformed JSON, hallucinated
                codes, mismatched IDs, or invalid confidence.
        """
        mode = query_plan.retrieval_mode

        if mode == RetrievalMode.DISABLED:
            raise LLMRerankerError(
                "LLMReranker is for grounded public/local modes only. "
                "disabled mode does not produce candidates and must use a "
                "separate LLM-only mapping path."
            )

        grounding_source = (
            GroundingSource.PUBLIC_API
            if mode == RetrievalMode.PUBLIC
            else GroundingSource.LOCAL_SAPBERT
        )

        # Empty candidate list — return unmapped without calling the provider
        if not candidates:
            return RerankDecision(
                selected_code=None,
                selected_candidate_id=None,
                is_unmapped=True,
                is_grounded=False,
                grounding_source=GroundingSource.NONE,
                retrieval_mode=mode,
                confidence=0.0,
                reasoning="No candidates were provided for reranking.",
                alternative_codes=[],
                policy="production_grounded",
            )

        # Stable candidate ID map: C1, C2, C3, ...
        candidate_map: dict[str, NormalizedCandidate] = {
            f"C{i + 1}": c for i, c in enumerate(candidates)
        }
        candidate_codes: set[str] = {c.code for c in candidates}

        messages = self._build_messages(query_plan, candidate_map)

        logger.debug(
            "LLMReranker.rerank: source_term=%r mode=%s num_candidates=%d",
            query_plan.original_term,
            mode.value,
            len(candidates),
        )

        response = self._provider.complete(messages, temperature=0.1, max_tokens=512)

        raw = _strip_fences(response.content)
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMRerankerError(
                f"LLM returned malformed JSON for "
                f"source_term={query_plan.original_term!r}: {exc}\n"
                f"Response (first 500 chars): {response.content[:500]!r}"
            ) from exc

        return self._build_decision(
            data,
            candidate_map=candidate_map,
            candidate_codes=candidate_codes,
            retrieval_mode=mode,
            grounding_source=grounding_source,
            query_plan=query_plan,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(
        self,
        query_plan: QueryPlan,
        candidate_map: dict[str, NormalizedCandidate],
    ) -> list[ChatMessage]:
        query_context = _build_query_context(query_plan)
        candidate_list_str = _build_candidate_list(candidate_map)
        candidate_ids = ", ".join(candidate_map.keys())

        content = self._prompt_template.format(
            query_context=query_context,
            candidate_list=candidate_list_str,
            candidate_ids=candidate_ids,
        )

        return [
            ChatMessage(
                role="system",
                content=(
                    "You are a biomedical ontology reranking assistant. "
                    "Return only valid JSON."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]

    @staticmethod
    def _build_decision(
        data: dict[str, Any],
        *,
        candidate_map: dict[str, NormalizedCandidate],
        candidate_codes: set[str],
        retrieval_mode: RetrievalMode,
        grounding_source: GroundingSource,
        query_plan: QueryPlan,
    ) -> RerankDecision:
        is_unmapped = bool(data.get("is_unmapped", False))
        selected_cid: str | None = data.get("selected_candidate_id")
        selected_code_raw: str | None = data.get("selected_code")
        selected_code: str | None = (
            str(selected_code_raw).strip() if selected_code_raw else None
        )

        # Parse confidence — reject values outside [0, 1]
        try:
            raw_conf = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            raw_conf = 0.0
        if not (0.0 <= raw_conf <= 1.0):
            raise LLMRerankerError(
                f"LLM returned confidence={raw_conf!r} which is outside [0.0, 1.0]; "
                f"confidence must be a float in the valid range."
            )
        confidence: float = raw_conf

        reasoning: str | None = data.get("reasoning") or None

        raw_alts = data.get("alternative_codes") or []
        if isinstance(raw_alts, list):
            alternative_codes = [
                str(c).strip() for c in raw_alts if c and str(c).strip()
            ]
        else:
            alternative_codes = []

        if is_unmapped:
            return RerankDecision(
                selected_code=None,
                selected_candidate_id=None,
                is_unmapped=True,
                is_grounded=False,
                grounding_source=GroundingSource.NONE,
                retrieval_mode=retrieval_mode,
                confidence=confidence,
                reasoning=reasoning,
                alternative_codes=[],
                policy="production_grounded",
            )

        # Validate selected_candidate_id exists in the map
        if not selected_cid or selected_cid not in candidate_map:
            raise LLMRerankerError(
                f"LLM returned selected_candidate_id={selected_cid!r} which is not "
                f"in the candidate list {list(candidate_map.keys())}; "
                f"the reranker must only select from provided candidates."
            )

        selected_candidate = candidate_map[selected_cid]

        # Validate selected_code matches the selected candidate's actual code
        if selected_code != selected_candidate.code:
            raise LLMRerankerError(
                f"LLM returned selected_code={selected_code!r} but "
                f"selected_candidate_id={selected_cid!r} maps to "
                f"code={selected_candidate.code!r}; "
                f"selected_code must match the candidate's code exactly."
            )

        # Belt-and-suspenders: selected_code must be in the candidate code set
        if selected_code not in candidate_codes:
            raise LLMRerankerError(
                f"LLM returned selected_code={selected_code!r} which is not in "
                f"the candidate code set; the reranker must not invent ontology codes."
            )

        # Enforce target_ontology_constraint
        if query_plan.target_ontology_constraint:
            target_upper = query_plan.target_ontology_constraint.upper()
            if selected_candidate.ontology != target_upper:
                raise LLMRerankerError(
                    f"LLM selected candidate with ontology={selected_candidate.ontology!r} "
                    f"but target_ontology_constraint={target_upper!r}; "
                    f"selected candidate must belong to the constrained ontology."
                )

        # Validate alternative_codes are all in the candidate code set
        for alt in alternative_codes:
            if alt not in candidate_codes:
                raise LLMRerankerError(
                    f"LLM returned alternative_code={alt!r} which is not in "
                    f"the candidate list; alternative codes must come from retrieved candidates."
                )

        return RerankDecision(
            selected_code=selected_code,
            selected_candidate_id=selected_cid,
            is_unmapped=False,
            is_grounded=True,
            grounding_source=grounding_source,
            retrieval_mode=retrieval_mode,
            confidence=confidence,
            reasoning=reasoning,
            alternative_codes=alternative_codes,
            policy="production_grounded",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


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
    if plan.target_ontology_constraint:
        lines.append(
            f"- Target ontology constraint (HARD): {plan.target_ontology_constraint}"
        )
    lines.append(f"- Retrieval mode: {plan.retrieval_mode.value}")
    if plan.reasoning:
        lines.append(f"- Plan reasoning: {plan.reasoning}")
    return "\n".join(lines)


def _build_candidate_list(candidate_map: dict[str, NormalizedCandidate]) -> str:
    """Format candidates as a numbered list for the LLM prompt."""
    parts: list[str] = []
    for cid, c in candidate_map.items():
        fields = [f"{cid}: {c.code} | {c.term} | {c.ontology}"]
        score_parts: list[str] = []
        if c.normalized_score is not None:
            score_parts.append(f"normalized_score={c.normalized_score:.4f}")
        if c.raw_score is not None:
            score_parts.append(f"raw_score={c.raw_score:.4f}")
        if score_parts:
            fields.append(f"[{', '.join(score_parts)}]")
        fields.append(f"Source: {c.source}")
        fields.append(f"Query: '{c.matched_query}'")
        if c.definition:
            defn = (
                c.definition[:200] + "..."
                if len(c.definition) > 200
                else c.definition
            )
            fields.append(f"Def: {defn}")
        parts.append(" | ".join(fields))
    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
