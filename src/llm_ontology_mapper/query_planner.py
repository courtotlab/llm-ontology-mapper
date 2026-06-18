"""
QueryPlanner — LLM-assisted source term interpretation before retrieval.

Calls a configured LLM provider to produce a structured QueryPlan from a
source term and optional context.  Does not call external ontology APIs,
does not produce final ontology codes, and does not perform retrieval.

Python enforces hard constraints after the LLM responds:
  - original_term and original_label are copied from the caller.
  - retrieval_mode is copied from the caller; the LLM cannot change it.
  - target_ontology overrides any ontology the LLM recommends.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import QueryPlan, RetrievalMode
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "assets" / "prompts"


# ─────────────────────────────────────────────────────────────────────────────
# Public error type
# ─────────────────────────────────────────────────────────────────────────────


class QueryPlanningError(Exception):
    """Raised when the LLM response cannot be parsed into a valid QueryPlan."""


# ─────────────────────────────────────────────────────────────────────────────
# QueryPlanner
# ─────────────────────────────────────────────────────────────────────────────


class QueryPlanner:
    """
    LLM-assisted query planner.

    Calls a configured provider to interpret a source term and produce a
    structured QueryPlan before retrieval.  All hard constraints are enforced
    by Python after the LLM returns its response.

    Usage::

        from llm_ontology_mapper.providers import OpenAIProvider
        from llm_ontology_mapper.query_planner import QueryPlanner

        planner = QueryPlanner(OpenAIProvider(model="gpt-4o"))
        plan = planner.plan(
            source_term="sys_bp",
            clinical_area="cardiology",
        )
        print(plan.expanded_queries)    # ['systolic blood pressure', 'systolic BP']
        print(plan.preferred_ontology)  # 'LOINC'
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
            else _PROMPT_DIR / "query_planner_prompt.txt"
        )
        self._prompt_template: str = path.read_text(encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(
        self,
        source_term: str,
        source_label: str | None = None,
        clinical_area: str | None = None,
        target_ontology: str | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
    ) -> QueryPlan:
        """
        Produce a QueryPlan by calling the LLM provider.

        The LLM infers: normalized_term, expanded_queries, inferred_meaning,
        semantic_type, candidate_ontologies, preferred_ontology, reasoning,
        confidence.

        Python always enforces:
        - original_term / original_label from caller (not LLM).
        - retrieval_mode from caller.
        - target_ontology: overrides preferred_ontology and candidate_ontologies.
        - expanded_queries: falls back to source_label or source_term if empty.
        - retrieval_disabled_reason: set for disabled mode.

        Raises:
            QueryPlanningError: LLM response is not valid JSON.
            ValueError:         retrieval_mode string is not a valid RetrievalMode.
        """
        mode = (
            RetrievalMode(retrieval_mode)
            if isinstance(retrieval_mode, str)
            else retrieval_mode
        )
        target_norm = target_ontology.upper().strip() if target_ontology else None

        messages = self._build_messages(
            source_term, source_label, clinical_area, target_norm, mode
        )

        logger.debug(
            "QueryPlanner.plan: source_term=%r retrieval_mode=%s target_ontology=%r",
            source_term, mode.value, target_norm,
        )
        response = self._provider.complete(messages, temperature=0.1, max_tokens=512)

        raw = _strip_fences(response.content)
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise QueryPlanningError(
                f"LLM returned malformed JSON for source_term={source_term!r}: {exc}\n"
                f"Response (first 500 chars): {response.content[:500]!r}"
            ) from exc

        return self._build_plan(data, source_term, source_label, target_norm, mode)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(
        self,
        source_term: str,
        source_label: str | None,
        clinical_area: str | None,
        target_ontology: str | None,
        mode: RetrievalMode,
    ) -> list[ChatMessage]:
        label_section = f"- Label: {source_label}\n" if source_label else ""
        area_section = f"- Clinical area: {clinical_area}\n" if clinical_area else ""
        ontology_section = (
            f"- Target ontology (HARD CONSTRAINT): {target_ontology}\n"
            if target_ontology
            else ""
        )

        content = self._prompt_template.format(
            source_term=source_term,
            optional_label_section=label_section,
            optional_clinical_area_section=area_section,
            optional_target_ontology_section=ontology_section,
        )

        return [
            ChatMessage(
                role="system",
                content=(
                    "You are a biomedical retrieval planning assistant. "
                    "Return only valid JSON."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]

    def _build_plan(
        self,
        data: dict[str, Any],
        source_term: str,
        source_label: str | None,
        target_ontology: str | None,
        mode: RetrievalMode,
    ) -> QueryPlan:
        # ── Extract LLM-provided fields with safe type coercion ───────────────

        normalized_term: str = (
            str(data.get("normalized_term") or "").strip() or source_term.strip()
        )

        raw_queries = data.get("expanded_queries")
        if isinstance(raw_queries, list):
            expanded_queries = [
                str(q).strip() for q in raw_queries if q and str(q).strip()
            ]
        elif isinstance(raw_queries, str) and raw_queries.strip():
            expanded_queries = [raw_queries.strip()]
        else:
            expanded_queries = []

        inferred_meaning: str | None = data.get("inferred_meaning") or None
        semantic_type: str | None = data.get("semantic_type") or None

        raw_onto = data.get("candidate_ontologies")
        if isinstance(raw_onto, list):
            candidate_ontologies = [
                str(o).upper().strip() for o in raw_onto if o and str(o).upper().strip()
            ]
        elif isinstance(raw_onto, str) and raw_onto.strip():
            candidate_ontologies = [raw_onto.upper().strip()]
        else:
            candidate_ontologies = []

        preferred_ontology: str | None = (
            str(data.get("preferred_ontology") or "").upper().strip() or None
        )
        reasoning: str | None = data.get("reasoning") or None

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        # ── Python-side enforcement: caller constraints always win ─────────────

        if target_ontology:
            preferred_ontology = target_ontology
            candidate_ontologies = [target_ontology]

        if not expanded_queries:
            expanded_queries = [
                source_label.strip() if source_label else source_term.strip()
            ]

        disabled_reason: str | None = None
        if mode == RetrievalMode.DISABLED:
            disabled_reason = "Retrieval disabled by caller"

        return QueryPlan(
            original_term=source_term,
            original_label=source_label,
            normalized_term=normalized_term,
            expanded_queries=expanded_queries,
            inferred_meaning=inferred_meaning,
            semantic_type=semantic_type,
            candidate_ontologies=candidate_ontologies,
            preferred_ontology=preferred_ontology,
            retrieval_mode=mode,
            target_ontology_constraint=target_ontology,
            retrieval_disabled_reason=disabled_reason,
            reasoning=reasoning,
            confidence=confidence,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
