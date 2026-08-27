"""
QueryPlanner — LLM-assisted source term interpretation before retrieval.

Calls a configured LLM provider to produce a structured QueryPlan from a
source term and optional context.  Does not call external ontology APIs,
does not produce final ontology codes, and does not perform retrieval.

Python enforces hard constraints after the LLM responds:
  - original_term and original_label are copied from the caller.
  - retrieval_mode is copied from the caller; the LLM cannot change it.
  - target_ontology overrides any ontology the LLM recommends.
  - allowed_target_ontologies restricts candidate and preferred ontologies.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import QueryPlan, RetrievalMode
from llm_ontology_mapper.providers import (
    REASONING_EMPTY_RESPONSE_RETRY_TOKENS,
    BaseLLMProvider,
    ChatMessage,
    CompletionResponse,
    LLMCallConfig,
    accumulate_usage,
    extract_completion_usage,
    openai_reasoning_effort_for_model,
    provider_uses_reasoning_model,
    resolve_llm_call_params,
)

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "assets" / "prompts"
_PLAN_MAX_TOKENS = 2048
_MAX_PLAN_ATTEMPTS = 2
"""Hard cap on provider calls per plan(): the normal call plus at most one
recovery attempt (empty-response-from-reasoning-model OR malformed-JSON --
never both, since the second attempt's outcome is never re-triaged)."""


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
        llm_call_config: LLMCallConfig | None = None,
    ) -> None:
        self._provider = provider
        self._llm_call_config = llm_call_config
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
        allowed_target_ontologies: list[str] | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.PUBLIC,
        *,
        source_description: str | None = None,
        source_type: str | None = None,
        timing_sink: dict[str, float] | None = None,
        usage_sink: dict[str, Any] | None = None,
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
        - allowed_target_ontologies: filters preferred/candidate ontologies.
        - expanded_queries: falls back to source_label or source_term if empty.
        - retrieval_disabled_reason: set for disabled mode.

        Raises:
            QueryPlanningError: LLM response is not valid JSON.
            ValueError:         retrieval_mode string is not a valid RetrievalMode.
        """
        mode = RetrievalMode(retrieval_mode) if isinstance(retrieval_mode, str) else retrieval_mode
        source_description_norm = _normalize_optional_text(
            source_description,
            field_name="source_description",
        )
        source_type_norm = _normalize_optional_text(source_type, field_name="source_type")
        target_norm = target_ontology.upper().strip() if target_ontology else None
        allowed_norm = _normalize_ontology_list(allowed_target_ontologies)
        if target_norm:
            allowed_norm = [target_norm]

        messages = self._build_messages(
            source_term,
            source_label,
            source_description_norm,
            source_type_norm,
            clinical_area,
            target_norm,
            allowed_norm,
            mode,
        )

        logger.debug(
            "QueryPlanner.plan: source_term=%r retrieval_mode=%s "
            "target_ontology=%r allowed_target_ontologies=%r",
            source_term,
            mode.value,
            target_norm,
            allowed_norm,
        )
        temperature, extra_kwargs = resolve_llm_call_params(
            self._llm_call_config,
            default_temperature=0.1,
            default_extra_kwargs=_openai_reasoning_kwargs(self._provider),
        )
        response = self._complete_plan_timed(
            messages,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
            timing_sink=timing_sink,
            usage_sink=usage_sink,
        )

        # At most one recovery attempt total: an empty response from a
        # reasoning model retries with a larger completion budget; a
        # non-empty-but-malformed response retries with the normal budget.
        # These are mutually exclusive (if/elif) so a pathological response
        # can never chain into a third provider call -- see
        # _MAX_PLAN_ATTEMPTS below.
        raw = _strip_fences(response.content)
        data: dict[str, Any] | None = None
        if raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "QueryPlanner returned malformed JSON; retrying once "
                    "(final allowed attempt). source_term=%r model=%s error=%s",
                    source_term,
                    getattr(self._provider, "model", None),
                    exc,
                )
                response = self._complete_plan_timed(
                    messages,
                    temperature=temperature,
                    extra_kwargs=extra_kwargs,
                    timing_sink=timing_sink,
                    usage_sink=usage_sink,
                )
                raw = _strip_fences(response.content)
        elif provider_uses_reasoning_model(self._provider):
            logger.warning(
                "QueryPlanner received empty response from reasoning model; "
                "retrying once with a larger completion budget (final allowed "
                "attempt). source_term=%r model=%s",
                source_term,
                getattr(self._provider, "model", None),
            )
            response = self._complete_plan_timed(
                messages,
                temperature=temperature,
                extra_kwargs=extra_kwargs,
                timing_sink=timing_sink,
                usage_sink=usage_sink,
                max_tokens=REASONING_EMPTY_RESPONSE_RETRY_TOKENS,
                min_completion_tokens=REASONING_EMPTY_RESPONSE_RETRY_TOKENS,
            )
            raw = _strip_fences(response.content)
        # else: empty content and not a reasoning model -- no retry available;
        # data stays None and the final parse below raises immediately.

        if data is None:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise QueryPlanningError(
                    f"LLM returned malformed JSON for source_term={source_term!r}: {exc}\n"
                    f"Response (first 500 chars): {response.content[:500]!r}"
                ) from exc

        return self._build_plan(
            data,
            source_term,
            source_label,
            source_description_norm,
            source_type_norm,
            target_norm,
            allowed_norm,
            mode,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _complete_plan_timed(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        extra_kwargs: dict[str, Any],
        timing_sink: dict[str, float] | None,
        usage_sink: dict[str, Any] | None,
        max_tokens: int = _PLAN_MAX_TOKENS,
        min_completion_tokens: int | None = None,
    ) -> CompletionResponse:
        """Call the provider once, accumulating timing/usage across attempts.

        Mirrors LLMReranker._complete_rerank_timed: timing_sink and usage_sink
        accumulate across the normal call and the empty-response retry (if
        any), so both billed calls are reflected in QueryPlanner's reported
        provider time and token usage rather than only the last attempt.
        """
        kwargs = dict(extra_kwargs)
        if min_completion_tokens is not None:
            kwargs["min_completion_tokens"] = min_completion_tokens
        provider_start = time.monotonic()
        try:
            response = self._provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        finally:
            if timing_sink is not None:
                timing_sink["query_planning_provider_ms"] = (
                    timing_sink.get("query_planning_provider_ms", 0.0)
                    + (time.monotonic() - provider_start) * 1000
                )
        if usage_sink is not None:
            usage_sink["query_planner"] = accumulate_usage(
                usage_sink.get("query_planner"),
                extract_completion_usage(response),
            )
        return response

    def _build_messages(
        self,
        source_term: str,
        source_label: str | None,
        source_description: str | None,
        source_type: str | None,
        clinical_area: str | None,
        target_ontology: str | None,
        allowed_target_ontologies: list[str] | None,
        mode: RetrievalMode,
    ) -> list[ChatMessage]:
        label_section = f"- Label: {source_label}\n" if source_label else ""
        description_section = (
            f"- Source description: {source_description}\n" if source_description else ""
        )
        source_type_section = f"- Source data type: {source_type}\n" if source_type else ""
        area_section = f"- Clinical area: {clinical_area}\n" if clinical_area else ""
        if target_ontology:
            ontology_section = f"- Target ontology (HARD CONSTRAINT): {target_ontology}\n"
        elif allowed_target_ontologies:
            ontology_section = (
                "- Allowed target ontologies (HARD ALLOW-LIST): "
                f"{', '.join(allowed_target_ontologies)}\n"
            )
        else:
            ontology_section = ""

        content = self._prompt_template.format(
            source_term=source_term,
            optional_label_section=label_section,
            optional_source_description_section=description_section,
            optional_source_type_section=source_type_section,
            optional_clinical_area_section=area_section,
            optional_target_ontology_section=ontology_section,
        )

        return [
            ChatMessage(
                role="system",
                content=(
                    "You are a biomedical retrieval planning assistant. Return only valid JSON."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]

    def _build_plan(
        self,
        data: dict[str, Any],
        source_term: str,
        source_label: str | None,
        source_description: str | None,
        source_type: str | None,
        target_ontology: str | None,
        allowed_target_ontologies: list[str] | None,
        mode: RetrievalMode,
    ) -> QueryPlan:
        # ── Extract LLM-provided fields with safe type coercion ───────────────

        normalized_term: str = str(data.get("normalized_term") or "").strip() or source_term.strip()

        raw_queries = data.get("expanded_queries")
        if isinstance(raw_queries, list):
            expanded_queries = [str(q).strip() for q in raw_queries if q and str(q).strip()]
        elif isinstance(raw_queries, str) and raw_queries.strip():
            expanded_queries = [raw_queries.strip()]
        else:
            expanded_queries = []

        inferred_meaning: str | None = data.get("inferred_meaning") or None
        semantic_type: str | None = data.get("semantic_type") or None

        raw_onto = data.get("candidate_ontologies")
        if isinstance(raw_onto, list):
            candidate_ontologies = _normalize_ontology_list(raw_onto) or []
        elif isinstance(raw_onto, str) and raw_onto.strip():
            candidate_ontologies = _normalize_ontology_list([raw_onto]) or []
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
        elif allowed_target_ontologies:
            allowed_set = set(allowed_target_ontologies)
            candidate_ontologies = [
                ontology for ontology in candidate_ontologies if ontology in allowed_set
            ]
            if preferred_ontology not in allowed_set:
                preferred_ontology = candidate_ontologies[0] if candidate_ontologies else None
            if not candidate_ontologies:
                candidate_ontologies = list(allowed_target_ontologies)

        if not expanded_queries:
            expanded_queries = [source_label.strip() if source_label else source_term.strip()]

        disabled_reason: str | None = None
        if mode == RetrievalMode.DISABLED:
            disabled_reason = "Retrieval disabled by caller"

        return QueryPlan(
            original_term=source_term,
            original_label=source_label,
            source_description=source_description,
            source_type=source_type,
            normalized_term=normalized_term,
            expanded_queries=expanded_queries,
            inferred_meaning=inferred_meaning,
            semantic_type=semantic_type,
            candidate_ontologies=candidate_ontologies,
            preferred_ontology=preferred_ontology,
            allowed_target_ontologies=allowed_target_ontologies,
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


def _openai_reasoning_kwargs(provider: BaseLLMProvider) -> dict[str, str]:
    if getattr(provider, "provider_name", "") != "openai":
        return {}
    effort = openai_reasoning_effort_for_model(str(getattr(provider, "model", "")))
    return {"reasoning_effort": effort} if effort else {}


def _normalize_optional_text(value: str | None, *, field_name: str) -> str | None:
    """Normalize optional caller-supplied source metadata for prompt context."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    text = value.strip()
    return text or None


def _normalize_ontology_list(ontologies: list[Any] | None) -> list[str] | None:
    if not ontologies:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for ontology in ontologies:
        text = str(ontology or "").upper().strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)

    return normalized or None
