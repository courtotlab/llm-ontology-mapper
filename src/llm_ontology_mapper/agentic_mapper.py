"""
AgenticMapper — tool-calling agent loop for ontology term mapping.

The loop drives the LLM with four SearchTools-backed MCP JSON-Schema tools,
grounds the final answer against accumulated tool results, and returns the
library's standard MappingResult with logic_type=AGENTIC.

Design invariants
─────────────────
• Grounding guarantee: the returned code MUST appear in an accumulated tool
  result (normalized form).  One corrective retry is allowed; on the second
  failure logic_type falls back to LLM with a note.
• Offline-testable: all HTTP lives in SearchTools; the loop only touches
  provider.complete_with_tools() and SearchTools.search_*() — both are
  mockable without any network.
• Behaviour-preserving: this module is wholly additive.  It never touches
  OntologyMapper, OntologyRetriever, complete(), or SapBERT.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable

from llm_ontology_mapper.models import (
    AlternativeMapping,
    LogicType,
    MappingResult,
)
from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage
from llm_ontology_mapper.search_tools import SearchTools

logger = logging.getLogger(__name__)


def _agentic_trace(event: str, **details: Any) -> None:
    """Emit structured agent-loop diagnostics only when explicitly enabled."""
    if os.environ.get("AGENTIC_DEBUG") != "1":
        return
    print(
        "[AGENTIC_DEBUG] "
        + json.dumps({"event": event, **details}, default=str, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Loop constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_ITERATIONS: int   = 6
TOTAL_TIMEOUT_S: float = 90.0
TOP_K_PER_TOOL: int   = 5

_OLS_CANONICAL_ONTOLOGIES: tuple[str, ...] = (
    "HP", "MONDO", "NCIT", "SNOMED", "UBERON", "CHEBI", "GO", "DOID",
    "MESH", "UO",
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP JSON-Schema tool definitions
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_SEARCH_OLS: dict[str, Any] = {
    "name": "search_ols",
    "description": (
        "Search EBI OLS4 for an ontology term. "
        "Use for phenotypes/symptoms (ontology=HP), diseases/disorders (ontology=MONDO or NCIT), "
        "anatomy (ontology=UBERON), chemicals (ontology=CHEBI), or clinical terms (ontology=SNOMED)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query":    {"type": "string",
                         "description": "Term to search for (expanded, not abbreviated)"},
            "ontology": {"type": "string",
                         "enum": list(_OLS_CANONICAL_ONTOLOGIES),
                         "description": "Canonical ontology short-name"},
        },
        "required": ["query", "ontology"],
    },
}

_TOOL_SEARCH_LOINC: dict[str, Any] = {
    "name": "search_loinc",
    "description": "Search LOINC for laboratory tests, clinical measurements, and vital signs.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Measurement or test name to search for"},
        },
        "required": ["query"],
    },
}

_TOOL_SEARCH_RXNORM: dict[str, Any] = {
    "name": "search_rxnorm",
    "description": "Search RxNorm for drug and medication names.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Drug or medication name to search for"},
        },
        "required": ["query"],
    },
}

_TOOL_SEARCH_ICD10: dict[str, Any] = {
    "name": "search_icd10",
    "description": "Search ICD-10-CM for disease diagnoses and billing codes.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Disease, diagnosis, or condition to search for"},
        },
        "required": ["query"],
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    _TOOL_SEARCH_OLS,
    _TOOL_SEARCH_LOINC,
    _TOOL_SEARCH_RXNORM,
    _TOOL_SEARCH_ICD10,
]

# OLS target aliases are separate from canonical values exposed to models.
_OLS_TARGET_ALIASES: dict[str, str] = {
    **{ontology: ontology for ontology in _OLS_CANONICAL_ONTOLOGIES},
    "HPO": "HP",
    "SNOMEDCT": "SNOMED",
    "SNOMED-CT": "SNOMED",
}
_OLS_ONTOLOGIES: frozenset[str] = frozenset(_OLS_TARGET_ALIASES)

# ─────────────────────────────────────────────────────────────────────────────
# Code normalisation (module-level, also used by the A4 test suite)
# ─────────────────────────────────────────────────────────────────────────────

_PREFIX_SEP_RE = re.compile(r'^([A-Za-z][A-Za-z0-9]*)([_/])([A-Za-z0-9])')


def normalize_code(raw: str, source_prefix: str | None = None) -> str:
    """
    Return a canonical ``UPPERCASEPREFIX:localid`` form.

    Rules applied in order:
    1. Strip surrounding whitespace.
    2. Replace the first ``_`` or ``/`` separator between a letter-prefix and
       the local-id with ``:``.  (e.g. ``HP_0002110`` → ``HP:0002110``)
    3. Uppercase the prefix; preserve the local-id case.
    4. If no colon is present after the above and *source_prefix* is given,
       prepend it (handles bare numeric codes such as ``8480-6`` from LOINC).
    """
    raw = raw.strip()
    # e.g. HP_0002110 → HP:0002110  (captures first separator only)
    raw = _PREFIX_SEP_RE.sub(lambda m: f"{m.group(1)}:{m.group(3)}", raw, count=1)
    if ":" in raw:
        prefix, local = raw.split(":", 1)
        return f"{prefix.strip().upper()}:{local.strip()}"
    if source_prefix:
        return f"{source_prefix.strip().upper()}:{raw}"
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_PREFIX_TO_ONTOLOGY: dict[str, str] = {
    "HP": "HPO", "HPO": "HPO",
    "MONDO": "MONDO", "NCIT": "NCIT",
    "LOINC": "LOINC",
    "ICD10": "ICD10", "ICD10CM": "ICD10",
    "RXNORM": "RXNORM", "RXCUI": "RXNORM",
    "SNOMEDCT": "SNOMED-CT", "SNOMED": "SNOMED-CT",
    "UBERON": "UBERON", "CHEBI": "CHEBI", "GO": "GO",
}


def _code_to_ontology(code: str) -> str:
    """Infer the ontology name from a CURIE prefix."""
    if ":" not in code:
        return "UNKNOWN"
    prefix = code.split(":", 1)[0].upper()
    return _PREFIX_TO_ONTOLOGY.get(prefix, prefix)


def _ontology_to_prefix(ontology: str) -> str:
    """Return the expected CURIE prefix for a named ontology."""
    _MAP: dict[str, str] = {
        "HPO": "HP",   "HP": "HP",
        "MONDO": "MONDO", "NCIT": "NCIT",
        "LOINC": "LOINC",
        "ICD10": "ICD10", "ICD10CM": "ICD10",
        "RXNORM": "RXNORM",
        "SNOMED": "SNOMEDCT", "SNOMED-CT": "SNOMEDCT", "SNOMEDCT": "SNOMEDCT",
        "UBERON": "UBERON", "CHEBI": "CHEBI",
    }
    return _MAP.get(ontology.upper(), ontology.upper())


def _resolve_ols_ontology(target_ontology: str | None) -> str | None:
    """Resolve a hard target constraint to the canonical OLS API identifier."""
    if not isinstance(target_ontology, str):
        return None
    return _OLS_TARGET_ALIASES.get(target_ontology.strip().upper())


def _is_meaningful_query(value: Any) -> bool:
    """Return whether *value* contains usable search text."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and any(character.isalnum() for character in value)
    )


def _extract_json(text: str) -> str:
    """Pull the first JSON object from a possibly-prose model response."""
    # Remove <think>…</think> blocks (some reasoning models emit these)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return obj.group(0)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_ROUTING_BLOCK = """\
SOURCE ROUTING — expand abbreviations first, then choose the right tool:
  measurement / vital sign / lab test  → search_loinc
  phenotype / symptom / clinical sign  → search_ols with ontology=HP
  disease / disorder / syndrome        → search_ols with ontology=MONDO or NCIT
  drug / medication                    → search_rxnorm
  diagnosis / billing / ICD code       → search_icd10
  anatomy / body part                  → search_ols with ontology=UBERON
  chemical / compound                  → search_ols with ontology=CHEBI"""


def _build_system_prompt(
    target_ontology: str | None,
    clinical_area: str | None,
) -> str:
    lines: list[str] = [
        "You are a biomedical ontology mapping expert.",
        "",
        "PROCESS:",
        "1. Expand any abbreviations in the term before searching",
        "   (e.g. 'sys_bp' → 'systolic blood pressure', 'htn' → 'hypertension').",
        "2. Call search tools to retrieve candidate codes.",
        "3. When you have sufficient candidates, return your final answer as JSON.",
        "",
        _SOURCE_ROUTING_BLOCK,
        'TOOL ARGUMENT EXAMPLE: search_ols({"query":"cough","ontology":"HP"})',
    ]

    if clinical_area:
        lines += [
            "",
            f"CLINICAL AREA HINT: This term belongs to the '{clinical_area}' domain.",
            "Prefer the ontology most appropriate for that domain.",
        ]

    if target_ontology:
        prefix = _ontology_to_prefix(target_ontology)
        lines += [
            "",
            f"TARGET ONTOLOGY (REQUIRED): You MUST return a code from {target_ontology!r}",
            f"(CURIE prefix: {prefix!r}). Only call tools that can return {target_ontology} codes.",
        ]

    lines += [
        "",
        "FINAL ANSWER FORMAT — respond with ONLY this JSON when you are ready:",
        "{",
        '  "code":       "<CURIE copied exactly from a tool result, e.g. LOINC:8480-6>",',
        '  "term":       "<official label>",',
        '  "ontology":   "<prefix, e.g. LOINC>",',
        '  "confidence": <float 0.0-1.0>,',
        '  "notes":      "<brief reasoning>"',
        "}",
        "",
        "CRITICAL: The 'code' field MUST be copied verbatim from a tool result."
        " Do not invent or modify codes.",
    ]
    return "\n".join(lines)


def _build_user_prompt(source_term: str, source_label: str | None) -> str:
    parts = [f"Map this clinical term to an ontology code:", f"Term: {source_term}"]
    if source_label:
        parts.append(f"Label: {source_label}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# AgenticMapper
# ─────────────────────────────────────────────────────────────────────────────


class AgenticMapper:
    """
    Tool-calling agent loop for ontology term mapping.

    The mapper drives an LLM with four search tools (OLS, LOINC, RxNorm,
    ICD-10), accumulates every code returned by those tools, and grounds the
    model's final answer against that accumulated set before returning.

    Args:
        provider:     A ``BaseLLMProvider`` that implements
                      ``complete_with_tools()``.
        search_tools: Configured ``SearchTools`` instance (carries base URLs,
                      LOINC credentials, timeouts).
        api_timeout:  Per-request HTTP timeout (seconds); defaults to
                      ``search_tools.api_timeout``.

    Example::

        from llm_ontology_mapper.agentic_mapper import AgenticMapper
        from llm_ontology_mapper.search_tools import SearchTools
        from llm_ontology_mapper.providers import OpenAIProvider

        mapper = AgenticMapper(
            provider=OpenAIProvider(model="gpt-4o"),
            search_tools=SearchTools(),
        )
        result = mapper.map(
            "sys_bp",
            source_label="Systolic blood pressure",
            target_ontology="LOINC",
        )
        # result.target_code  →  "LOINC:8480-6"
        # result.logic_type   →  LogicType.AGENTIC
    """

    MAX_ITERATIONS: int    = MAX_ITERATIONS
    TOTAL_TIMEOUT_S: float = TOTAL_TIMEOUT_S

    def __init__(
        self,
        provider: BaseLLMProvider,
        search_tools: SearchTools,
        api_timeout: int | None = None,
    ) -> None:
        self._provider     = provider
        self._search_tools = search_tools
        self._api_timeout  = (
            api_timeout if api_timeout is not None else search_tools.api_timeout
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def map(
        self,
        source_term: str,
        source_label: str | None = None,
        *,
        target_ontology: str | None = None,
        clinical_area: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> MappingResult:
        """
        Map *source_term* to an ontology code via an agentic tool-calling loop.

        Args:
            source_term:     Clinical field name or label to map.
            source_label:    Optional longer description / question text.
            target_ontology: Hard constraint — only codes with this prefix are
                             accepted (e.g. ``"LOINC"``, ``"HPO"``).
            clinical_area:   Soft routing hint for the system prompt
                             (e.g. ``"measurement"``, ``"phenotype"``).
            cancel_check:    Optional callable; when it returns ``True`` the
                             loop exits immediately returning UNMAPPED.

        Returns:
            :class:`MappingResult` with ``logic_type=AGENTIC`` on a grounded
            success, ``logic_type=LLM`` on a grounding-failure fallback, or
            ``target_code="UNMAPPED"`` on loop exhaustion or cancellation.
        """
        _cancel = cancel_check or (lambda: False)
        t_start = time.monotonic()

        bound_ols_ontology = _resolve_ols_ontology(target_ontology)
        tools    = self._select_tools(target_ontology, bound_ols_ontology)
        _agentic_trace(
            "mapping_start",
            target_ontology=target_ontology,
            target_bound=bound_ols_ontology is not None,
            pinned_ols_ontology=bound_ols_ontology,
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_build_system_prompt(target_ontology, clinical_area)),
            ChatMessage(role="user",   content=_build_user_prompt(source_term, source_label)),
        ]

        # norm_code → raw result dict from the tool
        accumulated: dict[str, dict[str, Any]] = {}
        attempted_calls: dict[tuple[str, str], list[dict[str, Any]]] = {}
        fallback_query = self._preferred_fallback_query(source_label, source_term)
        target_bound_query_fallback = (
            fallback_query
            if target_ontology is not None and len(tools) == 1
            else None
        )
        non_progress   = 0   # consecutive turns with no tool calls AND no valid JSON
        grounding_tried = False
        force_finalize_next = False
        duplicate_productive_seen = False

        for iteration in range(self.MAX_ITERATIONS):
            iteration_tools = [] if force_finalize_next else tools
            force_finalize_next = False
            search_ols_tool = next(
                (tool for tool in iteration_tools if tool.get("name") == "search_ols"),
                None,
            )
            search_ols_schema = (
                search_ols_tool.get("inputSchema", {})
                if search_ols_tool is not None
                else {}
            )
            _agentic_trace(
                "iteration_start",
                iteration=iteration,
                exposed_tools=[tool.get("name") for tool in iteration_tools],
                search_ols_schema_properties=list(
                    search_ols_schema.get("properties", {}).keys()
                ),
                search_ols_schema_required=search_ols_schema.get("required", []),
                target_bound=bound_ols_ontology is not None,
                pinned_ols_ontology=bound_ols_ontology,
            )

            if time.monotonic() - t_start > self.TOTAL_TIMEOUT_S:
                _agentic_trace(
                    "loop_exit",
                    iteration=iteration,
                    reason="total_timeout",
                )
                return self._unmapped(
                    source_term, source_label,
                    f"Total timeout ({self.TOTAL_TIMEOUT_S}s) exceeded "
                    f"after {iteration} iterations",
                )

            if _cancel():
                _agentic_trace(
                    "loop_exit",
                    iteration=iteration,
                    reason="cancelled",
                )
                return self._unmapped(source_term, source_label, "Cancelled by cancel_check")

            # ── LLM turn ─────────────────────────────────────────────────────
            try:
                text, tool_calls = self._provider.complete_with_tools(
                    messages,
                    iteration_tools,
                )
            except Exception as exc:
                _agentic_trace(
                    "provider_error",
                    iteration=iteration,
                    error=repr(exc),
                )
                logger.warning(
                    "AgenticMapper LLM error on iteration %d: %s", iteration, exc
                )
                return self._unmapped(source_term, source_label, f"LLM error: {exc}")

            _agentic_trace(
                "provider_response",
                iteration=iteration,
                text=text,
                tool_calls=[
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in tool_calls
                ],
                attempted_final_answer=bool(text and not tool_calls),
            )

            # ── Branch A: model issued tool calls ────────────────────────────
            if tool_calls:
                non_progress = 0
                result_blocks: list[str] = []

                for tc in tool_calls:
                    error: str | None = None
                    repeated_call = False
                    formatted_arguments = tc.arguments
                    try:
                        dispatch_arguments = self._prepare_tool_arguments(
                            tc.name,
                            tc.arguments,
                            bound_ols_ontology=bound_ols_ontology,
                            fallback_query=target_bound_query_fallback,
                        )
                        formatted_arguments = dispatch_arguments
                        call_key = self._normalized_tool_call_key(
                            tc.name,
                            dispatch_arguments,
                        )
                        if call_key in attempted_calls:
                            repeated_call = True
                            results = attempted_calls[call_key]
                            if results:
                                duplicate_productive_seen = True
                                force_finalize_next = True
                                error = (
                                    f"Duplicate {tc.name} call: this query already "
                                    "returned candidates. Do not search again. Select "
                                    "one of the returned codes and produce the final "
                                    "grounded JSON."
                                )
                                trace_event = "duplicate_productive_tool_call"
                            else:
                                error = (
                                    f"Repeated {tc.name} call: this query already "
                                    "returned no results. Do not repeat the same "
                                    "search. Use a different meaningful query."
                                )
                                trace_event = "duplicate_zero_result_tool_call"
                            _agentic_trace(
                                trace_event,
                                iteration=iteration,
                                tool=tc.name,
                                dispatch_arguments=dispatch_arguments,
                                result_count=len(results),
                            )
                        else:
                            try:
                                results = self._execute_tool(
                                    tc.name,
                                    dispatch_arguments,
                                    bound_ols_ontology=bound_ols_ontology,
                                )
                            except Exception as primary_exc:
                                attempted_calls[call_key] = []
                                fallback_arguments = self._fallback_tool_arguments(
                                    dispatch_arguments,
                                    target_bound_query_fallback,
                                )
                                if fallback_arguments is None:
                                    raise
                                _agentic_trace(
                                    "query_fallback",
                                    iteration=iteration,
                                    tool=tc.name,
                                    original_query=dispatch_arguments["query"],
                                    replacement_query=fallback_arguments["query"],
                                    reason="query_error_fallback",
                                    error=repr(primary_exc),
                                )
                                dispatch_arguments = fallback_arguments
                                formatted_arguments = fallback_arguments
                                call_key = self._normalized_tool_call_key(
                                    tc.name,
                                    dispatch_arguments,
                                )
                                if call_key in attempted_calls:
                                    results = attempted_calls[call_key]
                                else:
                                    results = self._execute_tool(
                                        tc.name,
                                        dispatch_arguments,
                                        bound_ols_ontology=bound_ols_ontology,
                                    )
                                    attempted_calls[call_key] = results
                            else:
                                attempted_calls[call_key] = results
                                fallback_arguments = self._fallback_tool_arguments(
                                    dispatch_arguments,
                                    target_bound_query_fallback,
                                )
                                if not results and fallback_arguments is not None:
                                    _agentic_trace(
                                        "query_fallback",
                                        iteration=iteration,
                                        tool=tc.name,
                                        original_query=dispatch_arguments["query"],
                                        replacement_query=fallback_arguments["query"],
                                        reason="zero_result_fallback",
                                    )
                                    dispatch_arguments = fallback_arguments
                                    formatted_arguments = fallback_arguments
                                    call_key = self._normalized_tool_call_key(
                                        tc.name,
                                        dispatch_arguments,
                                    )
                                    if call_key in attempted_calls:
                                        results = attempted_calls[call_key]
                                    else:
                                        results = self._execute_tool(
                                            tc.name,
                                            dispatch_arguments,
                                            bound_ols_ontology=bound_ols_ontology,
                                        )
                                        attempted_calls[call_key] = results
                    except ValueError as exc:
                        results = []
                        error = str(exc)
                        _agentic_trace(
                            "tool_argument_rejected",
                            iteration=iteration,
                            tool=tc.name,
                            arguments=tc.arguments,
                            reason=error,
                        )
                    except Exception as exc:
                        logger.warning("Tool %r raised: %s", tc.name, exc)
                        results = []
                        error = (
                            f"Tool execution failed: {exc}. "
                            "Retry with valid arguments or another search."
                        )
                        _agentic_trace(
                            "tool_execution_error",
                            iteration=iteration,
                            tool=tc.name,
                            arguments=tc.arguments,
                            reason=repr(exc),
                        )

                    for r in results:
                        raw = r.get("code", "")
                        if raw:
                            accumulated[normalize_code(raw)] = r

                    _agentic_trace(
                        "tool_results",
                        iteration=iteration,
                        tool=tc.name,
                        result_count=len(results),
                        first_results=[
                            {
                                "code": result.get("code"),
                                "term": result.get("term"),
                                "score": result.get("score"),
                            }
                            for result in results[:5]
                        ],
                    )
                    if repeated_call and results:
                        formatted_result = (
                            f"{tc.name}: error={error}\n"
                            + _format_tool_result(
                                tc.name,
                                formatted_arguments,
                                results,
                            )
                        )
                    else:
                        formatted_result = _format_tool_result(
                            tc.name,
                            formatted_arguments,
                            results,
                            error=error,
                        )
                    _agentic_trace(
                        "formatted_tool_result",
                        iteration=iteration,
                        tool=tc.name,
                        text=formatted_result,
                    )
                    result_blocks.append(formatted_result)

                if text:
                    messages.append(ChatMessage(role="assistant", content=text))

                messages.append(ChatMessage(
                    role="user",
                    content=(
                        "Tool results:\n\n"
                        + "\n\n".join(result_blocks)
                        + "\n\nNow select the best code from the results above, "
                          "or call more tools if needed. "
                          "When ready, return your final answer as JSON."
                    ),
                ))
                continue

            # ── Branch B: model returned text ────────────────────────────────
            if text:
                _agentic_trace(
                    "final_answer_attempt",
                    iteration=iteration,
                    text=text,
                )
                json_str = _extract_json(text)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    _agentic_trace(
                        "final_answer_rejected",
                        iteration=iteration,
                        reason="json_parse_error",
                        error=str(exc),
                        extracted_text=json_str,
                    )
                    non_progress += 1
                    if non_progress >= 2:
                        _agentic_trace(
                            "loop_exit",
                            iteration=iteration,
                            reason="two_unparseable_responses",
                        )
                        return self._unmapped(
                            source_term, source_label,
                            "Two consecutive non-progress turns (unparseable response)",
                        )
                    messages.append(ChatMessage(role="assistant", content=text))
                    messages.append(ChatMessage(
                        role="user",
                        content=(
                            "Your response could not be parsed as JSON. "
                            "Please return ONLY a valid JSON object with keys: "
                            "code, term, ontology, confidence, notes."
                        ),
                    ))
                    continue

                raw_code = (data.get("code") or "").strip()
                if not raw_code or raw_code.upper() == "UNMAPPED":
                    _agentic_trace(
                        "final_answer_rejected",
                        iteration=iteration,
                        reason="missing_or_unmapped_code",
                        parsed_answer=data,
                    )
                    non_progress += 1
                    if non_progress >= 2:
                        _agentic_trace(
                            "loop_exit",
                            iteration=iteration,
                            reason="two_answers_without_valid_code",
                        )
                        return self._unmapped(
                            source_term, source_label,
                            "Two consecutive non-progress turns (no valid code in response)",
                        )
                    messages.append(ChatMessage(role="assistant", content=text))
                    messages.append(ChatMessage(
                        role="user",
                        content=(
                            "Your response did not include a valid code. "
                            "Please call a search tool first, then return a code "
                            "copied exactly from the results."
                        ),
                    ))
                    continue

                # ── Grounding check ──────────────────────────────────────────
                non_progress = 0
                ok, reason = self._ground(raw_code, target_ontology, accumulated)
                _agentic_trace(
                    "grounding_check",
                    iteration=iteration,
                    code=raw_code,
                    accepted=ok,
                    rejection_reason=reason or None,
                    accumulated_codes=sorted(accumulated.keys()),
                )

                if ok:
                    _agentic_trace(
                        "loop_exit",
                        iteration=iteration,
                        reason="grounded_answer",
                        code=raw_code,
                    )
                    return self._build_result(source_term, source_label, data, accumulated)

                if not grounding_tried:
                    grounding_tried = True
                    avail = ", ".join(sorted(accumulated.keys())[:10]) or "(none yet)"
                    messages.append(ChatMessage(role="assistant", content=text))
                    messages.append(ChatMessage(
                        role="user",
                        content=(
                            f"The code '{raw_code}' was not found in any tool result "
                            f"({reason}). "
                            f"Available codes from your searches: {avail}. "
                            "Please choose a code that appeared exactly in a tool result "
                            "and return your final answer as JSON."
                        ),
                    ))
                    continue

                # Second grounding failure → LLM fallback
                _agentic_trace(
                    "loop_exit",
                    iteration=iteration,
                    reason="second_grounding_failure",
                    code=raw_code,
                    rejection_reason=reason,
                )
                return self._build_result(
                    source_term, source_label, data, accumulated,
                    logic_type=LogicType.LLM,
                    extra_note="Grounding check failed",
                )

            # ── Branch C: empty response (neither text nor tool calls) ────────
            _agentic_trace(
                "empty_provider_response",
                iteration=iteration,
                consecutive_non_progress=non_progress + 1,
            )
            non_progress += 1
            if non_progress >= 2:
                _agentic_trace(
                    "loop_exit",
                    iteration=iteration,
                    reason="two_empty_responses",
                )
                return self._unmapped(
                    source_term, source_label,
                    "Two consecutive non-progress turns (empty LLM response)",
                )
            messages.append(ChatMessage(
                role="user",
                content=(
                    "Please either call a search tool or return your final answer as JSON."
                ),
            ))

        _agentic_trace(
            "loop_exit",
            iteration=self.MAX_ITERATIONS,
            reason="max_iterations",
            accumulated_codes=sorted(accumulated.keys()),
        )
        fallback = self._grounded_retrieval_fallback(
            source_term,
            source_label,
            target_ontology,
            accumulated,
            duplicate_productive_seen=duplicate_productive_seen,
        )
        if fallback is not None:
            _agentic_trace(
                "loop_exit",
                iteration=self.MAX_ITERATIONS,
                reason="grounded_retrieval_fallback",
                code=fallback.target_code,
            )
            return fallback
        return self._unmapped(
            source_term, source_label,
            f"Max iterations ({self.MAX_ITERATIONS}) reached without a grounded answer",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_tools(
        self,
        target_ontology: str | None,
        bound_ols_ontology: str | None = None,
    ) -> list[dict[str, Any]]:
        """Restrict the tool set based on target_ontology (hard filter)."""
        if target_ontology is None:
            return list(ALL_TOOLS)
        onto_upper = target_ontology.upper()
        if bound_ols_ontology or onto_upper in _OLS_ONTOLOGIES:
            tool = copy.deepcopy(_TOOL_SEARCH_OLS)
            schema = tool["inputSchema"]
            schema["properties"].pop("ontology")
            schema["required"].remove("ontology")
            return [tool]
        if onto_upper == "LOINC":
            return [_TOOL_SEARCH_LOINC]
        if onto_upper in ("RXNORM", "RXCUI"):
            return [_TOOL_SEARCH_RXNORM]
        if onto_upper in ("ICD10", "ICD10CM"):
            return [_TOOL_SEARCH_ICD10]
        return list(ALL_TOOLS)   # unknown ontology — offer all

    def _execute_tool(
        self,
        name: str,
        arguments: Any,
        *,
        bound_ols_ontology: str | None = None,
    ) -> list[dict[str, Any]]:
        prepared = self._prepare_tool_arguments(
            name,
            arguments,
            bound_ols_ontology=bound_ols_ontology,
        )
        query = prepared["query"]

        if name == "search_ols":
            ontology = prepared["ontology"]
            _agentic_trace(
                "tool_dispatch",
                tool=name,
                target_bound=bound_ols_ontology is not None,
                pinned_ols_ontology=bound_ols_ontology,
                query=query,
                ontology=ontology,
            )
            return self._search_tools.search_ols(
                query,
                ontology,
                top_k=TOP_K_PER_TOOL,
            )
        if name == "search_loinc":
            _agentic_trace("tool_dispatch", tool=name, query=query)
            return self._search_tools.search_loinc(
                query, top_k=TOP_K_PER_TOOL
            )
        if name == "search_rxnorm":
            _agentic_trace("tool_dispatch", tool=name, query=query)
            return self._search_tools.search_rxnorm(
                query, top_k=TOP_K_PER_TOOL
            )
        if name == "search_icd10":
            _agentic_trace("tool_dispatch", tool=name, query=query)
            return self._search_tools.search_icd10(
                query, top_k=TOP_K_PER_TOOL
            )
        raise ValueError(
            f"Unknown tool {name!r}. Retry using one of the provided tools."
        )

    @staticmethod
    def _prepare_tool_arguments(
        name: str,
        arguments: Any,
        *,
        bound_ols_ontology: str | None = None,
        fallback_query: str | None = None,
    ) -> dict[str, str]:
        if not isinstance(arguments, dict):
            raise ValueError(
                f"Invalid arguments for {name!r}: expected an object; "
                f"you sent {arguments!r}. Retry with valid values."
            )

        original_query = arguments.get("query")
        if _is_meaningful_query(original_query):
            query = original_query.strip()
        elif _is_meaningful_query(fallback_query):
            query = fallback_query.strip()
            _agentic_trace(
                "query_fallback",
                tool=name,
                original_query=original_query,
                replacement_query=query,
                reason="invalid_query_fallback",
            )
        else:
            raise ValueError(
                f"Invalid 'query': expected meaningful plain search text; "
                f"you sent {original_query!r}. Retry with a valid value."
            )

        if name == "search_ols":
            ontology = bound_ols_ontology
            if ontology is None:
                supplied = arguments.get("ontology")
                if (
                    not isinstance(supplied, str)
                    or supplied.strip() not in _OLS_CANONICAL_ONTOLOGIES
                ):
                    allowed = ", ".join(_OLS_CANONICAL_ONTOLOGIES)
                    raise ValueError(
                        f"Invalid 'ontology': must be one of [{allowed}] as a "
                        f"plain string; you sent {supplied!r}. "
                        "Retry with a valid value."
                    )
                ontology = supplied.strip()
            return {"query": query, "ontology": ontology}
        if name in {"search_loinc", "search_rxnorm", "search_icd10"}:
            return {"query": query}
        raise ValueError(
            f"Unknown tool {name!r}. Retry using one of the provided tools."
        )

    @staticmethod
    def _preferred_fallback_query(
        source_label: str | None,
        source_term: str,
    ) -> str | None:
        for candidate in (source_label, source_term):
            if _is_meaningful_query(candidate):
                return candidate.strip()
        return None

    @staticmethod
    def _fallback_tool_arguments(
        arguments: dict[str, str],
        fallback_query: str | None,
    ) -> dict[str, str] | None:
        if not _is_meaningful_query(fallback_query):
            return None
        if " ".join(arguments["query"].split()).casefold() == (
            " ".join(fallback_query.split()).casefold()
        ):
            return None
        return {**arguments, "query": fallback_query.strip()}

    @staticmethod
    def _normalized_tool_call_key(
        name: str,
        arguments: dict[str, str],
    ) -> tuple[str, str]:
        normalized_arguments = {
            key: (
                " ".join(value.split()).casefold()
                if key == "query"
                else value.strip().upper()
            )
            for key, value in arguments.items()
        }
        return name, json.dumps(normalized_arguments, sort_keys=True)

    @staticmethod
    def _ground(
        raw_code: str,
        target_ontology: str | None,
        accumulated: dict[str, dict[str, Any]],
    ) -> tuple[bool, str]:
        norm = normalize_code(raw_code)
        if norm not in accumulated:
            return False, f"normalized {norm!r} not in tool results"
        if target_ontology:
            prefix = norm.split(":")[0] if ":" in norm else norm
            expected = _ontology_to_prefix(target_ontology)
            if prefix.upper() != expected.upper():
                return False, (
                    f"prefix {prefix!r} does not match target "
                    f"{target_ontology!r} (expected {expected!r})"
                )
        return True, ""

    def _build_result(
        self,
        source_term: str,
        source_label: str | None,
        data: dict[str, Any],
        accumulated: dict[str, dict[str, Any]],
        logic_type: LogicType = LogicType.AGENTIC,
        extra_note: str | None = None,
    ) -> MappingResult:
        raw_code   = (data.get("code") or "UNMAPPED").strip()
        norm_code  = normalize_code(raw_code)
        # Prefer term from model, fall back to the accumulated tool result
        term       = (data.get("term") or "").strip() or accumulated.get(norm_code, {}).get("term", "")
        confidence = float(data.get("confidence") or 0.5)
        confidence = max(0.0, min(1.0, confidence))
        code_ontology = _code_to_ontology(norm_code).upper()
        ontology = (
            code_ontology
            if code_ontology != "UNKNOWN"
            else (data.get("ontology") or "UNKNOWN").strip().upper()
        )
        base_note  = (data.get("notes") or "").strip()
        note       = "; ".join(filter(None, [extra_note, base_note])) or None

        # Alternatives from accumulated tool results (highest score first,
        # excluding the chosen code, up to 5 entries)
        alts = [
            AlternativeMapping(
                code=r.get("code", ""),
                term=r.get("term", ""),
                ontology=_code_to_ontology(r.get("code", "")).upper(),
                confidence=max(0.0, min(0.9, float(r.get("score", 0.5)))),
                source="agentic",
            )
            for norm, r in sorted(
                accumulated.items(),
                key=lambda kv: kv[1].get("score", 0.0),
                reverse=True,
            )
            if norm != norm_code and r.get("code")
        ][:5]

        return MappingResult(
            source_term=source_term,
            source_label=source_label,
            target_code=norm_code,
            target_term=term or raw_code,
            ontology=ontology,
            confidence=confidence,
            logic_type=logic_type,
            alternatives=alts,
            notes=note,
        )

    def _grounded_retrieval_fallback(
        self,
        source_term: str,
        source_label: str | None,
        target_ontology: str | None,
        accumulated: dict[str, dict[str, Any]],
        *,
        duplicate_productive_seen: bool,
    ) -> MappingResult | None:
        ranked = sorted(
            accumulated.items(),
            key=lambda item: self._candidate_score(item[1]),
            reverse=True,
        )
        grounded = [
            (code, candidate)
            for code, candidate in ranked
            if self._ground(code, target_ontology, accumulated)[0]
        ]
        if not grounded:
            return None

        code, candidate = grounded[0]
        score = self._candidate_score(candidate)
        fallback_note = (
            "Selected from grounded retrieval candidates after the agent "
            "repeated tool calls and did not produce a final answer."
            if duplicate_productive_seen
            else (
                "Selected from grounded retrieval candidates after the agent "
                "did not produce a final answer."
            )
        )
        return self._build_result(
            source_term,
            source_label,
            {
                "code": code,
                "term": candidate.get("term", ""),
                "ontology": _code_to_ontology(code),
                "confidence": min(0.75, score),
                "notes": fallback_note,
            },
            accumulated,
        )

    @staticmethod
    def _candidate_score(candidate: dict[str, Any]) -> float:
        try:
            score = float(candidate.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        return max(0.0, min(1.0, score))

    @staticmethod
    def _unmapped(
        source_term: str,
        source_label: str | None,
        reason: str,
    ) -> MappingResult:
        return MappingResult(
            source_term=source_term,
            source_label=source_label,
            target_code="UNKNOWN:UNMAPPED",  # pre-normalised: avoids model_validator prepending UNKNOWN:
            target_term="UNMAPPED",
            ontology="UNKNOWN",
            confidence=0.0,
            logic_type=LogicType.AGENTIC,
            notes=reason,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool-result formatting (used in the conversation context)
# ─────────────────────────────────────────────────────────────────────────────


def _format_tool_result(
    name: str,
    arguments: Any,
    results: list[dict[str, Any]],
    *,
    error: str | None = None,
) -> str:
    if isinstance(arguments, dict):
        arg_str = ", ".join(f'{k}="{v}"' for k, v in arguments.items())
    else:
        arg_str = repr(arguments)
    if error:
        return f"{name}({arg_str}): error={error}"
    if not results:
        return f"{name}({arg_str}): No results found."
    lines = [f"{name}({arg_str}): {len(results)} result(s):"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i}. code={r.get('code', '?')} | "
            f"term={r.get('term', '?')} | "
            f"score={r.get('score', 0):.3f}"
        )
    return "\n".join(lines)
