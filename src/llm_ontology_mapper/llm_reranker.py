"""
LLMReranker — candidate-grounded reranking via an LLM provider.

Receives a QueryPlan and a list of already-normalized NormalizedCandidate
objects, asks the LLM to select the best candidate or return UNMAPPED, and
returns a RerankDecision.

Hard constraints enforced by Python (LLM cannot override):
  - selected_candidate_id must be in the provided candidate list.
  - selected_code must match the code of the selected candidate.
  - selected_code must be in the candidate code set.
  - alternatives and alternative_codes must resolve to retrieved candidates.
  - target_ontology_constraint is enforced: selected candidate's ontology
    must match when the constraint is present.
  - allowed_target_ontologies is enforced: selected candidate and alternatives
    must belong to the allow-list when present.
  - disabled mode raises immediately — it does not produce candidates.

Phase 5A of the planned grounded mapping pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_ontology_mapper.models import (
    GroundingSource,
    NormalizedCandidate,
    QueryPlan,
    RerankAlternative,
    RerankDecision,
    RetrievalMode,
)
from llm_ontology_mapper.ontology_identity import validate_candidate_identity
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
from llm_ontology_mapper.target_eligibility import candidate_allowed_for_targets

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "assets" / "prompts"
_RERANK_MAX_TOKENS = 512
_REASONING_RERANK_MIN_COMPLETION_TOKENS = 4096
_REASONING_RERANK_RETRY_TOKENS = REASONING_EMPTY_RESPONSE_RETRY_TOKENS
_MAX_RERANK_ATTEMPTS = 2
"""Hard cap on provider calls per rerank(): the normal call plus at most one
recovery attempt (empty-response-from-reasoning-model OR malformed-JSON --
never both, since the second attempt's outcome is never re-triaged)."""


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
        llm_call_config: LLMCallConfig | None = None,
    ) -> None:
        self._provider = provider
        self._llm_call_config = llm_call_config
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
        *,
        timing_sink: dict[str, float] | None = None,
        usage_sink: dict[str, Any] | None = None,
        strict_target_ontology: bool = False,
    ) -> RerankDecision:
        """
        Select the best candidate for the source term or return UNMAPPED.

        Args:
            query_plan:  The planned interpretation of the source term.
            candidates:  Already-normalized and merged candidate objects.
            strict_target_ontology: When True, require canonical native-ontology
                membership only; disables the EFO imported-term provenance
                exception for candidate filtering, selection, and alternatives.

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
        allowed_ontologies = _effective_allowed_target_ontologies(query_plan)
        eligible_candidates = _filter_allowed_candidates(
            candidates, allowed_ontologies, strict_target_ontology=strict_target_ontology
        )

        # Empty candidate list — return unmapped without calling the provider
        if not eligible_candidates:
            if timing_sink is not None:
                timing_sink["llm_reranker_provider_ms"] = 0.0
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
                alternatives=[],
                policy="production_grounded",
            )

        # Stable candidate ID map: C1, C2, C3, ...
        candidate_map: dict[str, NormalizedCandidate] = {
            f"C{i + 1}": c for i, c in enumerate(eligible_candidates)
        }
        candidate_codes: set[str] = {c.code for c in eligible_candidates}

        messages = self._build_messages(query_plan, candidate_map)

        logger.debug(
            "LLMReranker.rerank: source_term=%r mode=%s num_candidates=%d",
            query_plan.original_term,
            mode.value,
            len(eligible_candidates),
        )

        response = self._complete_rerank_timed(
            messages, timing_sink=timing_sink, usage_sink=usage_sink
        )

        # At most one recovery attempt total: an empty response from a
        # reasoning model retries with a larger completion budget; a
        # non-empty-but-malformed response retries with the normal reranker
        # budget. These are mutually exclusive (if/elif) so a pathological
        # response can never chain into a third provider call -- the second
        # attempt's outcome is never re-triaged, only parsed or failed below.
        raw = _strip_fences(response.content)
        data: dict[str, Any] | None = None
        if raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "LLMReranker returned malformed JSON; retrying once "
                    "(final allowed attempt). source_term=%r candidate_count=%d "
                    "model=%s error=%s",
                    query_plan.original_term,
                    len(eligible_candidates),
                    getattr(self._provider, "model", None),
                    exc,
                )
                response = self._complete_rerank_timed(
                    messages, timing_sink=timing_sink, usage_sink=usage_sink
                )
                raw = _strip_fences(response.content)
        elif _provider_uses_reasoning_model(self._provider):
            logger.warning(
                "LLMReranker received empty response from reasoning model; "
                "retrying once with a larger completion budget (final allowed "
                "attempt). source_term=%r candidate_count=%d model=%s",
                query_plan.original_term,
                len(eligible_candidates),
                getattr(self._provider, "model", None),
            )
            response = self._complete_rerank_timed(
                messages,
                timing_sink=timing_sink,
                usage_sink=usage_sink,
                max_tokens=_REASONING_RERANK_RETRY_TOKENS,
                min_completion_tokens=_REASONING_RERANK_RETRY_TOKENS,
            )
            raw = _strip_fences(response.content)
        # else: empty content and not a reasoning model -- no retry available;
        # data stays None and the checks below fail immediately.

        if data is None:
            if not raw.strip():
                raise LLMRerankerError(
                    "LLM returned empty response; for reasoning models this may "
                    "indicate max_completion_tokens was too low or no visible output "
                    "was produced. "
                    f"source_term={query_plan.original_term!r} "
                    f"candidate_count={len(eligible_candidates)} "
                    f"model={getattr(self._provider, 'model', None)!r}"
                )
            try:
                data = json.loads(raw)
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
            strict_target_ontology=strict_target_ontology,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _complete_rerank(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = _RERANK_MAX_TOKENS,
        min_completion_tokens: int = _REASONING_RERANK_MIN_COMPLETION_TOKENS,
    ) -> CompletionResponse:
        default_kwargs: dict[str, Any] = {}
        if _provider_uses_reasoning_model(self._provider):
            default_kwargs["min_completion_tokens"] = min_completion_tokens
            default_kwargs["reasoning_effort"] = (
                _openai_stage_reasoning_effort(self._provider) or "minimal"
            )
        temperature, kwargs = resolve_llm_call_params(
            self._llm_call_config,
            default_temperature=0.1,
            default_extra_kwargs=default_kwargs,
        )
        return self._provider.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _complete_rerank_timed(
        self,
        messages: list[ChatMessage],
        *,
        timing_sink: dict[str, float] | None,
        usage_sink: dict[str, Any] | None = None,
        max_tokens: int = _RERANK_MAX_TOKENS,
        min_completion_tokens: int = _REASONING_RERANK_MIN_COMPLETION_TOKENS,
    ) -> CompletionResponse:
        provider_start = time.monotonic()
        try:
            response = self._complete_rerank(
                messages,
                max_tokens=max_tokens,
                min_completion_tokens=min_completion_tokens,
            )
            if usage_sink is not None:
                usage_sink["llm_reranker"] = accumulate_usage(
                    usage_sink.get("llm_reranker"),
                    extract_completion_usage(response),
                )
            return response
        finally:
            if timing_sink is not None:
                timing_sink["llm_reranker_provider_ms"] = (
                    timing_sink.get("llm_reranker_provider_ms", 0.0)
                    + (time.monotonic() - provider_start) * 1000
                )

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
                    "You are a biomedical ontology reranking assistant. Return only valid JSON."
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
        strict_target_ontology: bool = False,
    ) -> RerankDecision:
        is_unmapped = bool(data.get("is_unmapped", False))
        selected_cid: str | None = data.get("selected_candidate_id")
        selected_code_raw: str | None = data.get("selected_code")
        selected_code: str | None = str(selected_code_raw).strip() if selected_code_raw else None

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

        if is_unmapped:
            # No candidate was selected, so there is no "selected confidence"
            # ceiling to enforce here (unlike the selected branch below) --
            # alternatives keep their own reranker-assigned confidence even
            # though the final decision confidence stays 0.0.
            allowed_ontologies = _normalize_allowed_target_ontologies(
                query_plan.allowed_target_ontologies
            )
            alternative_codes, alternatives = _parse_alternatives(
                data=data,
                candidate_map=candidate_map,
                candidate_codes=candidate_codes,
                selected_code=None,
                allowed_ontologies=allowed_ontologies,
                strict_target_ontology=strict_target_ontology,
            )
            return RerankDecision(
                selected_code=None,
                selected_candidate_id=None,
                is_unmapped=True,
                is_grounded=False,
                grounding_source=GroundingSource.NONE,
                retrieval_mode=retrieval_mode,
                confidence=confidence,
                reasoning=reasoning,
                alternative_codes=alternative_codes,
                alternatives=alternatives,
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
        if query_plan.target_ontology_constraint and not candidate_allowed_for_targets(
            selected_candidate,
            [query_plan.target_ontology_constraint],
            strict_target_ontology=strict_target_ontology,
        ):
            raise LLMRerankerError(
                f"LLM selected candidate with ontology={selected_candidate.ontology!r} "
                f"but target_ontology_constraint={query_plan.target_ontology_constraint.upper()!r}; "
                f"selected candidate must belong to the constrained ontology."
            )
        allowed_ontologies = _normalize_allowed_target_ontologies(
            query_plan.allowed_target_ontologies
        )
        if allowed_ontologies is not None and not candidate_allowed_for_targets(
            selected_candidate,
            allowed_ontologies,
            strict_target_ontology=strict_target_ontology,
        ):
            raise LLMRerankerError(
                f"LLM selected candidate with ontology={selected_candidate.ontology!r} "
                f"outside allowed_target_ontologies={sorted(allowed_ontologies)!r}; "
                f"selected candidate must belong to the caller-selected allow-list."
            )

        alternative_codes, alternatives = _parse_alternatives(
            data=data,
            candidate_map=candidate_map,
            candidate_codes=candidate_codes,
            selected_code=selected_code,
            allowed_ontologies=allowed_ontologies,
            strict_target_ontology=strict_target_ontology,
        )
        higher_alternatives = [alt for alt in alternatives if alt.confidence > confidence]
        if higher_alternatives:
            raise LLMRerankerError(
                "LLM returned alternative confidence greater than selected "
                "confidence; alternatives must use the same final confidence "
                "scale and be less than or equal to the selected candidate."
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
            alternatives=alternatives,
            policy="production_grounded",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_allowed_target_ontologies(
    ontologies: list[str] | None,
) -> set[str] | None:
    if ontologies is None:
        return None
    allowed = {
        str(ontology or "").upper().strip()
        for ontology in ontologies
        if str(ontology or "").strip()
    }
    return allowed or None


def _effective_allowed_target_ontologies(query_plan: QueryPlan) -> set[str] | None:
    allowed = _normalize_allowed_target_ontologies(query_plan.allowed_target_ontologies)
    if allowed is not None:
        return allowed
    if query_plan.target_ontology_constraint:
        return _normalize_allowed_target_ontologies([query_plan.target_ontology_constraint])
    return None


def _filter_allowed_candidates(
    candidates: Sequence[NormalizedCandidate],
    allowed_ontologies: set[str] | None,
    *,
    strict_target_ontology: bool = False,
) -> list[NormalizedCandidate]:
    if allowed_ontologies is None:
        return _filter_consistent_candidates(candidates)
    return [
        candidate
        for candidate in _filter_consistent_candidates(candidates)
        if candidate_allowed_for_targets(
            candidate, allowed_ontologies, strict_target_ontology=strict_target_ontology
        )
    ]


def _filter_consistent_candidates(
    candidates: Sequence[NormalizedCandidate],
) -> list[NormalizedCandidate]:
    result: list[NormalizedCandidate] = []
    for candidate in candidates:
        validation = validate_candidate_identity(
            ontology=candidate.ontology,
            code=candidate.code,
        )
        if validation.valid:
            result.append(candidate)
    return result


def fallback_alternative_explanation(candidate: NormalizedCandidate) -> str:
    """Return a concise reviewer-facing fallback explanation for a candidate."""
    term = candidate.term.strip()
    lower = term.lower()

    if re.search(r"\b(mean|average|averaged)\b", lower):
        return "Use if the field stores an average across repeated measurements."
    if re.search(r"\b(noninvasive|non-invasive|cuff)\b", lower):
        return "Use if the field indicates noninvasive cuff-based measurement."
    if re.search(r"\b(invasive|arterial line)\b", lower):
        return "Use if the measurement was obtained through invasive arterial monitoring."
    if re.search(r"\b(exercise|stress)\b", lower):
        return "Use if the measurement was taken during exercise or stress testing."
    if re.search(r"\b(aorta|aortic)\b", lower):
        return "Use if the measurement site is specifically the aorta."
    if re.search(r"\b(baseline|pre[- ]?treatment)\b", lower):
        return "Use if the field captures baseline values before treatment or follow-up."
    if re.search(r"\b(first encounter|first visit|initial encounter|initial visit)\b", lower):
        return "Use if the field captures the first clinical encounter."
    if re.search(r"\b(initial)\b", lower):
        return "Use if the field captures the initial measurement."
    if re.search(r"\b(follow[- ]?up|followup)\b", lower):
        return "Use if the field captures a follow-up timepoint."

    body_site = _candidate_body_site(lower)
    if body_site:
        return f"Use if the measurement site is specifically the {body_site}."

    specimen = _candidate_specimen(lower)
    if specimen:
        return f"Use if the source field uses a {specimen} specimen."

    if re.search(r"\bdiastolic\b", lower):
        return "Use if the field captures diastolic blood pressure."
    if re.search(r"\bsystolic\b", lower):
        return "Use if the field captures systolic blood pressure."

    concise_term = _concise_candidate_term(term)
    return f"Use only if the field specifically means {concise_term}."


def _parse_alternatives(
    *,
    data: dict[str, Any],
    candidate_map: dict[str, NormalizedCandidate],
    candidate_codes: set[str],
    selected_code: str | None,
    allowed_ontologies: set[str] | None,
    max_alternatives: int = 5,
    strict_target_ontology: bool = False,
) -> tuple[list[str], list[RerankAlternative]]:
    """Parse alternatives from LLM output.

    Structured alternatives carry final reranker confidence and are safe to
    expose as MappingResult alternatives. Legacy alternative_codes are retained
    only as compatibility identifiers; they do not have comparable confidence.
    """
    alternatives: list[RerankAlternative] = []
    seen_codes: set[str] = set()

    raw_structured = data.get("alternatives") or []
    if isinstance(raw_structured, list):
        for item in raw_structured:
            if len(alternatives) >= max_alternatives:
                break
            if not isinstance(item, dict):
                continue
            alt = _parse_structured_alternative(
                item=item,
                candidate_map=candidate_map,
                candidate_codes=candidate_codes,
                selected_code=selected_code,
                allowed_ontologies=allowed_ontologies,
                strict_target_ontology=strict_target_ontology,
            )
            if alt is None or alt.code in seen_codes:
                continue
            alternatives.append(alt)
            seen_codes.add(alt.code)

    legacy_codes: list[str] = []
    raw_legacy = data.get("alternative_codes") or []
    if isinstance(raw_legacy, list):
        for raw in raw_legacy:
            if len(legacy_codes) >= max_alternatives:
                break
            code = _resolve_legacy_alternative_code(raw, candidate_map)
            if not code or code == selected_code or code not in candidate_codes:
                continue
            if code in legacy_codes:
                continue
            legacy_codes.append(code)

    alternatives = sorted(
        alternatives[:max_alternatives],
        key=lambda alt: alt.confidence,
        reverse=True,
    )
    alternative_codes = [alt.code for alt in alternatives]
    for code in legacy_codes:
        if len(alternative_codes) >= max_alternatives:
            break
        if code not in alternative_codes:
            alternative_codes.append(code)

    return alternative_codes, alternatives


def _parse_structured_alternative(
    *,
    item: dict[str, Any],
    candidate_map: dict[str, NormalizedCandidate],
    candidate_codes: set[str],
    selected_code: str | None,
    allowed_ontologies: set[str] | None,
    strict_target_ontology: bool = False,
) -> RerankAlternative | None:
    candidate_id = str(item.get("candidate_id") or "").strip()
    if candidate_id not in candidate_map:
        return None

    candidate = candidate_map[candidate_id]
    if allowed_ontologies is not None and not candidate_allowed_for_targets(
        candidate,
        allowed_ontologies,
        strict_target_ontology=strict_target_ontology,
    ):
        return None
    code = str(item.get("code") or "").strip()
    if code != candidate.code or code == selected_code or code not in candidate_codes:
        return None

    raw_confidence = item.get("confidence")
    if raw_confidence is None:
        return None
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None

    explanation = str(item.get("explanation") or "").strip()
    if not explanation:
        explanation = fallback_alternative_explanation(candidate)

    try:
        return RerankAlternative(
            candidate_id=candidate_id,
            code=code,
            confidence=confidence,
            explanation=explanation,
        )
    except ValueError:
        return None


def _resolve_legacy_alternative_code(
    raw: Any,
    candidate_map: dict[str, NormalizedCandidate],
) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if value in candidate_map:
        return candidate_map[value].code
    return value


def _candidate_body_site(lower_term: str) -> str | None:
    site_patterns = [
        (r"\bpulmonary artery\b", "pulmonary artery"),
        (r"\bradial artery\b", "radial artery"),
        (r"\bfemoral artery\b", "femoral artery"),
        (r"\bcarotid artery\b", "carotid artery"),
        (r"\bbrachial artery\b", "brachial artery"),
        (r"\bleft arm\b", "left arm"),
        (r"\bright arm\b", "right arm"),
        (r"\barm\b", "arm"),
        (r"\bankle\b", "ankle"),
        (r"\bwrist\b", "wrist"),
        (r"\bleg\b", "leg"),
    ]
    for pattern, site in site_patterns:
        if re.search(pattern, lower_term):
            return site
    return None


def _candidate_specimen(lower_term: str) -> str | None:
    specimen_patterns = [
        (r"\bcerebrospinal fluid\b|\bcsf\b", "cerebrospinal fluid"),
        (r"\burine\b", "urine"),
        (r"\bserum\b", "serum"),
        (r"\bplasma\b", "plasma"),
        (r"\bsaliva\b", "saliva"),
        (r"\bsputum\b", "sputum"),
        (r"\bstool\b|\bfeces\b", "stool"),
        (r"\btissue\b", "tissue"),
        (r"\bswab\b", "swab"),
        (r"\bblood specimen\b|\bwhole blood\b", "blood"),
    ]
    for pattern, specimen in specimen_patterns:
        if re.search(pattern, lower_term):
            return specimen
    return None


def _concise_candidate_term(term: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9:/.-]*", term)
    if not words:
        return "this candidate concept"
    return " ".join(words[:max_words])


def _provider_uses_reasoning_model(provider: BaseLLMProvider) -> bool:
    return provider_uses_reasoning_model(provider)


def _openai_stage_reasoning_effort(provider: BaseLLMProvider) -> str | None:
    if getattr(provider, "provider_name", "") != "openai":
        return None
    return openai_reasoning_effort_for_model(str(getattr(provider, "model", "")))


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
        lines.append(f"- Target ontology constraint (HARD): {plan.target_ontology_constraint}")
    if plan.allowed_target_ontologies:
        lines.append(
            f"- Allowed target ontologies (HARD): {', '.join(plan.allowed_target_ontologies)}"
        )
    lines.append(f"- Retrieval mode: {plan.retrieval_mode.value}")
    if plan.reasoning:
        lines.append(f"- Plan reasoning: {plan.reasoning}")
    return "\n".join(lines)


def _build_candidate_list(candidate_map: dict[str, NormalizedCandidate]) -> str:
    """Format candidates as a numbered list for the LLM prompt."""
    parts: list[str] = []
    for cid, c in candidate_map.items():
        retrieved_from = (
            ", ".join(c.retrieved_from_ontologies)
            if c.retrieved_from_ontologies
            else "none"
        )
        fields = [
            f"{cid}: code={c.code}",
            f"term={c.term}",
            f"native_ontology={c.ontology}",
            f"retrieved_from_ontologies={retrieved_from}",
        ]
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
            defn = c.definition[:200] + "..." if len(c.definition) > 200 else c.definition
            fields.append(f"Def: {defn}")
        parts.append(" | ".join(fields))
    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
