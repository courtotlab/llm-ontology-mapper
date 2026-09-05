"""
Centralized model-pricing configuration for API cost estimation.

Pricing is a manually-verified snapshot, not something read from an API, so
values here must never be guessed. PRICING_SNAPSHOT_DATE and each entry are
persisted verbatim into benchmark_config.json so historical benchmark costs
remain reproducible if provider pricing changes later.

Reasoning-token billing note
─────────────────────────────
OpenAI's `completion_tokens` (this codebase's `output_tokens`) already
INCLUDES reasoning tokens -- `completion_tokens_details.reasoning_tokens` is
a breakdown of output_tokens, not an additive count. calculate_cost_usd
therefore bills the full output_tokens figure once at the output rate and
never adds reasoning_tokens on top, which would double-charge reasoning
usage. Likewise `input_tokens` (prompt_tokens) already includes cached
tokens as a subset (`prompt_tokens_details.cached_tokens`); the non-cached
portion is billed at the input rate and the cached portion at the cheaper
cached-input rate.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_SNAPSHOT_DATE = "2026-08-19"


@dataclass(frozen=True)
class ModelPricing:
    """Verified per-1M-token USD pricing for one model."""

    model: str
    input_per_1m_usd: float
    cached_input_per_1m_usd: float
    output_per_1m_usd: float
    snapshot_date: str


# Verified pricing snapshot, supplied directly by the repository owner for
# this benchmark run (2026-08-19). Do not extrapolate or guess new entries --
# add a new verified ModelPricing when a new model needs benchmarking.
PRICING_SNAPSHOT: dict[str, ModelPricing] = {
    "gpt-5.6-luna": ModelPricing(
        model="gpt-5.6-luna",
        input_per_1m_usd=0.20,
        cached_input_per_1m_usd=0.02,
        output_per_1m_usd=1.20,
        snapshot_date=PRICING_SNAPSHOT_DATE,
    ),
    "gpt-5.4-mini": ModelPricing(
        model="gpt-5.4-mini",
        input_per_1m_usd=0.75,
        cached_input_per_1m_usd=0.075,
        output_per_1m_usd=4.50,
        snapshot_date=PRICING_SNAPSHOT_DATE,
    ),
    "gpt-5-mini": ModelPricing(
        model="gpt-5-mini",
        input_per_1m_usd=0.25,
        cached_input_per_1m_usd=0.025,
        output_per_1m_usd=2.00,
        snapshot_date=PRICING_SNAPSHOT_DATE,
    ),
    "gpt-4.1-mini": ModelPricing(
        model="gpt-4.1-mini",
        input_per_1m_usd=0.40,
        cached_input_per_1m_usd=0.10,
        output_per_1m_usd=1.60,
        snapshot_date=PRICING_SNAPSHOT_DATE,
    ),
}


def get_pricing(model: str) -> ModelPricing:
    try:
        return PRICING_SNAPSHOT[model]
    except KeyError as exc:
        raise KeyError(
            f"No verified pricing snapshot for model={model!r}. "
            f"Known models: {sorted(PRICING_SNAPSHOT)}. "
            "Add a verified ModelPricing entry to PRICING_SNAPSHOT before "
            "benchmarking a new model -- cost must never be estimated."
        ) from exc


def calculate_cost_usd(
    *,
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    pricing: ModelPricing,
) -> float | None:
    """
    Compute estimated API cost in USD from recorded token usage.

    Returns None when input_tokens/output_tokens are both unavailable (e.g.
    a row with no LLM call), rather than fabricating a zero cost.
    """
    if input_tokens is None and output_tokens is None:
        return None

    cached = min(cached_input_tokens or 0, input_tokens or 0)
    non_cached_input = max((input_tokens or 0) - cached, 0)

    return (
        non_cached_input / 1_000_000 * pricing.input_per_1m_usd
        + cached / 1_000_000 * pricing.cached_input_per_1m_usd
        + (output_tokens or 0) / 1_000_000 * pricing.output_per_1m_usd
    )
