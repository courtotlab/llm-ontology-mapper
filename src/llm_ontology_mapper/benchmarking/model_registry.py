"""
Explicit allow-list of benchmark models.

One invocation of scripts/run_model_benchmark.py runs exactly one of these
models, twice, over the full dataset -- never all of them. This registry
exists so an unreviewed/unsupported model string can never accidentally
launch a real, paid benchmark run: get_model_config() raises for anything
not listed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkModelConfig:
    """Locked per-model benchmark configuration.

    reasoning_effort is None for non-reasoning models (e.g. gpt-4.1-mini) --
    the benchmark must never send a reasoning_effort parameter for those,
    and must report requested_reasoning_effort="N/A" rather than pretend
    support.
    """

    model: str
    is_reasoning: bool
    reasoning_effort: str | None


ALLOWED_MODELS: dict[str, BenchmarkModelConfig] = {
    "gpt-5.6-luna": BenchmarkModelConfig(model="gpt-5.6-luna", is_reasoning=True, reasoning_effort="low"),
    "gpt-5.4-mini": BenchmarkModelConfig(model="gpt-5.4-mini", is_reasoning=True, reasoning_effort="low"),
    "gpt-5-mini": BenchmarkModelConfig(model="gpt-5-mini", is_reasoning=True, reasoning_effort="low"),
    "gpt-4.1-mini": BenchmarkModelConfig(model="gpt-4.1-mini", is_reasoning=False, reasoning_effort=None),
}


def get_model_config(name: str) -> BenchmarkModelConfig:
    try:
        return ALLOWED_MODELS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported benchmark model {name!r}. "
            f"Allowed models: {sorted(ALLOWED_MODELS)}. "
            "This benchmark intentionally rejects unreviewed models -- add "
            "a new reviewed BenchmarkModelConfig entry to run a new model."
        ) from exc
