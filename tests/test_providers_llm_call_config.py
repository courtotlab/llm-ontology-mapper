"""
Unit tests for the benchmark-facing LLM call configuration plumbing added to
providers.py: LLMCallConfig, resolve_llm_call_params, force_temperature,
strict mode, and usage extraction/accumulation helpers.

Run with:  pytest tests/test_providers_llm_call_config.py -v -m unit
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.providers import (
    ChatMessage,
    CompletionResponse,
    LLMCallConfig,
    OpenAIProvider,
    accumulate_usage,
    extract_completion_usage,
    resolve_llm_call_params,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# resolve_llm_call_params
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_llm_call_params_none_config_preserves_defaults() -> None:
    temperature, kwargs = resolve_llm_call_params(
        None, default_temperature=0.1, default_extra_kwargs={"reasoning_effort": "medium"}
    )
    assert temperature == 0.1
    assert kwargs == {"reasoning_effort": "medium"}


def test_resolve_llm_call_params_overrides_temperature_seed_reasoning() -> None:
    config = LLMCallConfig(temperature=0.0, seed=42, reasoning_effort="low", force_temperature=True, strict=True)
    temperature, kwargs = resolve_llm_call_params(
        config, default_temperature=0.1, default_extra_kwargs={"reasoning_effort": "medium"}
    )
    assert temperature == 0.0
    assert kwargs == {
        "reasoning_effort": "low",
        "seed": 42,
        "force_temperature": True,
        "strict": True,
    }


def test_resolve_llm_call_params_partial_override_leaves_rest_default() -> None:
    config = LLMCallConfig(seed=7)
    temperature, kwargs = resolve_llm_call_params(config, default_temperature=0.1, default_extra_kwargs={})
    assert temperature == 0.1
    assert kwargs == {"seed": 7}


def test_resolve_llm_call_params_force_temperature_none_omits_provider_default() -> None:
    """force_temperature=True with temperature=None means 'omit temperature'
    (provider default) -- it must NOT fall back to the component's own
    default_temperature the way an ordinary unset override would."""
    config = LLMCallConfig(
        temperature=None, seed=42, reasoning_effort="low", force_temperature=True, strict=True
    )
    temperature, kwargs = resolve_llm_call_params(config, default_temperature=0.1, default_extra_kwargs={})
    assert temperature is None
    assert kwargs == {
        "reasoning_effort": "low",
        "seed": 42,
        "force_temperature": True,
        "strict": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# force_temperature bypasses the reasoning-model silent temperature drop
# ─────────────────────────────────────────────────────────────────────────────


def _mock_openai_response(model: str = "gpt-5.6-luna") -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = '{"ok": true}'
    resp.model = model
    resp.usage = None
    resp.system_fingerprint = None
    return resp


def test_reasoning_model_drops_temperature_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response()
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete([ChatMessage(role="user", content="hi")], temperature=0.0, max_tokens=16)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs


def test_force_temperature_sends_temperature_for_reasoning_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response()
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=16,
        force_temperature=True,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0


def test_temperature_none_omits_parameter_for_reasoning_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPT-5.6 Luna incompatibility this fixes: when the benchmark asks for
    provider-default temperature (None), the request must never include
    temperature=None and must never substitute temperature=1 on the model's
    behalf."""
    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response()
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete([ChatMessage(role="user", content="hi")], temperature=None, max_tokens=16)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs


def test_temperature_none_omits_parameter_even_with_force_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force_temperature bypasses the reasoning-model silent-drop for a real
    numeric override, but temperature=None must still omit the parameter --
    force never substitutes a fabricated default value."""
    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response()
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hi")],
        temperature=None,
        max_tokens=16,
        force_temperature=True,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs


def test_temperature_none_omits_parameter_for_non_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temperature omission must be consistent across model families, not
    just the reasoning models that already dropped it silently -- gpt-4.1-mini
    normally always sends temperature, but an explicit None still omits it."""
    provider = OpenAIProvider(model="gpt-4.1-mini")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(model="gpt-4.1-mini")
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete([ChatMessage(role="user", content="hi")], temperature=None, max_tokens=16)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs


def test_seed_is_forwarded_to_request(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-4.1-mini")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(model="gpt-4.1-mini")
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete([ChatMessage(role="user", content="hi")], temperature=0.0, max_tokens=16, seed=42)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["seed"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# strict mode: unsupported reasoning_effort raises instead of being dropped
# ─────────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_instead_of_silently_dropping_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()

    rejection = openai.APIStatusError(
        "Unrecognized request argument supplied: reasoning_effort",
        response=MagicMock(status_code=400),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = rejection
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    with pytest.raises(RuntimeError):
        provider.complete(
            [ChatMessage(role="user", content="hi")],
            temperature=0.1,
            max_tokens=16,
            reasoning_effort="low",
            strict=True,
        )


def test_non_strict_mode_silently_drops_rejected_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    provider = OpenAIProvider(model="gpt-5.6-luna")
    mock_client = MagicMock()

    rejection = openai.APIStatusError(
        "Unrecognized request argument supplied: reasoning_effort",
        response=MagicMock(status_code=400),
        body=None,
    )
    ok_response = _mock_openai_response()
    mock_client.chat.completions.create.side_effect = [rejection, ok_response]
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    resp = provider.complete(
        [ChatMessage(role="user", content="hi")],
        temperature=0.1,
        max_tokens=16,
        reasoning_effort="low",
        strict=False,
    )
    assert resp.content == '{"ok": true}'
    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert "reasoning_effort" not in second_call_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Usage extraction / accumulation
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_completion_usage_reads_cached_and_reasoning_tokens() -> None:
    raw = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=100),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=50),
        ),
        system_fingerprint="fp_123",
    )
    response = CompletionResponse(content="x", model="gpt-5.6-luna", prompt_tokens=500, completion_tokens=200, raw=raw)
    usage = extract_completion_usage(response)
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 200
    assert usage["cached_input_tokens"] == 100
    assert usage["reasoning_tokens"] == 50
    assert usage["system_fingerprint"] == "fp_123"


def test_extract_completion_usage_never_fabricates_missing_detail() -> None:
    response = CompletionResponse(content="x", model="gpt-4.1-mini", prompt_tokens=10, completion_tokens=5, raw=None)
    usage = extract_completion_usage(response)
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["cached_input_tokens"] is None
    assert usage["reasoning_tokens"] is None
    assert usage["system_fingerprint"] is None


def test_accumulate_usage_sums_across_calls() -> None:
    first = {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2, "reasoning_tokens": None,
             "model": "gpt-5.6-luna", "system_fingerprint": "fp_1", "call_count": 1}
    second = {"input_tokens": 20, "output_tokens": 15, "cached_input_tokens": None, "reasoning_tokens": 3,
              "model": "gpt-5.6-luna", "system_fingerprint": "fp_2", "call_count": 1}
    merged = accumulate_usage(first, second)
    assert merged["input_tokens"] == 30
    assert merged["output_tokens"] == 20
    assert merged["cached_input_tokens"] == 2
    assert merged["reasoning_tokens"] == 3
    assert merged["call_count"] == 2
    assert merged["system_fingerprint"] == "fp_2"


def test_accumulate_usage_first_call_returns_copy() -> None:
    first = extract_completion_usage(
        CompletionResponse(content="x", model="m", prompt_tokens=1, completion_tokens=1, raw=None)
    )
    merged = accumulate_usage(None, first)
    assert merged == first
    assert merged is not first
