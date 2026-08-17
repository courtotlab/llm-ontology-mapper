"""
Unit tests for LLM provider layer (providers.py).

Uses pytest-mock / monkeypatching — no real API calls.

Run with:  pytest tests/test_providers.py -v -m unit
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_ontology_mapper.providers import (
    AnthropicProvider,
    ChatMessage,
    CompletionResponse,
    LLMProviderFactory,
    OllamaProvider,
    OpenAIProvider,
    _RetryableError,
    is_reasoning_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# LLMProviderFactory
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("provider,expected_class", [
    ("openai",    OpenAIProvider),
    ("github",    OpenAIProvider),
    ("azure",     OpenAIProvider),
    ("anthropic", AnthropicProvider),
    ("ollama",    OllamaProvider),
    ("local",     OllamaProvider),
])
def test_factory_returns_correct_class(provider: str, expected_class: type) -> None:
    instance = LLMProviderFactory.from_config(provider=provider, model="test-model")
    assert isinstance(instance, expected_class)


@pytest.mark.unit
def test_factory_github_sets_base_url() -> None:
    p = LLMProviderFactory.from_config(provider="github", model="gpt-4o")
    assert isinstance(p, OpenAIProvider)
    assert p._base_url == LLMProviderFactory._GITHUB_BASE_URL


@pytest.mark.unit
def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMProviderFactory.from_config(provider="notareal", model="x")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIProvider — retry logic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_openai_provider_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-4o", max_retries=3)

    call_count = 0

    def _flaky_do_complete(messages, temperature, max_tokens, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _RetryableError("rate limit")
        return CompletionResponse(content='{"code":"HP:0000001"}', model="gpt-4o")

    monkeypatch.setattr(provider, "_do_complete", _flaky_do_complete)
    monkeypatch.setattr("time.sleep", lambda _: None)  # skip back-off delay

    resp = provider.complete([ChatMessage(role="user", content="test")])
    assert call_count == 3
    assert resp.content == '{"code":"HP:0000001"}'


@pytest.mark.unit
def test_openai_provider_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-4o", max_retries=2)

    monkeypatch.setattr(provider, "_do_complete", MagicMock(side_effect=_RetryableError("timeout")))
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed after 2 retries"):
        provider.complete([ChatMessage(role="user", content="x")])


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIProvider — request kwargs
# ─────────────────────────────────────────────────────────────────────────────


def _make_openai_completion_response(
    *,
    content: str = "ok",
    model: str = "gpt-4.1-mini",
) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage = None
    return resp


@pytest.mark.unit
def test_openai_complete_gpt_41_mini_uses_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="gpt-4.1-mini")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = (
        _make_openai_completion_response(model="gpt-4.1-mini")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hello")],
        max_tokens=512,
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 512
    assert "max_completion_tokens" not in call_kwargs


@pytest.mark.unit
def test_openai_complete_gpt_5_omits_temperature_and_uses_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="gpt-5")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = (
        _make_openai_completion_response(model="gpt-5")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hello")],
        max_tokens=512,
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert "temperature" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 512
    assert call_kwargs["reasoning_effort"] == "minimal"
    assert "max_tokens" not in call_kwargs


@pytest.mark.unit
def test_openai_complete_o_series_omits_temperature_and_uses_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="o3-mini")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = (
        _make_openai_completion_response(model="o3-mini")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hello")],
        max_tokens=256,
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert "temperature" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 256
    assert call_kwargs["reasoning_effort"] == "low"
    assert "max_tokens" not in call_kwargs


@pytest.mark.unit
def test_openai_complete_gpt_5_honors_min_completion_token_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="gpt-5")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = (
        _make_openai_completion_response(model="gpt-5")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hello")],
        max_tokens=512,
        min_completion_tokens=4096,
        reasoning_effort="minimal",
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["max_completion_tokens"] == 4096
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["reasoning_effort"] == "minimal"


@pytest.mark.unit
def test_openai_complete_preserves_explicit_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="gpt-5")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = (
        _make_openai_completion_response(model="gpt-5")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    provider.complete(
        [ChatMessage(role="user", content="hello")],
        max_tokens=512,
        reasoning_effort="low",
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["reasoning_effort"] == "low"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5-2026-06-01", True),
        ("o1", True),
        ("o3-mini", True),
        ("o4", True),
        ("gpt-4.1-mini", False),
        ("claude-3-5-sonnet", False),
    ],
)
def test_is_reasoning_model(model: str, expected: bool) -> None:
    assert is_reasoning_model(model) is expected


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIProvider — provider_name
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_openai_provider_name_standard() -> None:
    assert OpenAIProvider(model="gpt-4o").provider_name == "openai"


@pytest.mark.unit
def test_openai_provider_name_github() -> None:
    p = OpenAIProvider(model="gpt-4o", base_url="https://models.inference.ai.azure.com")
    assert p.provider_name == "github"


@pytest.mark.unit
def test_openai_provider_name_azure() -> None:
    p = OpenAIProvider(model="gpt-4o", azure_endpoint="https://myendpoint.openai.azure.com")
    assert p.provider_name == "azure"


# ─────────────────────────────────────────────────────────────────────────────
# Missing SDK raises ImportError with install hint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_openai_provider_raises_import_error_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    provider = OpenAIProvider(model="gpt-4o")
    provider._client = None  # ensure lazy init runs

    with pytest.raises(ImportError, match="pip install"):
        provider._get_client()
