"""
Unit tests for LLM provider layer (providers.py).

Uses pytest-mock / monkeypatching — no real API calls.

Run with:  pytest tests/test_providers.py -v -m unit
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from llm_ontology_mapper.providers import (
    AnthropicProvider,
    ChatMessage,
    CompletionResponse,
    LLMProviderFactory,
    OllamaProvider,
    OpenAIProvider,
    _RetryableError,
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
