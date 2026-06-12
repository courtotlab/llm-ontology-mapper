"""
Unit tests for A2: Provider tool-calling layer.

Covers:
  - to_provider_tools() translation per provider
  - OpenAIProvider.complete_with_tools() — single and parallel tool calls
  - AnthropicProvider.complete_with_tools() — text-before-tool_use, parallel
  - OllamaProvider.complete_with_tools() — format="json" absent, empty guard

All SDK calls are mocked; no live services needed.

Run with:  pytest tests/test_tool_calling.py -v -m unit
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from llm_ontology_mapper.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    ToolCall,
    _RetryableError,
    to_provider_tools,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_ols",
        "description": "Search EBI OLS for an ontology term",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "ontology": {"type": "string"},
            },
            "required": ["query", "ontology"],
        },
    },
    {
        "name": "search_loinc",
        "description": "Search LOINC for a measurement or lab test",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# to_provider_tools — format translation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_to_provider_tools_openai_shape() -> None:
    result = to_provider_tools("openai", _SAMPLE_TOOLS)
    assert len(result) == 2
    t = result[0]
    assert t["type"] == "function"
    assert "function" in t
    assert t["function"]["name"] == "search_ols"
    assert t["function"]["description"] == "Search EBI OLS for an ontology term"
    assert "parameters" in t["function"]
    assert t["function"]["parameters"]["type"] == "object"


@pytest.mark.unit
def test_to_provider_tools_anthropic_shape() -> None:
    result = to_provider_tools("anthropic", _SAMPLE_TOOLS)
    assert len(result) == 2
    t = result[0]
    # Anthropic uses flat name/description/input_schema — no "type":"function" wrapper
    assert t["name"] == "search_ols"
    assert t["description"] == "Search EBI OLS for an ontology term"
    assert "input_schema" in t
    assert t["input_schema"]["type"] == "object"
    assert "type" not in t


@pytest.mark.unit
def test_to_provider_tools_ollama_is_openai_style() -> None:
    result = to_provider_tools("ollama", _SAMPLE_TOOLS)
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search_ols"


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["github", "azure", "local"])
def test_to_provider_tools_openai_style_variants(provider: str) -> None:
    result = to_provider_tools(provider, _SAMPLE_TOOLS)
    assert result[0]["type"] == "function"


@pytest.mark.unit
def test_to_provider_tools_empty_list() -> None:
    assert to_provider_tools("openai", []) == []


@pytest.mark.unit
def test_to_provider_tools_falls_back_to_inputschema_then_parameters() -> None:
    """Tool with only 'parameters' key (not 'inputSchema') is still translated."""
    tool = {"name": "x", "description": "y", "parameters": {"type": "object", "properties": {}}}
    result = to_provider_tools("openai", [tool])
    assert result[0]["function"]["parameters"]["type"] == "object"


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIProvider.complete_with_tools
# ─────────────────────────────────────────────────────────────────────────────

def _make_openai_tc(tc_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_openai_resp(content: str | None, tool_calls: list) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = tool_calls or None
    return resp


@pytest.mark.unit
def test_openai_complete_with_tools_single_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="gpt-4o")

    tc = _make_openai_tc("call_1", "search_ols", {"query": "cough", "ontology": "HPO"})
    mock_resp = _make_openai_resp(None, [tc])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: cough")],
        _SAMPLE_TOOLS,
    )

    assert text is None
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCall)
    assert calls[0].id == "call_1"
    assert calls[0].name == "search_ols"
    assert calls[0].arguments == {"query": "cough", "ontology": "HPO"}


@pytest.mark.unit
def test_openai_complete_with_tools_parallel_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model may call multiple tools in one turn (parallel tool use)."""
    provider = OpenAIProvider(model="gpt-4o")

    tc1 = _make_openai_tc("call_1", "search_ols",   {"query": "blood pressure", "ontology": "HPO"})
    tc2 = _make_openai_tc("call_2", "search_loinc",  {"query": "blood pressure"})
    mock_resp = _make_openai_resp(None, [tc1, tc2])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: sys_bp")],
        _SAMPLE_TOOLS,
    )

    assert text is None
    assert len(calls) == 2
    assert calls[0].name == "search_ols"
    assert calls[1].name == "search_loinc"


@pytest.mark.unit
def test_openai_complete_with_tools_text_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the model returns text (final answer), tool_calls is empty."""
    provider = OpenAIProvider(model="gpt-4o")

    mock_resp = _make_openai_resp('{"code":"HP:0002110","term":"Bronchiectasis"}', [])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: cough")],
        _SAMPLE_TOOLS,
    )

    assert text == '{"code":"HP:0002110","term":"Bronchiectasis"}'
    assert calls == []


@pytest.mark.unit
def test_openai_complete_with_tools_passes_tools_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK create() must receive the tools= kwarg in OpenAI format."""
    provider = OpenAIProvider(model="gpt-4o")

    mock_resp = _make_openai_resp("done", [])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete_with_tools(
        [ChatMessage(role="user", content="x")],
        _SAMPLE_TOOLS[:1],
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert "tools" in call_kwargs
    assert call_kwargs["tools"][0]["type"] == "function"
    assert call_kwargs["max_tokens"] == 1024
    assert "max_completion_tokens" not in call_kwargs


@pytest.mark.unit
def test_openai_complete_with_tools_uses_completion_tokens_for_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="o3-mini")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_resp("done", [])
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete_with_tools(
        [ChatMessage(role="user", content="x")],
        _SAMPLE_TOOLS[:1],
        max_tokens=321,
    )

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["max_completion_tokens"] == 321
    assert "max_tokens" not in call_kwargs


@pytest.mark.unit
def test_openai_complete_with_tools_retries_with_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(model="custom-model")
    message = (
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    error = openai.BadRequestError(
        message=message,
        response=response,
        body={"error": {"message": message}},
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        error,
        _make_openai_resp("done", []),
    ]
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete_with_tools(
        [ChatMessage(role="user", content="x")],
        _SAMPLE_TOOLS[:1],
        max_tokens=456,
    )

    first_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
    second_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert first_kwargs["max_tokens"] == 456
    assert "max_completion_tokens" not in first_kwargs
    assert second_kwargs["max_completion_tokens"] == 456
    assert "max_tokens" not in second_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# AnthropicProvider.complete_with_tools
# ─────────────────────────────────────────────────────────────────────────────

def _make_anthropic_text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _make_anthropic_tool_block(block_id: str, name: str, input_data: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = block_id
    b.name = name
    b.input = input_data
    return b


def _make_anthropic_resp(blocks: list) -> MagicMock:
    resp = MagicMock()
    resp.content = blocks
    return resp


import sys as _sys
import types as _types


def _fake_anthropic_module() -> MagicMock:
    """Build a minimal fake anthropic module so the SDK import succeeds."""
    mod = MagicMock()
    # RateLimitError and APIStatusError must be real exception classes
    mod.RateLimitError = type("RateLimitError", (Exception,), {})
    mod.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 400})
    return mod


@pytest.fixture()
def patch_anthropic(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake anthropic into sys.modules for the duration of a test."""
    fake = _fake_anthropic_module()
    monkeypatch.setitem(_sys.modules, "anthropic", fake)
    return fake


@pytest.mark.unit
def test_anthropic_complete_with_tools_text_before_tool_use(
    monkeypatch: pytest.MonkeyPatch, patch_anthropic: MagicMock,
) -> None:
    """
    Anthropic may return a text block BEFORE a tool_use block in the same response.
    Both must be captured: text from the text block, ToolCall from the tool_use block.
    """
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

    blocks = [
        _make_anthropic_text_block("Let me search for that term."),
        _make_anthropic_tool_block("tu_1", "search_ols", {"query": "cough", "ontology": "HPO"}),
    ]
    mock_resp = _make_anthropic_resp(blocks)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: cough")],
        _SAMPLE_TOOLS,
    )

    assert text == "Let me search for that term."
    assert len(calls) == 1
    assert calls[0].id == "tu_1"
    assert calls[0].name == "search_ols"
    assert calls[0].arguments == {"query": "cough", "ontology": "HPO"}


@pytest.mark.unit
def test_anthropic_complete_with_tools_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch, patch_anthropic: MagicMock,
) -> None:
    """Multiple tool_use blocks in one Anthropic response → all captured."""
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

    blocks = [
        _make_anthropic_tool_block("tu_1", "search_ols",  {"query": "sys bp", "ontology": "HPO"}),
        _make_anthropic_tool_block("tu_2", "search_loinc", {"query": "systolic blood pressure"}),
    ]
    mock_resp = _make_anthropic_resp(blocks)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: sys_bp")],
        _SAMPLE_TOOLS,
    )

    assert text is None
    assert len(calls) == 2
    assert {c.name for c in calls} == {"search_ols", "search_loinc"}
    ids = {c.id for c in calls}
    assert "tu_1" in ids and "tu_2" in ids


@pytest.mark.unit
def test_anthropic_complete_with_tools_tool_use_only(
    monkeypatch: pytest.MonkeyPatch, patch_anthropic: MagicMock,
) -> None:
    """Tool-only response (no text block) → text is None."""
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

    blocks = [
        _make_anthropic_tool_block("tu_1", "search_ols", {"query": "fever", "ontology": "HPO"}),
    ]
    mock_resp = _make_anthropic_resp(blocks)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: fever")],
        _SAMPLE_TOOLS,
    )

    assert text is None
    assert len(calls) == 1


@pytest.mark.unit
def test_anthropic_complete_with_tools_passes_tools_in_anthropic_format(
    monkeypatch: pytest.MonkeyPatch, patch_anthropic: MagicMock,
) -> None:
    """SDK create() must receive tools= in Anthropic format (input_schema, no type wrapper)."""
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

    mock_resp = _make_anthropic_resp([_make_anthropic_text_block("done")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete_with_tools(
        [ChatMessage(role="user", content="x")],
        _SAMPLE_TOOLS[:1],
    )

    _, call_kwargs = mock_client.messages.create.call_args
    assert "tools" in call_kwargs
    tool = call_kwargs["tools"][0]
    assert "input_schema" in tool
    assert "type" not in tool


# ─────────────────────────────────────────────────────────────────────────────
# OllamaProvider.complete_with_tools
# ─────────────────────────────────────────────────────────────────────────────

def _make_ollama_resp(content: str, tool_calls: list[dict]) -> dict:
    """Return a dict-style Ollama response (the SDK can also return objects)."""
    return {"message": {"content": content, "tool_calls": tool_calls}}


@pytest.mark.unit
def test_ollama_complete_with_tools_single_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(model="llama3")

    raw_tc = {"function": {"name": "search_ols", "arguments": {"query": "cough", "ontology": "HPO"}}}
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_resp("", [raw_tc])
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: cough")],
        _SAMPLE_TOOLS,
    )

    assert len(calls) == 1
    assert calls[0].name == "search_ols"
    assert calls[0].arguments == {"query": "cough", "ontology": "HPO"}


@pytest.mark.unit
def test_ollama_complete_with_tools_no_format_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """complete_with_tools() must NOT pass format='json' to the Ollama SDK."""
    provider = OllamaProvider(model="llama3")

    raw_tc = {"function": {"name": "search_loinc", "arguments": {"query": "bp"}}}
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_resp("", [raw_tc])
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete_with_tools(
        [ChatMessage(role="user", content="Map: bp")],
        _SAMPLE_TOOLS,
    )

    _, call_kwargs = mock_client.chat.call_args
    # format="json" must NOT be present
    assert "format" not in call_kwargs
    # tools= must be present
    assert "tools" in call_kwargs


@pytest.mark.unit
def test_ollama_complete_with_tools_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No tool calls AND no content → explicit RuntimeError (not silent loop)."""
    provider = OllamaProvider(model="llama3")

    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_resp("", [])
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    with pytest.raises(RuntimeError, match="neither tool calls nor text"):
        provider.complete_with_tools(
            [ChatMessage(role="user", content="x")],
            _SAMPLE_TOOLS,
        )


@pytest.mark.unit
def test_ollama_complete_with_tools_text_only_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model returned plain text (final answer) — no tool calls."""
    provider = OllamaProvider(model="llama3")

    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_resp(
        '{"code":"HP:0002110","term":"Bronchiectasis"}', []
    )
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    text, calls = provider.complete_with_tools(
        [ChatMessage(role="user", content="x")],
        _SAMPLE_TOOLS,
    )

    assert text == '{"code":"HP:0002110","term":"Bronchiectasis"}'
    assert calls == []


@pytest.mark.unit
def test_ollama_complete_still_uses_format_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing complete() path must still pass format='json' (unchanged)."""
    provider = OllamaProvider(model="llama3")

    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": '{"code":"HP:001"}'}}
    monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

    from llm_ontology_mapper.providers import ChatMessage
    provider.complete([ChatMessage(role="user", content="x")])

    _, call_kwargs = mock_client.chat.call_args
    assert call_kwargs.get("format") == "json"


# ─────────────────────────────────────────────────────────────────────────────
# BaseLLMProvider — default raises NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_base_provider_complete_with_tools_raises_not_implemented() -> None:
    """Any provider that hasn't overridden complete_with_tools() raises."""
    from llm_ontology_mapper.providers import BaseLLMProvider, ChatMessage, CompletionResponse

    class _MinimalProvider(BaseLLMProvider):
        def complete(self, messages, temperature=0.1, max_tokens=1024, **kwargs):
            return CompletionResponse(content="x", model=self.model)

    p = _MinimalProvider(model="test")
    with pytest.raises(NotImplementedError, match="does not implement complete_with_tools"):
        p.complete_with_tools([ChatMessage(role="user", content="x")], [])
