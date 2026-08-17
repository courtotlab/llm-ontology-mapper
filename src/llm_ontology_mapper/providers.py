"""
Abstract LLM provider interface.

This module defines the contract that every backend (OpenAI, Anthropic, Ollama,
GitHub Models, Azure OpenAI) must satisfy.  The rest of the package only ever
calls BaseLLMProvider methods — no provider SDK bleeds into mapper.py or retriever.py.

Pattern: Strategy + Factory
  • BaseLLMProvider  — abstract strategy
  • LLMProviderFactory.from_config()  — constructs the right concrete strategy

Adding a new provider:
  1. Add a class in this file that subclasses BaseLLMProvider.
  2. Implement _do_complete(messages, temperature, max_tokens) → CompletionResponse.
     - Raise _RetryableError for transient failures (rate-limit, 5xx) so the
       base-class retry loop in _complete_with_retry() handles back-off automatically.
     - Raise RuntimeError for permanent failures (bad auth, invalid model, etc.).
     - Do NOT override complete() — it delegates to _complete_with_retry() for free.
  3. Optionally override provider_name (used in MappingMetadata).
  4. Add an elif branch in LLMProviderFactory.from_config() mapping a shorthand
     string (e.g. "myprovider") to your new class.
  5. Add a test in tests/test_providers.py.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Value objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChatMessage:
    """Single message in a chat conversation."""

    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResponse:
    """Normalised response from any LLM provider."""

    content: str
    model: str
    prompt_tokens: int | None    = None
    completion_tokens: int | None = None
    raw: Any | None              = None   # original SDK response object


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str           # provider-assigned call ID (used to route results back)
    name: str         # tool/function name
    arguments: dict[str, Any]  # parsed argument dict


# ─────────────────────────────────────────────────────────────────────────────
# Tool-format translation
# ─────────────────────────────────────────────────────────────────────────────


def to_provider_tools(provider: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Translate MCP JSON-Schema tool definitions to the wire format expected by
    a given provider.

    Input tool shape (MCP / library-internal):
        {
            "name": "search_ols",
            "description": "...",
            "inputSchema": {"type": "object", "properties": {...}, "required": [...]}
        }

    Output per provider:
        OpenAI / Ollama / GitHub / Azure:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic:
            {"name": ..., "description": ..., "input_schema": {...}}
    """
    p = provider.lower()
    result: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        schema = (
            tool.get("inputSchema")
            or tool.get("parameters")
            or {"type": "object", "properties": {}}
        )
        if p == "anthropic":
            result.append({
                "name": name,
                "description": description,
                "input_schema": schema,
            })
        else:
            # OpenAI-style: covers openai, ollama, github, azure, local
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class BaseLLMProvider(abc.ABC):
    """
    Abstract strategy for a single LLM backend.

    Concrete subclasses live in:
        providers/openai_provider.py      (OpenAI, GitHub Models, Azure OpenAI)
        providers/anthropic_provider.py   (Anthropic Claude)
        providers/ollama_provider.py      (local Ollama)
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        self.model       = model
        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries

    # ── Sync completion ───────────────────────────────────────────────────────

    @abc.abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Send a chat completion request and return a normalised response.

        Implementations must:
        - Retry on transient errors (rate-limit, 5xx) up to self.max_retries
        - Raise ValueError for authentication failures (no retry)
        - Raise RuntimeError for non-retryable provider errors
        - Log at DEBUG level: model, token counts, latency
        """

    # ── Tool-calling completion ───────────────────────────────────────────────

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str | None, list[ToolCall]]:
        """
        Send a chat completion request that may produce tool calls.

        Args:
            messages: Conversation history (system + user turns).
            tools:    MCP JSON-Schema tool definitions (inputSchema key).
            temperature, max_tokens: standard sampling params.

        Returns:
            (text, tool_calls)
            text:       The model's text content, or None when it produced only
                        tool calls with no accompanying text.
            tool_calls: Ordered list of ToolCall objects (empty when the model
                        responded with plain text).

        Raises:
            RuntimeError: Provider returned neither text nor tool calls (Ollama
                          empty-response guard), or the underlying API failed.
            NotImplementedError: Provider subclass has not implemented this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement complete_with_tools(). "
            "Override this method in the provider subclass."
        )

    # ── Retry helper (shared) ─────────────────────────────────────────────────

    def _complete_with_retry(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Template-method wrapper that adds retry + latency logging around
        the provider-specific _do_complete() implementation.

        Subclasses should implement _do_complete() instead of complete() when
        they want free retry/logging behaviour.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = self._do_complete(messages, temperature, max_tokens, **kwargs)
                latency = (time.monotonic() - t0) * 1000
                logger.debug(
                    "LLM call OK | provider=%s model=%s attempt=%d latency=%.0f ms "
                    "prompt_tokens=%s completion_tokens=%s",
                    self.__class__.__name__, self.model, attempt, latency,
                    resp.prompt_tokens, resp.completion_tokens,
                )
                return resp
            except _RetryableError as exc:
                wait = 2 ** attempt          # exponential back-off: 2, 4, 8 …
                logger.warning(
                    "Retryable error (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)
                last_exc = exc
            except (ValueError, RuntimeError):
                raise

        raise RuntimeError(
            f"LLM provider {self.__class__.__name__} failed after "
            f"{self.max_retries} retries"
        ) from last_exc

    def _do_complete(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Provider-specific implementation.  Override this when you want the
        base class to handle retries for you.  Otherwise override complete().
        """
        raise NotImplementedError

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier used in MappingMetadata."""
        return self.__class__.__name__.replace("Provider", "").lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel exception (internal use only)
# ─────────────────────────────────────────────────────────────────────────────


class _RetryableError(Exception):
    """Raised inside _do_complete to signal a transient error worth retrying."""


OPENAI_REASONING_EFFORT_BY_MODEL = {
    "gpt-5.1": "low",
    "gpt-5.6-luna": "medium",
}


def is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI reasoning-style model families."""
    normalized = model.lower().rsplit("/", 1)[-1]
    return normalized.startswith(("o1", "o3", "o4", "gpt-5"))


def openai_reasoning_effort_for_model(model: str) -> str | None:
    """Return the application-selected OpenAI reasoning effort for exact models."""
    normalized = model.lower().rsplit("/", 1)[-1]
    return OPENAI_REASONING_EFFORT_BY_MODEL.get(normalized)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete providers (thin wrappers — heavy SDK imports are guarded)
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIProvider(BaseLLMProvider):
    """
    Covers: OpenAI API, GitHub Models (inference.ai), Azure OpenAI.

    Environment variables (any one of):
      OPENAI_API_KEY        — standard OpenAI
      GITHUB_TOKEN          — GitHub Models (set base_url automatically)
      AZURE_OPENAI_API_KEY  — Azure (also set azure_endpoint kwarg)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,          # override for GitHub / Azure
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-02-15-preview",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._base_url         = base_url
        self._azure_endpoint   = azure_endpoint
        self._azure_api_version = azure_api_version
        self._client: Any      = None   # lazy — avoid import cost at module load

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "Install the 'openai' extra: pip install 'llm-ontology-mapper[openai]'"
                ) from exc

            if self._azure_endpoint:
                self._client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self._azure_endpoint,
                    api_version=self._azure_api_version,
                )
            else:
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self._base_url,
                )
        return self._client

    def _token_limit_kwargs(
        self,
        max_tokens: int,
        min_completion_tokens: int | None = None,
    ) -> dict[str, int]:
        """Return the Chat Completions token-limit parameter for this model."""
        if self._uses_reasoning_style_parameters():
            effective_tokens = max(max_tokens, min_completion_tokens or 0)
            return {"max_completion_tokens": effective_tokens}
        return {"max_tokens": max_tokens}

    def _temperature_kwargs(self, temperature: float) -> dict[str, float]:
        """Return sampling kwargs supported by this model family."""
        if self._uses_reasoning_style_parameters():
            return {}
        return {"temperature": temperature}

    def _uses_reasoning_style_parameters(self) -> bool:
        """True for OpenAI models that reject max_tokens or non-default temperature."""
        return is_reasoning_model(self.model)

    def _reasoning_kwargs(self, requested_effort: str | None) -> dict[str, str]:
        """Return conservative reasoning controls for compatible model families."""
        if not self._uses_reasoning_style_parameters():
            return {}
        if requested_effort:
            return {"reasoning_effort": requested_effort}
        model = self.model.lower().rsplit("/", 1)[-1]
        if model.startswith("gpt-5"):
            return {"reasoning_effort": "minimal"}
        return {"reasoning_effort": "low"}

    @staticmethod
    def _requires_max_completion_tokens(exc: Exception) -> bool:
        """Detect OpenAI's model-specific rejection of max_tokens."""
        message = str(exc).lower()
        return (
            "max_tokens" in message
            and "max_completion_tokens" in message
            and ("unsupported" in message or "not supported" in message)
        )

    @staticmethod
    def _rejects_parameter(exc: Exception, parameter_name: str) -> bool:
        """Detect provider rejection of an optional request parameter."""
        message = str(exc).lower()
        parameter = parameter_name.lower()
        return (
            parameter in message
            and (
                "unsupported" in message
                or "not supported" in message
                or "unrecognized" in message
                or "unknown parameter" in message
            )
        )

    def _create_chat_completion(
        self,
        client: Any,
        request_kwargs: dict[str, Any],
        *,
        max_tokens: int,
    ) -> Any:
        import openai  # noqa: PLC0415  # type: ignore[import-untyped]

        for _ in range(3):
            try:
                return client.chat.completions.create(**request_kwargs)
            except openai.APIStatusError as exc:
                if (
                    "max_tokens" in request_kwargs
                    and self._requires_max_completion_tokens(exc)
                ):
                    request_kwargs.pop("max_tokens")
                    request_kwargs["max_completion_tokens"] = max_tokens
                    continue
                if (
                    "reasoning_effort" in request_kwargs
                    and self._rejects_parameter(exc, "reasoning_effort")
                ):
                    request_kwargs.pop("reasoning_effort")
                    continue
                raise

        return client.chat.completions.create(**request_kwargs)

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        return self._complete_with_retry(messages, temperature, max_tokens, **kwargs)

    def _do_complete(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        import openai  # noqa: PLC0415  # type: ignore[import-untyped]

        client = self._get_client()
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        min_completion_tokens = kwargs.pop("min_completion_tokens", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        request_kwargs = {
            "model": self.model,
            "messages": sdk_messages,
            **self._temperature_kwargs(temperature),
            **self._token_limit_kwargs(
                max_tokens,
                min_completion_tokens=min_completion_tokens,
            ),
            **self._reasoning_kwargs(reasoning_effort),
            **kwargs,
        }
        try:
            resp = self._create_chat_completion(
                client,
                request_kwargs,
                max_tokens=max_tokens,
            )
        except openai.RateLimitError as exc:
            raise _RetryableError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        content = resp.choices[0].message.content or ""
        if not content.strip():
            logger.warning(
                "OpenAI returned empty assistant content | model=%s finish_reason=%s usage=%s",
                getattr(resp, "model", self.model),
                getattr(resp.choices[0], "finish_reason", None),
                _safe_usage_dict(getattr(resp, "usage", None)),
            )

        return CompletionResponse(
            content=content,
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
            completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            raw=resp,
        )

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str | None, list[ToolCall]]:
        import json as _json

        import openai  # noqa: PLC0415  # type: ignore[import-untyped]

        client = self._get_client()
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        provider_tools = to_provider_tools(self.provider_name, tools)
        min_completion_tokens = kwargs.pop("min_completion_tokens", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        request_kwargs = {
            "model": self.model,
            "messages": sdk_messages,
            "tools": provider_tools,
            **self._temperature_kwargs(temperature),
            **self._token_limit_kwargs(
                max_tokens,
                min_completion_tokens=min_completion_tokens,
            ),
            **self._reasoning_kwargs(reasoning_effort),
            **kwargs,
        }
        try:
            resp = self._create_chat_completion(
                client,
                request_kwargs,
                max_tokens=max_tokens,
            )
        except openai.RateLimitError as exc:
            raise _RetryableError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        msg = resp.choices[0].message
        text: str | None = msg.content or None
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_json.loads(tc.function.arguments),
            ))
        return text, tool_calls

    @property
    def provider_name(self) -> str:
        if self._azure_endpoint:
            return "azure"
        if self._base_url and "models.inference" in (self._base_url or ""):
            return "github"
        return "openai"


def _safe_usage_dict(usage: Any) -> dict[str, Any] | None:
    """Return non-sensitive token usage diagnostics from an SDK usage object."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result or None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude models."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "Install the 'anthropic' extra: pip install 'llm-ontology-mapper[anthropic]'"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        return self._complete_with_retry(messages, temperature, max_tokens, **kwargs)

    def _do_complete(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        import anthropic  # noqa: PLC0415  # type: ignore[import-untyped]

        client = self._get_client()
        # Anthropic separates system from user messages
        system = next((m.content for m in messages if m.role == "system"), "")
        sdk_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        try:
            resp = client.messages.create(
                model=self.model,
                system=system,
                messages=sdk_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except anthropic.RateLimitError as exc:
            raise _RetryableError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        content = resp.content[0].text if resp.content else ""
        return CompletionResponse(
            content=content,
            model=resp.model,
            prompt_tokens=resp.usage.input_tokens if resp.usage else None,
            completion_tokens=resp.usage.output_tokens if resp.usage else None,
            raw=resp,
        )

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str | None, list[ToolCall]]:
        import anthropic  # noqa: PLC0415  # type: ignore[import-untyped]

        client = self._get_client()
        system = next((m.content for m in messages if m.role == "system"), "")
        sdk_messages = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        provider_tools = to_provider_tools("anthropic", tools)
        try:
            resp = client.messages.create(
                model=self.model,
                system=system,
                messages=sdk_messages,
                tools=provider_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except anthropic.RateLimitError as exc:
            raise _RetryableError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        text: str | None = None
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        return text, tool_calls

    @property
    def provider_name(self) -> str:
        return "anthropic"


class OllamaProvider(BaseLLMProvider):
    """
    Ollama models — local or remote server (llama3, mistral, phi3, …).

    Args:
        model:    Ollama model name, e.g. "llama3", "mistral", "phi3".
        base_url: Full URL to the Ollama server, e.g.
                  "http://localhost:11434" (default) or
                  "http://gpu-vm.internal:11434".
        api_key:  Optional Bearer token for Ollama servers protected by an
                  authentication proxy.  Sent as the Authorization header.
        **kwargs: Forwarded to BaseLLMProvider (timeout, max_retries, …).
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._base_url = base_url  # None → ollama SDK uses localhost:11434

    def _get_client(self) -> Any:
        try:
            import ollama  # noqa: PLC0415  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install the 'ollama' extra: pip install 'llm-ontology-mapper[ollama]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._base_url:
            kwargs["host"] = self._base_url
        if self.api_key:
            kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
        return ollama.Client(**kwargs) if kwargs else ollama

    def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        return self._complete_with_retry(messages, temperature, max_tokens, **kwargs)

    def _do_complete(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        client = self._get_client()
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            resp = client.chat(
                model=self.model,
                messages=sdk_messages,
                options={"temperature": temperature},
                format="json",
            )
        except Exception as exc:
            # Ollama raises plain exceptions for connection errors — treat as retryable
            err = str(exc).lower()
            if any(kw in err for kw in ("connection", "timeout", "refused", "reset")):
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        return CompletionResponse(content=content, model=self.model, raw=resp)

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str | None, list[ToolCall]]:
        client = self._get_client()
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        provider_tools = to_provider_tools("ollama", tools)
        try:
            resp = client.chat(
                model=self.model,
                messages=sdk_messages,
                tools=provider_tools,
                options={"temperature": temperature},
                # format="json" intentionally omitted for tool-calling mode
            )
        except Exception as exc:
            err = str(exc).lower()
            if any(kw in err for kw in ("connection", "timeout", "refused", "reset")):
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        msg = resp["message"] if isinstance(resp, dict) else resp.message
        if isinstance(msg, dict):
            content: str | None = msg.get("content") or None
            raw_tool_calls = msg.get("tool_calls") or []
        else:
            content = msg.content or None
            raw_tool_calls = msg.tool_calls or []

        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", {}),
                ))
            else:
                tool_calls.append(ToolCall(
                    id=getattr(tc, "id", ""),
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        if not tool_calls and not content:
            raise RuntimeError(
                "Ollama complete_with_tools: model returned neither tool calls nor text content"
            )

        return content, tool_calls

    @property
    def provider_name(self) -> str:
        return "ollama"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


class LLMProviderFactory:
    """
    Construct a BaseLLMProvider from a simple config dict or keyword arguments.

    Usage::

        provider = LLMProviderFactory.from_config(
            provider="openai",
            model="gpt-4o",
            api_key="sk-...",
        )
        # GitHub Models:
        provider = LLMProviderFactory.from_config(
            provider="github",
            model="gpt-4o",
            api_key="ghp_...",
        )
        # Local Ollama (default localhost:11434):
        provider = LLMProviderFactory.from_config(
            provider="ollama",
            model="llama3",
        )
        # Remote Ollama on a VM or GPU server:
        provider = LLMProviderFactory.from_config(
            provider="ollama",
            model="llama3",
            base_url="http://gpu-vm.internal:11434",
            api_key="token-if-protected",   # optional
        )
    """

    _GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

    @classmethod
    def from_config(
        cls,
        provider: str,
        model: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Args:
            provider: One of "openai" | "github" | "azure" | "anthropic" | "ollama"
            model:    Model identifier (passed through to the backend)
            api_key:  API key / token (reads from env if None).
                      For Ollama, used as a Bearer token when the server is
                      behind an auth proxy.
            **kwargs: Forwarded to the provider constructor.
                      OpenAI/Azure: azure_endpoint=, azure_api_version=
                      Ollama:       base_url="http://host:11434"
        """
        p = provider.lower()

        if p == "openai":
            return OpenAIProvider(model=model, api_key=api_key, **kwargs)

        if p == "github":
            return OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=cls._GITHUB_BASE_URL,
                **kwargs,
            )

        if p == "azure":
            return OpenAIProvider(model=model, api_key=api_key, **kwargs)

        if p == "anthropic":
            return AnthropicProvider(model=model, api_key=api_key, **kwargs)

        if p in ("ollama", "local"):
            return OllamaProvider(model=model, api_key=api_key, **kwargs)

        raise ValueError(
            f"Unknown provider {provider!r}. "
            "Choose from: openai, github, azure, anthropic, ollama"
        )
