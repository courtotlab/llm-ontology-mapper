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
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=sdk_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except openai.RateLimitError as exc:
            raise _RetryableError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _RetryableError(str(exc)) from exc
            raise RuntimeError(str(exc)) from exc

        return CompletionResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
            completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            raw=resp,
        )

    @property
    def provider_name(self) -> str:
        if self._azure_endpoint:
            return "azure"
        if self._base_url and "models.inference" in (self._base_url or ""):
            return "github"
        return "openai"


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
