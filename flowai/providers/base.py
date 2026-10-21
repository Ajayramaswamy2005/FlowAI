"""Provider Abstraction Layer - Core data structures and BaseProvider."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field

from flowai.errors import ConfigurationError, ProviderError


class ToolCall(BaseModel):
    """Represents a model tool/function call request."""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of an executed tool call."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """Unified message structure across all LLM providers."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ProviderUsage(BaseModel):
    """Token and cost telemetry for a model invocation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ProviderResponse(BaseModel):
    """Standard non-streaming completion output."""
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: ProviderUsage = Field(default_factory=ProviderUsage)


class StreamChunk(BaseModel):
    """Incremental chunk emitted during streaming."""
    text_delta: str = ""
    tool_call_delta: ToolCall | None = None
    finish_reason: str | None = None
    usage: ProviderUsage | None = None


class BaseProvider(ABC):
    """Abstract Base Class for all LLM Provider implementations in FlowAI.

    Subclasses must implement:
      - generate(): non-streaming text/tool generation
      - stream(): asynchronous token/tool streaming
      - embeddings(): vector embedding generation

    Streaming contract: a provider yields text as `text_delta` chunks, and
    emits every tool call the model requested exactly once as a *complete*
    `tool_call_delta` (fragments are assembled inside the provider). Agents
    rely on that, so they can act on a streamed tool call.
    """

    #: Environment variable holding this provider's key, if it needs one.
    api_key_env: str | None = None
    #: Local providers such as Ollama need no key.
    requires_api_key: bool = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.extra_kwargs = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Execute a non-streaming completion."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Execute a streaming completion yielding token/tool deltas."""
        ...

    @abstractmethod
    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Compute text vector embeddings."""
        ...

    @property
    def provider_name(self) -> str:
        """Short lowercase name, e.g. 'anthropic' for AnthropicProvider."""
        return self.__class__.__name__.replace("Provider", "").lower()

    def require_api_key(self) -> None:
        """Fail early, and in plain words, when the key for this provider is missing."""
        if not self.requires_api_key or self.api_key:
            return
        env_hint = f" Set {self.api_key_env}," if self.api_key_env else ""
        raise ConfigurationError(
            f"No API key for the {self.provider_name} provider.{env_hint}"
            " or run 'flowai configure' to save one."
        )

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Turn an HTTP error into a ProviderError that quotes the API's own message."""
        if response.status_code < 400:
            return
        # A streamed response has not been read yet; read it before touching .text.
        if not response.is_closed:
            await response.aread()
        detail = response.text.strip()
        try:
            payload = json.loads(detail)
            error = payload.get("error", payload)
            if isinstance(error, dict):
                detail = error.get("message") or json.dumps(error)
            elif isinstance(error, str):
                detail = error
        except (json.JSONDecodeError, AttributeError):
            pass
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise ProviderError(self.provider_name, response.status_code, detail or response.reason_phrase)

    def parse_tool_call(self, raw_call: dict[str, Any]) -> ToolCall:
        """Helper to safely parse tool call dictionary into ToolCall object."""
        tc_id = raw_call.get("id", "")
        function_data = raw_call.get("function", {})
        name = function_data.get("name", "")
        args_raw = function_data.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"raw": args_raw}
        else:
            args = args_raw or {}
        return ToolCall(id=tc_id, name=name, arguments=args)
