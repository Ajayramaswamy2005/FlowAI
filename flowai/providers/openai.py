"""OpenAI and OpenRouter Provider implementations."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from flowai.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    ProviderUsage,
    StreamChunk,
    ToolCall,
)


class OpenAIProvider(BaseProvider):
    """Standard OpenAI Provider (GPT-4o, GPT-4o-mini, o1, etc.)."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                msg["content"] = m.content
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            out.append(msg)
        return out

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()

        choice = data["choices"][0]
        raw_msg = choice.get("message", {})
        content = raw_msg.get("content") or ""

        tool_calls: list[ToolCall] = []
        if raw_msg.get("tool_calls"):
            for tc in raw_msg["tool_calls"]:
                tool_calls.append(self.parse_tool_call(tc))

        raw_usage = data.get("usage", {})
        usage = ProviderUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=usage,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        self.require_api_key()

        # Tool calls arrive in fragments keyed by index: the name in one chunk,
        # the arguments spread over many. They are assembled here and emitted
        # once complete, so the agent always sees a whole call.
        pending: dict[int, dict[str, str]] = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_dict = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = None
                    if chunk_dict.get("usage"):
                        raw = chunk_dict["usage"]
                        usage = ProviderUsage(
                            prompt_tokens=raw.get("prompt_tokens", 0),
                            completion_tokens=raw.get("completion_tokens", 0),
                            total_tokens=raw.get("total_tokens", 0),
                        )

                    choices = chunk_dict.get("choices") or []
                    if not choices:
                        if usage:
                            yield StreamChunk(usage=usage)
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    for raw_tc in delta.get("tool_calls") or []:
                        index = raw_tc.get("index", 0)
                        slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if raw_tc.get("id"):
                            slot["id"] = raw_tc["id"]
                        fn = raw_tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

                    text_delta = delta.get("content") or ""
                    if text_delta or usage:
                        yield StreamChunk(text_delta=text_delta, usage=usage)

                    if finish_reason:
                        for slot in pending.values():
                            yield StreamChunk(tool_call_delta=self._assemble_tool_call(slot))
                        pending.clear()
                        yield StreamChunk(finish_reason=finish_reason)

        # A stream cut short before finish_reason still carries usable calls.
        for slot in pending.values():
            yield StreamChunk(tool_call_delta=self._assemble_tool_call(slot))

    @staticmethod
    def _assemble_tool_call(slot: dict[str, str]) -> ToolCall:
        """Turn an accumulated streaming fragment into a complete ToolCall."""
        raw_args = slot.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            arguments = {"raw": raw_args}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return ToolCall(id=slot.get("id") or "", name=slot.get("name") or "", arguments=arguments)

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload = {
            "input": texts,
            "model": model or "text-embedding-3-small",
        }
        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter Provider for routing across hundreds of open/commercial models."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"

    def __init__(
        self,
        model: str = "deepseek/deepseek-r1:free",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        resolved_url = base_url or os.getenv("OPENROUTER_BASE_URL", self.DEFAULT_BASE_URL)
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        headers["HTTP-Referer"] = "https://github.com/Ajayramaswamy2005/FlowAI"
        headers["X-Title"] = "FlowAI Agent Harness"
        return headers
