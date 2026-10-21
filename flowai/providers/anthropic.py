"""Anthropic Claude Provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from flowai.errors import ProviderError
from flowai.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    ProviderUsage,
    StreamChunk,
    ToolCall,
)


class AnthropicProvider(BaseProvider):
    """Anthropic Provider for the Claude model family."""

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        resolved_url = base_url or os.getenv("ANTHROPIC_BASE_URL", self.DEFAULT_BASE_URL)
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
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _convert_messages(self, messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_prompts = []
        anthropic_messages = []

        for m in messages:
            if m.role == "system":
                if m.content:
                    system_prompts.append(m.content)
            elif m.role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content or "",
                        }
                    ],
                })
            elif m.role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_messages.append({"role": "user", "content": m.content or ""})

        system = "\n\n".join(system_prompts)
        return system, anthropic_messages

    def _convert_tools(self, tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        anthropic_tools = []
        for t in tools:
            # Handle standard OpenAI format or direct schema
            if "function" in t:
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                anthropic_tools.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        url = f"{self.base_url.rstrip('/')}/messages"
        system, formatted_msgs = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._convert_tools(tools)

        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()

        text_parts = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        raw_usage = data.get("usage", {})
        usage = ProviderUsage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
        )

        return ProviderResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", "end_turn"),
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
        url = f"{self.base_url.rstrip('/')}/messages"
        system, formatted_msgs = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._convert_tools(tools)

        self.require_api_key()
        # A tool_use block arrives as a start event, then its input as a run of
        # partial JSON deltas; it is assembled here and emitted once, complete.
        pending: dict[int, dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    ev_type = event.get("type")

                    if ev_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            pending[event.get("index", 0)] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "json": "",
                            }

                    elif ev_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(text_delta=delta.get("text", ""))
                        elif delta.get("type") == "input_json_delta":
                            slot = pending.get(event.get("index", 0))
                            if slot is not None:
                                slot["json"] += delta.get("partial_json", "")

                    elif ev_type == "content_block_stop":
                        slot = pending.pop(event.get("index", 0), None)
                        if slot:
                            yield StreamChunk(tool_call_delta=self._assemble_tool_call(slot))

                    elif ev_type == "message_delta":
                        raw_usage = event.get("usage") or {}
                        if raw_usage:
                            yield StreamChunk(
                                usage=ProviderUsage(
                                    prompt_tokens=raw_usage.get("input_tokens", 0),
                                    completion_tokens=raw_usage.get("output_tokens", 0),
                                    total_tokens=raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
                                ),
                                finish_reason=(event.get("delta") or {}).get("stop_reason"),
                            )

                    elif ev_type == "message_stop":
                        yield StreamChunk(finish_reason="end_turn")

                    elif ev_type == "error":
                        detail = (event.get("error") or {}).get("message", "unknown streaming error")
                        raise ProviderError(self.provider_name, None, detail)

    @staticmethod
    def _assemble_tool_call(slot: dict[str, Any]) -> ToolCall:
        """Turn an accumulated tool_use block into a complete ToolCall."""
        raw = slot.get("json") or ""
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            arguments = {"raw": raw}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return ToolCall(id=slot.get("id", ""), name=slot.get("name", ""), arguments=arguments)

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        raise NotImplementedError("Anthropic does not natively host vector embeddings API; use an embedding provider.")
