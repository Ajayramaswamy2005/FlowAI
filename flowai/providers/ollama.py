"""Ollama Local Inference Provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from flowai.errors import ConfigurationError
from flowai.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    ProviderUsage,
    StreamChunk,
    ToolCall,
)


class OllamaProvider(BaseProvider):
    """Local Ollama Provider running at http://localhost:11434."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    requires_api_key = False

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=resolved_url,
            timeout=timeout,
            **kwargs,
        )

    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in m.tool_calls
                ]
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
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            # Transform OpenAI-style tools to Ollama tools if needed
            ollama_tools = []
            for t in tools:
                if "function" in t:
                    ollama_tools.append(t)
                else:
                    ollama_tools.append({"type": "function", "function": t})
            payload["tools"] = ollama_tools

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                await self._raise_for_status(resp)
                data = resp.json()
        except httpx.ConnectError as exc:
            raise self._not_running(exc) from exc

        raw_msg = data.get("message", {})
        content = raw_msg.get("content", "")
        tool_calls: list[ToolCall] = []

        if raw_msg.get("tool_calls"):
            for i, tc in enumerate(raw_msg["tool_calls"]):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}_{fn.get('name')}",
                        name=fn.get("name", ""),
                        arguments=args,
                    )
                )

        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        usage = ProviderUsage(
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=prompt_eval + eval_count,
        )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop" if data.get("done") else "length",
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
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    await self._raise_for_status(response)
                    call_index = 0
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        raw_msg = data.get("message", {})
                        content_delta = raw_msg.get("content", "")
                        done = bool(data.get("done"))

                        # Ollama sends each tool call whole, so emit them all.
                        for raw_tc in raw_msg.get("tool_calls") or []:
                            fn = raw_tc.get("function", {})
                            call_index += 1
                            yield StreamChunk(
                                tool_call_delta=ToolCall(
                                    id=f"call_{call_index}_{fn.get('name', '')}",
                                    name=fn.get("name", ""),
                                    arguments=self._parse_arguments(fn.get("arguments", {})),
                                )
                            )

                        if content_delta or done:
                            yield StreamChunk(
                                text_delta=content_delta,
                                finish_reason="stop" if done else None,
                                usage=self._usage_from(data) if done else None,
                            )
        except httpx.ConnectError as exc:
            raise self._not_running(exc) from exc

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/api/embed"
        payload = {
            "model": model or self.model,
            "input": texts,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                await self._raise_for_status(resp)
                data = resp.json()
        except httpx.ConnectError as exc:
            raise self._not_running(exc) from exc
        return data.get("embeddings", [])

    def _not_running(self, exc: Exception) -> ConfigurationError:
        """Ollama is local: a refused connection means it is not started."""
        return ConfigurationError(
            f"Cannot reach Ollama at {self.base_url} ({exc}). "
            "Start it with 'ollama serve', or point FlowAI elsewhere with OLLAMA_BASE_URL."
        )

    @staticmethod
    def _parse_arguments(args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return {"raw": args}
        return args if isinstance(args, dict) else {"value": args}

    @staticmethod
    def _usage_from(data: dict[str, Any]) -> ProviderUsage:
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        return ProviderUsage(
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=prompt_eval + eval_count,
        )
