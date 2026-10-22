"""Google Gemini Provider implementation."""

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
from flowai.security.keystore import resolve_api_key


class GeminiProvider(BaseProvider):
    """Google Gemini Provider (Gemini 2.5 Flash, Gemini 1.5 Pro, etc.)."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env = "GEMINI_API_KEY"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        resolved_key = api_key or resolve_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        resolved_url = base_url or os.getenv("GEMINI_BASE_URL", self.DEFAULT_BASE_URL)
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        # The key travels in a header, never in the URL: URLs end up in error
        # messages, proxy logs and tracebacks.
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key or ""}

    def _convert_messages(self, messages: Sequence[Message]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_instruction = None
        contents = []

        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content or ""}]}
            elif m.role == "tool":
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": m.name or "tool",
                            "response": {"result": m.content or ""},
                        }
                    }],
                })
            elif m.role == "assistant":
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    })
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": m.content or ""}]})

        return system_instruction, contents

    def _convert_tools(self, tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate OpenAI-style tool schemas into Gemini functionDeclarations."""
        declarations = []
        for t in tools:
            fn = t.get("function", t)
            parameters = dict(fn.get("parameters") or {"type": "object", "properties": {}})
            # Gemini rejects an empty-property object schema, so omit it entirely.
            if not parameters.get("properties"):
                parameters = None
            declaration = {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
            }
            if parameters:
                declaration["parameters"] = parameters
            declarations.append(declaration)
        return [{"functionDeclarations": declarations}]

    @staticmethod
    def _usage_from(meta: dict[str, Any]) -> ProviderUsage:
        return ProviderUsage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        )

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        system_instruction, contents = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = self._convert_tools(tools)

        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            await self._raise_for_status(resp)
            data = resp.json()

        candidates = data.get("candidates", [])
        content_text = ""
        tool_calls: list[ToolCall] = []

        if candidates:
            cand = candidates[0]
            parts = cand.get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    content_text += p["text"]
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=fc.get("name", "call"),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                        )
                    )

        usage = self._usage_from(data.get("usageMetadata", {}))

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason="stop",
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
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:streamGenerateContent?alt=sse"
        system_instruction, contents = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = self._convert_tools(tools)

        self.require_api_key()
        call_index = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "text" in part:
                                yield StreamChunk(text_delta=part["text"])
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                call_index += 1
                                yield StreamChunk(
                                    tool_call_delta=ToolCall(
                                        id=f"{fc.get('name', 'call')}_{call_index}",
                                        name=fc.get("name", ""),
                                        arguments=fc.get("args", {}) or {},
                                    )
                                )
                        if candidate.get("finishReason"):
                            yield StreamChunk(finish_reason=candidate["finishReason"])

                    if data.get("usageMetadata"):
                        yield StreamChunk(usage=self._usage_from(data["usageMetadata"]))

    async def embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        url = f"{self.base_url.rstrip('/')}/models/{model or 'text-embedding-004'}:batchEmbedContents"
        requests = [{"model": f"models/{model or 'text-embedding-004'}", "content": {"parts": [{"text": t}]}} for t in texts]
        self.require_api_key()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers(), json={"requests": requests})
            await self._raise_for_status(resp)
            data = resp.json()
        return [e["values"] for e in data.get("embeddings", [])]
