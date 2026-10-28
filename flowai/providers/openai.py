"""OpenAI-compatible providers: OpenAI itself, OpenRouter, and (in nvidia.py) NVIDIA.

Many servers speak the chat-completions protocol, but not all of them accept
every field. A model on NVIDIA's catalog may refuse `tools`, a self-hosted
server may refuse `stream_options`, an older model may refuse the `system`
role. Rather than fail, the provider notices which field the error names,
stops sending it, tells the user, and tries again. Rate limits and transient
server errors are retried with backoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
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
from flowai.security.keystore import resolve_api_key

#: Statuses worth waiting out: rate limits and a busy or restarting server.
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_RETRY_DELAY = 30.0

#: Optional request features, and how to recognise an error complaining about each.
OPTIONAL_FEATURES: list[tuple[str, re.Pattern[str], str]] = [
    ("stream_options", re.compile(r"stream_options|include_usage", re.I), "token usage reporting"),
    ("tools", re.compile(r"\btools?\b|tool[ _-]?(?:use|call|choice)|function[ _-]?call", re.I), "tool calling"),
    ("system_role", re.compile(r"\bsystem\b", re.I), "the system role"),
    ("max_tokens", re.compile(r"max_tokens|max_completion_tokens|maximum context", re.I), "max_tokens"),
    ("temperature", re.compile(r"temperature", re.I), "temperature"),
]
FEATURE_LABELS = {name: label for name, _, label in OPTIONAL_FEATURES}

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


async def _backoff(seconds: float) -> None:
    await asyncio.sleep(seconds)


class ThinkSplitter:
    """Separates <think>...</think> reasoning from the answer in streamed text.

    Reasoning models served over plain chat-completions (DeepSeek-R1 and
    friends on NVIDIA's catalog) put their thinking inline. Tags can be split
    across chunks, so a possible partial tag is held back until it resolves.
    """

    def __init__(self) -> None:
        self.inside = False
        self.buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        self.buffer += text
        answer: list[str] = []
        thought: list[str] = []
        while self.buffer:
            tag = THINK_CLOSE if self.inside else THINK_OPEN
            index = self.buffer.find(tag)
            target = thought if self.inside else answer
            if index >= 0:
                target.append(self.buffer[:index])
                self.buffer = self.buffer[index + len(tag):]
                self.inside = not self.inside
                continue
            keep = next((n for n in range(len(tag) - 1, 0, -1) if self.buffer.endswith(tag[:n])), 0)
            target.append(self.buffer[: len(self.buffer) - keep])
            self.buffer = self.buffer[len(self.buffer) - keep:]
            break
        return "".join(answer), "".join(thought)

    def flush(self) -> tuple[str, str]:
        rest, self.buffer = self.buffer, ""
        return ("", rest) if self.inside else (rest, "")


def split_thinking(text: str) -> tuple[str, str]:
    """(answer, reasoning) for a complete, non-streamed reply."""
    splitter = ThinkSplitter()
    answer, thought = splitter.feed(text)
    tail_answer, tail_thought = splitter.flush()
    return (answer + tail_answer).strip(), (thought + tail_thought).strip()


class OpenAIProvider(BaseProvider):
    """Standard OpenAI Provider (GPT-4o, GPT-4o-mini, o1, etc.)."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"
    api_key_env = "OPENAI_API_KEY"
    #: Environment names this provider's key is read from. Each subclass names
    #: its own, so one provider's key is never sent to another provider's server.
    key_env_names: tuple[str, ...] = ("OPENAI_API_KEY",)
    base_url_env_names: tuple[str, ...] = ("OPENAI_BASE_URL",)
    #: Sent when the caller gives no max_tokens; some servers default very low.
    default_max_tokens: int | None = None

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        env_url = next((os.environ[n] for n in self.base_url_env_names if os.getenv(n)), None)
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            api_key=api_key or resolve_api_key(*self.key_env_names),
            base_url=base_url or env_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            **kwargs,
        )
        #: Features this server has rejected; they are left out from now on.
        self.unsupported: set[str] = set()
        self._notices: list[str] = []

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------- payloads

    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        flatten_tools = "tools" in self.unsupported
        out: list[dict[str, Any]] = []
        for m in messages:
            if flatten_tools and m.role == "tool":
                # Without tool support the server rejects the tool role, so a
                # result from earlier in the conversation is replayed as text.
                out.append({"role": "user", "content": f"[result of tool {m.name}]\n{m.content or ''}"})
                continue
            msg: dict[str, Any] = {"role": m.role}
            content = m.content
            if m.tool_calls and flatten_tools:
                calls = "; ".join(f"{tc.name}({json.dumps(tc.arguments)})" for tc in m.tool_calls)
                content = f"{content or ''}\n[called tools: {calls}]".strip()
            if content is not None:
                msg["content"] = content
            if m.tool_calls and not flatten_tools:
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
            if m.tool_call_id and not flatten_tools:
                msg["tool_call_id"] = m.tool_call_id
            if m.name and m.role == "tool" and not flatten_tools:
                msg["name"] = m.name
            out.append(msg)
        if "system_role" in self.unsupported:
            out = self._merge_system(out)
        return out

    @staticmethod
    def _merge_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold system messages into the first user message, for models without a system role."""
        system = "\n\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if not system:
            return rest
        for m in rest:
            if m["role"] == "user":
                m["content"] = f"{system}\n\n{m.get('content') or ''}"
                return rest
        return [{"role": "user", "content": system}, *rest]

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        off = self.unsupported
        payload: dict[str, Any] = {"model": self.model, "messages": self._convert_messages(messages)}
        if "temperature" not in off:
            payload["temperature"] = temperature
        limit = max_tokens or self.default_max_tokens
        if limit and "max_tokens" not in off:
            payload["max_tokens"] = limit
        if tools and "tools" not in off:
            payload["tools"] = list(tools)
        if stream:
            payload["stream"] = True
            if "stream_options" not in off:
                payload["stream_options"] = {"include_usage": True}
        return payload

    # ------------------------------------------------------------- recovery

    @staticmethod
    def _features_in(payload: dict[str, Any]) -> list[str]:
        present = [name for name in ("stream_options", "tools", "max_tokens", "temperature") if name in payload]
        if any(m.get("role") == "system" for m in payload.get("messages", [])):
            present.append("system_role")
        return present

    def _rejected_feature(self, message: str, payload: dict[str, Any]) -> str | None:
        present = self._features_in(payload)
        for name, pattern, _ in OPTIONAL_FEATURES:
            if name in present and pattern.search(message):
                return name
        return None

    async def _recover(self, response: httpx.Response, payload: dict[str, Any], retries: int) -> str:
        """Decide what to do about an error response: 'retry', 'degrade', or raise."""
        status = response.status_code
        if status in RETRY_STATUSES and retries < MAX_RETRIES:
            if not response.is_closed:
                await response.aread()
            await _backoff(self._retry_delay(response, retries))
            return "retry"
        try:
            await self._raise_for_status(response)
        except ProviderError as exc:
            if status in (400, 404, 422):
                feature = self._rejected_feature(exc.message, payload)
                if feature:
                    self.unsupported.add(feature)
                    self._notices.append(
                        f"{self.model} does not accept {FEATURE_LABELS[feature]}; continuing without it."
                    )
                    return "degrade"
            raise
        return "retry"

    @staticmethod
    def _retry_delay(response: httpx.Response, retries: int) -> float:
        header = response.headers.get("retry-after", "")
        try:
            return min(float(header), MAX_RETRY_DELAY)
        except ValueError:
            return min(1.5 * (2 ** retries) + random.uniform(0, 0.5), MAX_RETRY_DELAY)

    def take_notices(self) -> list[str]:
        notices, self._notices = self._notices, []
        return notices

    # ------------------------------------------------------------- calls

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        self.require_api_key()
        retries = 0
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while True:
                    payload = self._payload(messages, tools, temperature, max_tokens, stream=False)
                    resp = await client.post(url, headers=self._get_headers(), json=payload)
                    if resp.status_code < 400:
                        break
                    if await self._recover(resp, payload, retries) == "retry":
                        retries += 1
                data = resp.json()
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        choice = (data.get("choices") or [{}])[0]
        raw_msg = choice.get("message") or {}
        content, inline_thought = split_thinking(raw_msg.get("content") or "")
        reasoning = raw_msg.get("reasoning_content") or raw_msg.get("reasoning") or inline_thought

        tool_calls = [self.parse_tool_call(tc) for tc in raw_msg.get("tool_calls") or []]
        for tc in tool_calls:
            tc.id = tc.id or f"call_{uuid.uuid4().hex[:12]}"

        raw_usage = data.get("usage") or {}
        usage = ProviderUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return ProviderResponse(
            content=content,
            reasoning=reasoning or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "stop",
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
        self.require_api_key()
        retries = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while True:
                    payload = self._payload(messages, tools, temperature, max_tokens, stream=True)
                    async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                        if response.status_code >= 400:
                            if await self._recover(response, payload, retries) == "retry":
                                retries += 1
                            continue
                        for notice in self.take_notices():
                            yield StreamChunk(notice=notice)
                        async for chunk in self._read_stream(response):
                            yield chunk
                        return
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    async def _read_stream(self, response: httpx.Response) -> AsyncIterator[StreamChunk]:
        # Tool calls arrive in fragments keyed by index: the name in one chunk,
        # the arguments spread over many. They are assembled here and emitted
        # once complete, so the agent always sees a whole call.
        pending: dict[int, dict[str, str]] = {}
        splitter = ThinkSplitter()

        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk_dict = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if chunk_dict.get("error"):
                error = chunk_dict["error"]
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise ProviderError(self.provider_name, None, message or "stream error")

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
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            for raw_tc in delta.get("tool_calls") or []:
                index = raw_tc.get("index", 0)
                slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if raw_tc.get("id"):
                    slot["id"] = raw_tc["id"]
                fn = raw_tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                args = fn.get("arguments")
                if args:
                    slot["arguments"] += args if isinstance(args, str) else json.dumps(args)

            text_delta, inline_thought = splitter.feed(delta.get("content") or "")
            reasoning = (delta.get("reasoning_content") or delta.get("reasoning") or "") + inline_thought
            if text_delta or reasoning or usage:
                yield StreamChunk(text_delta=text_delta, reasoning_delta=reasoning, usage=usage)

            if finish_reason:
                for slot in pending.values():
                    yield StreamChunk(tool_call_delta=self._assemble_tool_call(slot))
                pending.clear()
                yield StreamChunk(finish_reason=finish_reason)

        tail_text, tail_thought = splitter.flush()
        if tail_text or tail_thought:
            yield StreamChunk(text_delta=tail_text, reasoning_delta=tail_thought)
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
        # Some servers omit the id; the follow-up tool message needs one to refer to.
        call_id = slot.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        return ToolCall(id=call_id, name=slot.get("name") or "", arguments=arguments)

    async def list_models(self) -> list[str]:
        url = f"{self.base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self._get_headers())
            await self._raise_for_status(resp)
            data = resp.json()
        return sorted(item.get("id", "") for item in data.get("data", []) if item.get("id"))

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
    DEFAULT_MODEL = "deepseek/deepseek-r1:free"
    api_key_env = "OPENROUTER_API_KEY"
    key_env_names = ("OPENROUTER_API_KEY",)
    base_url_env_names = ("OPENROUTER_BASE_URL",)

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        headers["HTTP-Referer"] = "https://github.com/Ajayramaswamy2005/FlowAI"
        headers["X-Title"] = "FlowAI Agent Harness"
        return headers
