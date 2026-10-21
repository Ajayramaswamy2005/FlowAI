"""Streaming: providers assemble tool calls, and a streamed agent runs them."""

import json

import httpx
import pytest

from flowai.agents import Agent
from flowai.errors import ConfigurationError, ProviderError
from flowai.providers.anthropic import AnthropicProvider
from flowai.providers.base import BaseProvider, Message, ProviderResponse, StreamChunk, ToolCall
from flowai.providers.openai import OpenAIProvider
from flowai.tools import tool


def sse(events):
    return "".join(f"data: {json.dumps(e) if not isinstance(e, str) else e}\n\n" for e in events)


@pytest.fixture
def fake_http(monkeypatch):
    """Route every httpx.AsyncClient through a handler returning a canned body."""
    state = {}

    def install(body: str, status: int = 200):
        def handler(request):
            state["request"] = json.loads(request.content)
            return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

        real = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        return state

    return install


async def collect(provider, tools=None):
    return [c async for c in provider.stream([Message(role="user", content="hi")], tools=tools)]


@pytest.mark.asyncio
async def test_openai_stream_assembles_fragmented_tool_calls(fake_http):
    body = sse([
        {"choices": [{"delta": {"content": "Let me check."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ris"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "now", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        "[DONE]",
    ])
    state = fake_http(body)
    chunks = await collect(OpenAIProvider(api_key="k"), tools=[{"type": "function", "function": {"name": "x"}}])

    calls = [c.tool_call_delta for c in chunks if c.tool_call_delta]
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_a", "get_weather", {"city": "Paris"}),
        ("call_b", "now", {}),
    ]
    assert "".join(c.text_delta for c in chunks) == "Let me check."
    assert any(c.usage and c.usage.total_tokens == 15 for c in chunks)
    assert state["request"]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_anthropic_stream_assembles_tool_use(fake_http):
    body = sse([
        {"type": "message_start", "message": {}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Reading."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": "RE'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'ADME.md"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ])
    fake_http(body)
    chunks = await collect(AnthropicProvider(api_key="k"))
    calls = [c.tool_call_delta for c in chunks if c.tool_call_delta]
    assert len(calls) == 1
    assert (calls[0].id, calls[0].name, calls[0].arguments) == ("tu_1", "read_file", {"path": "README.md"})
    assert "".join(c.text_delta for c in chunks) == "Reading."


@pytest.mark.asyncio
async def test_http_errors_quote_the_api_message(fake_http):
    fake_http(json.dumps({"error": {"message": "invalid x-api-key"}}), status=401)
    with pytest.raises(ProviderError, match="HTTP 401.*invalid x-api-key"):
        await collect(AnthropicProvider(api_key="bad"))


@pytest.mark.asyncio
async def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY.*flowai configure"):
        await collect(OpenAIProvider(api_key=""))


class ScriptedStreamProvider(BaseProvider):
    """Streams one scripted turn per call: a list of chunks each time."""

    def __init__(self, turns):
        super().__init__(model="scripted")
        self.turns = turns
        self.seen: list[list[Message]] = []

    async def generate(self, messages, tools=None, **kwargs):
        return ProviderResponse(content="")

    async def stream(self, messages, tools=None, **kwargs):
        self.seen.append(list(messages))
        for chunk in self.turns[len(self.seen) - 1]:
            yield chunk

    async def embeddings(self, texts, model=None, **kwargs):
        return []


@pytest.mark.asyncio
async def test_streamed_agent_runs_tools_and_continues():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    provider = ScriptedStreamProvider([
        [StreamChunk(text_delta="Adding. "), StreamChunk(tool_call_delta=ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}))],
        [StreamChunk(text_delta="The sum is 5.")],
    ])
    agent = Agent(provider=provider, tools=[add])
    events = []
    agent.event_bus.subscribe(lambda e: events.append(e.event))

    text = "".join([c.text_delta async for c in agent.stream("2+3?")])

    assert text == "Adding. The sum is 5."
    assert len(provider.seen) == 2
    tool_msg = provider.seen[1][-1]
    assert tool_msg.role == "tool" and tool_msg.content == "5" and tool_msg.tool_call_id == "c1"
    assert [m.role for m in agent.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert "tool_start" in events and "tool_end" in events


@pytest.mark.asyncio
async def test_run_accumulates_usage_across_tool_rounds():
    from flowai.providers.base import ProviderUsage

    @tool
    def ping() -> str:
        return "pong"

    class TwoStep(ScriptedStreamProvider):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        async def generate(self, messages, tools=None, **kwargs):
            self.calls += 1
            usage = ProviderUsage(prompt_tokens=100, completion_tokens=10)
            if self.calls == 1:
                return ProviderResponse(tool_calls=[ToolCall(id="p", name="ping")], usage=usage)
            return ProviderResponse(content="done", usage=usage)

    agent = Agent(provider=TwoStep(), tools=[ping])
    assert await agent.run("go") == "done"
    assert agent.tracer.total_tokens() == 220
