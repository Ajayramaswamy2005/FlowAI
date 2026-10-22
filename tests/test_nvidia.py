"""NVIDIA: every key name works, self-hosted NIMs need none, and picky models still answer."""

import json

import httpx
import pytest

from flowai.config.settings import Settings
from flowai.errors import ConfigurationError, ProviderError
from flowai.providers import openai as openai_mod
from flowai.providers.base import Message
from flowai.providers.nvidia import NvidiaProvider
from flowai.providers.openai import ThinkSplitter, split_thinking

KEY = "nvapi-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
TOOLS = [{"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object", "properties": {}}}}]
KEY_VARS = ("NVIDIA_API_KEY", "NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY", "NVIDIA_BASE_URL", "NIM_BASE_URL",
            "OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWAI_HOME", str(tmp_path / ".flowai"))
    monkeypatch.setenv("FLOWAI_KEYRING", "0")
    monkeypatch.chdir(tmp_path)  # away from any flowai.yaml in the checkout
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    async def no_wait(seconds):
        return None

    monkeypatch.setattr(openai_mod, "_backoff", no_wait)


@pytest.fixture
def server(monkeypatch):
    """A scripted NVIDIA endpoint: each request gets the next canned response."""
    state = {"requests": [], "responses": []}

    def handler(request):
        body = json.loads(request.content) if request.content else None
        state["requests"].append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        status, payload, headers = state["responses"].pop(0)
        if isinstance(payload, str):
            return httpx.Response(status, text=payload, headers={"content-type": "text/event-stream", **headers})
        return httpx.Response(status, json=payload, headers=headers)

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    def respond(*responses):
        state["responses"].extend((r[0], r[1], r[2] if len(r) > 2 else {}) for r in responses)
        return state

    return respond


def sse(*events):
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)


def ok_completion(text="hello"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}


async def collect(provider, tools=None):
    return [c async for c in provider.stream([Message(role="system", content="be brief"),
                                              Message(role="user", content="hi")], tools=tools)]


# ------------------------------------------------------------------- keys


@pytest.mark.parametrize("env_name", ["NVIDIA_API_KEY", "NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY"])
def test_every_nvidia_key_name_is_accepted(monkeypatch, env_name):
    monkeypatch.setenv(env_name, KEY)
    assert NvidiaProvider().api_key == KEY
    cfg = Settings.load()
    assert cfg.provider == "nvidia" and cfg.has_credentials("nvidia")


def test_an_openai_key_is_never_sent_to_nvidia(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key-that-belongs-elsewhere-0000")
    provider = NvidiaProvider()
    assert not provider.api_key
    with pytest.raises(ConfigurationError, match="NVIDIA_API_KEY"):
        provider.require_api_key()


def test_self_hosted_nim_needs_no_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")
    provider = NvidiaProvider(model="meta/llama-3.1-8b-instruct")
    assert provider.self_hosted and provider.base_url == "http://localhost:8000/v1"
    provider.require_api_key()  # does not raise
    assert Settings.load().has_credentials("nvidia")


def test_key_problems_are_explained_before_any_request():
    assert NvidiaProvider(api_key=KEY).key_problem() is None
    assert "nvapi-" in NvidiaProvider(api_key="b2xkLW5nYy1rZXktZm9ybWF0").key_problem()
    assert "spaces" in NvidiaProvider(api_key=KEY + "\n").key_problem()


# ---------------------------------------------------------- compatibility


@pytest.mark.asyncio
async def test_model_without_tool_support_still_answers(server):
    state = server(
        (400, {"error": {"message": "This model does not support tool calling"}}),
        (200, sse({"choices": [{"delta": {"content": "fine"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]")),
    )
    provider = NvidiaProvider(api_key=KEY, model="google/gemma-2-9b-it")
    chunks = await collect(provider, tools=TOOLS)
    assert "tools" in state["requests"][0]["body"] and "tools" not in state["requests"][1]["body"]
    assert any(c.notice and "tool calling" in c.notice for c in chunks)
    assert "".join(c.text_delta for c in chunks) == "fine"
    # The next turn does not try tools again.
    server((200, sse({"choices": [{"delta": {"content": "again"}, "finish_reason": "stop"}]}, "[DONE]")))
    await collect(provider, tools=TOOLS)
    assert "tools" not in state["requests"][2]["body"]


@pytest.mark.asyncio
async def test_unknown_fields_and_system_role_are_dropped(server):
    state = server(
        (422, {"detail": [{"type": "extra_forbidden", "loc": ["body", "stream_options"], "msg": "Extra inputs"}]}),
        (400, {"error": {"message": "System role not supported"}}),
        (200, sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}, "[DONE]")),
    )
    await collect(NvidiaProvider(api_key=KEY))
    last = state["requests"][-1]["body"]
    assert "stream_options" not in last
    assert [m["role"] for m in last["messages"]] == ["user"]
    assert last["messages"][0]["content"].startswith("be brief")


@pytest.mark.asyncio
async def test_rate_limits_are_retried(server):
    state = server((429, {"error": "Too Many Requests"}, {"retry-after": "1"}), (200, ok_completion("done")))
    reply = await NvidiaProvider(api_key=KEY).generate([Message(role="user", content="hi")])
    assert reply.content == "done" and len(state["requests"]) == 2


@pytest.mark.asyncio
async def test_default_max_tokens_and_key_header(server):
    state = server((200, ok_completion()))
    await NvidiaProvider(api_key=KEY).generate([Message(role="user", content="hi")])
    request = state["requests"][0]
    assert request["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert request["headers"]["authorization"] == f"Bearer {KEY}"
    assert request["body"]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_errors_say_what_to_do(server):
    server((401, {"status": 401, "title": "Unauthorized", "detail": "Authentication failed"}))
    with pytest.raises(ProviderError, match="nvapi-"):
        await NvidiaProvider(api_key="old-ngc-key-0123456789abcdef").generate([Message(role="user", content="x")])
    server((404, {"detail": "Function not found for account"}))
    with pytest.raises(ProviderError, match="flowai models nvidia"):
        await NvidiaProvider(api_key=KEY, model="nope/nope").generate([Message(role="user", content="x")])


@pytest.mark.asyncio
async def test_key_echoed_in_an_error_is_redacted(server):
    server((401, {"error": {"message": f"Invalid key {KEY}"}}))
    with pytest.raises(ProviderError) as exc:
        await NvidiaProvider(api_key=KEY).generate([Message(role="user", content="x")])
    assert KEY not in str(exc.value)


@pytest.mark.asyncio
async def test_list_models(server):
    server((200, {"data": [{"id": "meta/llama-3.3-70b-instruct"}, {"id": "deepseek-ai/deepseek-r1"}]}))
    assert await NvidiaProvider(api_key=KEY).list_models() == ["deepseek-ai/deepseek-r1", "meta/llama-3.3-70b-instruct"]


# --------------------------------------------------------------- reasoning


def test_think_tags_split_across_chunks():
    splitter = ThinkSplitter()
    pieces = ["<thi", "nk>weigh", "ing it</th", "ink>The answer", " is 4."]
    answer, thought = "", ""
    for p in pieces:
        a, t = splitter.feed(p)
        answer, thought = answer + a, thought + t
    a, t = splitter.flush()
    assert (answer + a, thought + t) == ("The answer is 4.", "weighing it")
    assert split_thinking("<think>hmm</think>\n\nYes.") == ("Yes.", "hmm")


@pytest.mark.asyncio
async def test_reasoning_streams_separately_from_the_answer(server):
    server((200, sse(
        {"choices": [{"delta": {"reasoning_content": "Let me think."}}]},
        {"choices": [{"delta": {"content": "<think>more</think>Answer"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        "[DONE]",
    )))
    chunks = await collect(NvidiaProvider(api_key=KEY, model="deepseek-ai/deepseek-r1"))
    assert "".join(c.reasoning_delta for c in chunks) == "Let me think.more"
    assert "".join(c.text_delta for c in chunks) == "Answer"


@pytest.mark.asyncio
async def test_tool_calls_without_ids_get_one(server):
    server((200, sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "list_dir", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    )))
    chunks = await collect(NvidiaProvider(api_key=KEY), tools=TOOLS)
    call = next(c.tool_call_delta for c in chunks if c.tool_call_delta)
    assert call.name == "list_dir" and call.id.startswith("call_")


@pytest.mark.asyncio
async def test_retired_and_unavailable_models_get_suggestions(server):
    server((410, {"title": "Gone", "detail": "The model 'meta/llama-3.3-70b-instruct' has reached its end of life"}))
    provider = NvidiaProvider(api_key=KEY, model="meta/llama-3.3-70b-instruct")
    with pytest.raises(ProviderError, match="retired"):
        await provider.generate([Message(role="user", content="x")])
    server((200, {"data": [{"id": "meta/llama-3.2-90b-vision-instruct"}, {"id": "openai/gpt-oss-20b"},
                           {"id": "nvidia/nemotron-3.5-embed-8b"}]}))
    suggestions = await provider.suggest_models()
    assert "openai/gpt-oss-20b" in suggestions
    assert "nvidia/nemotron-3.5-embed-8b" not in suggestions


@pytest.mark.asyncio
async def test_a_timeout_is_named(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("", request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    with pytest.raises(ProviderError, match="no response"):
        await collect(NvidiaProvider(api_key=KEY))


@pytest.mark.asyncio
async def test_a_bare_model_name_is_resolved_to_its_catalog_id(server):
    state = server(
        (404, "404 page not found"),
        (200, {"data": [{"id": "z-ai/glm-5.3-flash"}, {"id": "z-ai/glm-5.3"}]}),
        (200, sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}, "[DONE]")),
    )
    provider = NvidiaProvider(api_key=KEY, model="glm-5.3-flash")
    chunks = await collect(provider)
    assert provider.model == "z-ai/glm-5.3-flash"
    assert state["requests"][-1]["body"]["model"] == "z-ai/glm-5.3-flash"
    assert any(c.notice and "z-ai/glm-5.3-flash" in c.notice for c in chunks)
