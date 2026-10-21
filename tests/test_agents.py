"""Tests for Agent execution and provider mocking."""

import pytest

from flowai.agents import Agent, CriticAgent, PlannerAgent, SupervisorAgent, ToolAgent
from flowai.providers.base import BaseProvider, ProviderResponse, StreamChunk, ToolCall
from flowai.tools import tool


class MockProvider(BaseProvider):
    """Mock provider for deterministic agent tests."""

    def __init__(self, canned_replies: list[ProviderResponse] | None = None):
        super().__init__(model="mock-model")
        self.replies = canned_replies or [ProviderResponse(content="Mocked response")]
        self.call_count = 0

    async def generate(self, messages, tools=None, **kwargs):
        resp = self.replies[min(self.call_count, len(self.replies) - 1)]
        self.call_count += 1
        return resp

    async def stream(self, messages, tools=None, **kwargs):
        yield StreamChunk(text_delta="Mocked ")
        yield StreamChunk(text_delta="stream")

    async def embeddings(self, texts, model=None, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.asyncio
async def test_agent_basic_run():
    provider = MockProvider([ProviderResponse(content="The capital of France is Paris.")])
    agent = Agent(name="GeoAgent", provider=provider)
    reply = await agent.run("What is the capital of France?")
    assert "Paris" in reply


@pytest.mark.asyncio
async def test_agent_tool_loop():
    @tool
    def get_stock(symbol: str) -> float:
        return 150.25

    tool_call = ToolCall(id="call_1", name="get_stock", arguments={"symbol": "AAPL"})
    first_resp = ProviderResponse(content="", tool_calls=[tool_call])
    second_resp = ProviderResponse(content="AAPL is currently trading at $150.25.")

    provider = MockProvider([first_resp, second_resp])
    agent = Agent(name="StockBot", provider=provider, tools=[get_stock])

    result = await agent.run("What is AAPL price?")
    assert "150.25" in result
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_specialized_agents_instantiation():
    provider = MockProvider()
    tool_agent = ToolAgent(provider=provider)
    planner = PlannerAgent(provider=provider)
    critic = CriticAgent(provider=provider)
    supervisor = SupervisorAgent(workers={"tool": tool_agent, "planner": planner}, provider=provider)

    assert tool_agent.name == "ToolExpert"
    assert planner.name == "Planner"
    assert critic.name == "Critic"
    assert supervisor.name == "Supervisor"


@pytest.mark.asyncio
async def test_agent_model_switching_preserves_context():
    # Turn 1 with Model A
    provider_a = MockProvider([ProviderResponse(content="I am Model A. You told me your name is Alice.")])
    agent = Agent(name="MultiModelAgent", provider=provider_a, model="model-a")

    reply_1 = await agent.run("My name is Alice.")
    assert "Alice" in reply_1
    assert len(agent.messages) >= 3  # system, user, assistant

    # Switch to Model B mid-conversation
    provider_b = MockProvider([ProviderResponse(content="I am Model B. Nice to meet you, Alice!")])
    agent.switch_model(model="model-b", provider=provider_b)

    assert agent.model == "model-b"
    # History must be preserved
    assert any("Alice" in (m.content or "") for m in agent.messages)

    reply_2 = await agent.run("What is my name?")
    assert "Alice" in reply_2
