"""Base Agent class and execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, Field

from flowai.memory.base import BaseMemory
from flowai.observability.events import EventBus
from flowai.observability.tracer import Tracer
from flowai.providers.base import BaseProvider, StreamChunk
from flowai.providers.registry import get_provider
from flowai.tools.base import BaseTool
from flowai.tools.registry import ToolRegistry


class AgentContext(BaseModel):
    """Context passed across agent turns."""
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract Base Class for all agents in FlowAI."""

    def __init__(
        self,
        name: str = "Agent",
        model: str = "gpt-4o",
        provider: str | BaseProvider = "openai",
        system_prompt: str = "You are a helpful autonomous AI assistant.",
        tools: Sequence[BaseTool | Any] | ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        tracer: Tracer | None = None,
        event_bus: EventBus | None = None,
    ):
        self.name = name
        self.model = model
        self.provider = get_provider(provider, model=model) if isinstance(provider, str) else provider
        self.system_prompt = system_prompt
        self.tools = ToolRegistry(tools) if not isinstance(tools, ToolRegistry) else tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.tracer = tracer or Tracer()
        self.event_bus = event_bus or EventBus()

    @abstractmethod
    async def run(self, prompt: str, **kwargs: Any) -> str:
        """Execute agent reasoning on input prompt and return output."""
        ...

    @abstractmethod
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream response tokens and event deltas."""
        ...
