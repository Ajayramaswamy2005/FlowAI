"""Memory System Base Classes and Interfaces."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """Atomic unit of agent memory storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: str = "assistant"  # user, assistant, system, context
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = None
    score: float | None = None


class BaseMemory(ABC):
    """Abstract interface for all memory backends in FlowAI."""

    @abstractmethod
    async def add(self, record: MemoryRecord | str, **metadata: Any) -> str:
        """Store a new memory item and return its identifier."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryRecord]:
        """Search memory for relevant records."""
        ...

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        """Retrieve recent interaction history."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored memories in this instance/session."""
        ...
