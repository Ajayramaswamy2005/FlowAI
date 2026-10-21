"""In-memory, Short-term, and Conversation memory implementations."""

from __future__ import annotations

from typing import Any

from flowai.memory.base import BaseMemory, MemoryRecord


class InMemory(BaseMemory):
    """Fast in-memory store for session or ephemeral testing."""

    def __init__(self):
        self._records: list[MemoryRecord] = []

    async def add(self, record: MemoryRecord | str, **metadata: Any) -> str:
        if isinstance(record, str):
            role = metadata.pop("role", "assistant")
            rec = MemoryRecord(content=record, role=role, metadata=metadata)
        else:
            rec = record
        self._records.append(rec)
        return rec.id

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryRecord]:
        """Simple keyword substring search over in-memory records."""
        q = query.lower()
        matched = []
        for r in reversed(self._records):
            if q in r.content.lower():
                matched.append(r)
                if len(matched) >= limit:
                    break
        return matched

    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        return self._records[-limit:]

    async def clear(self) -> None:
        self._records.clear()


class ShortTermMemory(InMemory):
    """Sliding-window short term memory buffer."""

    def __init__(self, capacity: int = 20):
        super().__init__()
        self.capacity = capacity

    async def add(self, record: MemoryRecord | str, **metadata: Any) -> str:
        rec_id = await super().add(record, **metadata)
        if len(self._records) > self.capacity:
            self._records = self._records[-self.capacity:]
        return rec_id


class ConversationMemory(InMemory):
    """Conversation history buffer preserving turn order and message roles."""

    def format_history(self) -> str:
        """Format history into plain dialogue string."""
        lines = []
        for r in self._records:
            lines.append(f"{r.role.capitalize()}: {r.content}")
        return "\n".join(lines)
