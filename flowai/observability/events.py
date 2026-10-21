"""Event Bus and Streaming Event definitions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """Event emitted during execution for observers, webhooks, or SSE."""
    event: str  # e.g., "token", "tool_start", "tool_end", "agent_start", "error"
    data: dict[str, Any] = Field(default_factory=dict)


EventHandler = Callable[[StreamEvent], Coroutine[Any, Any, None] | None]


class EventBus:
    """Asynchronous publish-subscribe event bus for real-time telemetry."""

    def __init__(self):
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register an observer handler."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def publish(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """Broadcast an event to all subscribers."""
        ev = StreamEvent(event=event_name, data=data or {})
        for handler in self._handlers:
            try:
                res = handler(ev)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass  # Observers should never crash the core loop
