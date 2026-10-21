"""State Management - Pydantic-based State containers and history."""

from __future__ import annotations

import copy
import uuid
from typing import Any

from pydantic import BaseModel, Field

from flowai.providers.base import Message


class AgentState(BaseModel):
    """Immutable state container for LangGraph-style workflows."""
    session_id: str = Field(default_factory=lambda: f"ses_{uuid.uuid4().hex[:8]}")
    step: int = 0
    messages: list[Message] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> AgentState:
        """Return a new copy of state with updated key-value."""
        new_values = copy.deepcopy(self.values)
        new_values[key] = value
        return self.model_copy(update={"values": new_values, "step": self.step + 1})

    def update_values(self, updates: dict[str, Any]) -> AgentState:
        """Merge a dictionary into state values."""
        new_values = copy.deepcopy(self.values)
        new_values.update(updates)
        return self.model_copy(update={"values": new_values, "step": self.step + 1})

    def append_message(self, message: Message | dict[str, Any]) -> AgentState:
        new_msgs = list(self.messages)
        if isinstance(message, dict):
            new_msgs.append(Message(**message))
        else:
            new_msgs.append(message)
        return self.model_copy(update={"messages": new_msgs})
