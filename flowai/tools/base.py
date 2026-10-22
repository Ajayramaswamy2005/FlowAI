"""Base data classes for Tool definitions and schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolParameterSchema(BaseModel):
    """JSON schema representing tool function parameters."""
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """OpenAI/Anthropic compatible tool metadata representation."""
    name: str
    description: str
    parameters: ToolParameterSchema


class BaseTool(ABC):
    """Abstract class for all tools in FlowAI."""
    name: str
    description: str
    schema_definition: ToolDefinition
    is_async: bool = False
    # Tools that change the machine (shell, arbitrary code) ask before running.
    requires_approval: bool = False
    # Tools from outside FlowAI (MCP servers); their output is treated as untrusted.
    external: bool = False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with keyword arguments and return the result."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to standard OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema_definition.parameters.model_dump(),
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema_definition.parameters.model_dump(),
        }
