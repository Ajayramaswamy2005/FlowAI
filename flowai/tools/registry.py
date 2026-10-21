"""Tool Registry for tool registration, lookup, schema export, and safe execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from flowai.tools.base import BaseTool
from flowai.tools.decorator import tool

# Called with (tool_name, arguments); return False to refuse the call.
Approver = Callable[[str, dict], Any]

DENIED_MESSAGE = "The user declined this call. Do not retry it; ask what they would prefer instead."


class ToolRegistry:
    """Central store for agent tools.

    A registry with an `approver` asks before running any tool marked
    `requires_approval`. Library callers get no approver (they wrote the code
    themselves); the CLI attaches one, because there the model picks the
    commands and the user should see them first.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
        approver: Approver | None = None,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.approver = approver
        if tools:
            for t in tools:
                self.register(t)

    def register(self, t: BaseTool | Callable[..., Any]) -> BaseTool:
        """Register a tool instance or raw function."""
        if not isinstance(t, BaseTool):
            t = tool(t)
        self._tools[t.name] = t
        return t

    def set_approver(self, approver: Approver | None) -> None:
        """Attach (or clear) the callback consulted before approval-gated tools run."""
        self.approver = approver

    def get(self, name: str) -> BaseTool | None:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def get_schemas(self, format: str = "openai") -> list[dict[str, Any]]:
        """Export tool definitions in openai or anthropic schema format."""
        if format == "anthropic":
            return [t.to_anthropic_schema() for t in self._tools.values()]
        return [t.to_openai_schema() for t in self._tools.values()]

    async def _approved(self, tool_instance: BaseTool, kwargs: dict[str, Any]) -> bool:
        if not tool_instance.requires_approval or self.approver is None:
            return True
        verdict = self.approver(tool_instance.name, kwargs)
        if asyncio.iscoroutine(verdict):
            verdict = await verdict
        return bool(verdict)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute tool safely, surfacing errors rather than crashing the harness."""
        tool_instance = self.get(tool_name)
        if not tool_instance:
            return (
                f"Error: tool '{tool_name}' does not exist. "
                f"Available tools: {', '.join(self._tools) or '(none)'}"
            )
        if not await self._approved(tool_instance, kwargs):
            return DENIED_MESSAGE
        try:
            return await tool_instance.execute(**kwargs)
        except Exception as exc:
            return f"Error executing {tool_name}: {type(exc).__name__}: {exc}"
