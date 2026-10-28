"""Tool Registry for tool registration, lookup, schema export, and safe execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from flowai.security.policy import Risk
from flowai.security.secrets import sanitize_terminal
from flowai.security.shield import Shield
from flowai.tools.base import BaseTool
from flowai.tools.decorator import tool

if TYPE_CHECKING:
    from flowai.plugins.hooks import HookManager

# Called with (tool_name, arguments) or (tool_name, arguments, risk); return False to refuse.
Approver = Callable[..., Any]

DENIED_MESSAGE = "This call was not approved, so it did not run. Do not retry it; ask the user what they would prefer instead."


def _accepts_risk(approver: Approver) -> bool:
    try:
        params = list(inspect.signature(approver).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 3


class ToolRegistry:
    """Central store for agent tools.

    A registry with an `approver` asks before running any tool marked
    `requires_approval`. Library callers get no approver (they wrote the code
    themselves); the CLI attaches one, because there the model picks the
    commands and the user should see them first.

    A registry with a `shield` also grades every call (blocking the ones that
    would leak a credential, asking about the dangerous ones) and cleans every
    result before the model sees it. See flowai.security.shield.

    A registry with `hooks` runs plugins' pre_tool hooks before the shield
    grades a call (so rewritten arguments are graded too) and post_tool hooks
    after the shield cleaned the result (which is then cleaned again).
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
        approver: Approver | None = None,
        shield: Shield | None = None,
        hooks: HookManager | None = None,
    ):
        self._tools: dict[str, BaseTool] = {}
        self._builtin: set[str] = set()
        self.approver = approver
        self.shield = shield
        self.hooks = hooks
        if tools:
            for t in tools:
                self.register(t)

    def register(self, t: BaseTool | Callable[..., Any], builtin: bool | None = None) -> BaseTool:
        """Register a tool instance or raw function."""
        if not isinstance(t, BaseTool):
            t = tool(t)
        self._tools[t.name] = t
        if builtin is None:
            builtin = not getattr(t, "external", False)
        if builtin:
            self._builtin.add(t.name)
        else:
            self._builtin.discard(t.name)
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

    async def _approved(self, tool_instance: BaseTool, kwargs: dict[str, Any], risk: Risk | None) -> bool:
        gated = tool_instance.requires_approval or (risk is not None and risk.needs_confirmation)
        if not gated or self.approver is None:
            return True
        if _accepts_risk(self.approver):
            verdict = self.approver(tool_instance.name, kwargs, risk)
        else:
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
        ctx = None
        if self.hooks:
            from flowai.plugins.hooks import ToolCallContext

            ctx = await self.hooks.run("pre_tool", ToolCallContext(tool=tool_name, arguments=dict(kwargs)))
            if ctx.blocked:
                return f"Stopped by a FlowAI plugin: {ctx.reason}. This call was not run; do not retry it."
            if not isinstance(ctx.arguments, dict):
                return f"Stopped by a FlowAI plugin: a pre_tool hook replaced the arguments of {tool_name} with a non-dict."
            kwargs = ctx.arguments

        builtin = tool_name in self._builtin
        risk = self.shield.assess(tool_name, kwargs, builtin=builtin) if self.shield else None
        if risk is not None and risk.blocked:
            return self.shield.blocked_message(risk)
        if not await self._approved(tool_instance, kwargs, risk):
            return DENIED_MESSAGE
        try:
            result = await tool_instance.execute(**kwargs)
        except Exception as exc:
            result = f"Error executing {tool_name}: {type(exc).__name__}: {exc}"
        if self.shield is not None:
            text = result if isinstance(result, str) else str(result)
            result = self.shield.filter_output(tool_name, text, builtin=builtin)

        if ctx is not None and self.hooks.hooks("post_tool"):
            ctx.result = result if isinstance(result, str) else str(result)
            await self.hooks.run("post_tool", ctx)
            if ctx.result is not None:
                result = str(ctx.result)
                if self.shield is not None:
                    # A hook could have added anything; it gets the same cleaning as tool output.
                    result = sanitize_terminal(self.shield.redactor.redact(result))
        return result
