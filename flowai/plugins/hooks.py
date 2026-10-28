"""Pre and post hooks: the points where a plugin can watch or steer an agent.

Five hook points, each called with a mutable context:

    pre_prompt    before the user's prompt reaches the model   (PromptContext)
    post_response after the model's final answer               (ResponseContext)
    pre_tool      before a tool call runs                      (ToolCallContext)
    post_tool     after a tool call ran                        (ToolCallContext)
    on_error      when a turn fails                            (ErrorContext)

A hook changes what happens by editing its context (rewrite `ctx.prompt`,
`ctx.arguments`, `ctx.result`, `ctx.reply`) or by calling `ctx.block(reason)`.
Hooks may be plain functions or coroutines.

Hooks never get to weaken the shield. A pre_tool hook runs *before* the shield
grades the call, so arguments it rewrites are graded like any others; a
post_tool hook sees output that is already redacted, and whatever it returns is
redacted again before the model reads it.

A failing hook must not take the agent down with it, and must not open a hole
either, so failures are handled by where they happen: a `pre_*` hook that
raises blocks what it was guarding (fail closed), while a `post_*` or
`on_error` hook that raises is skipped and the original value is kept.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flowai.security.secrets import redact

HOOK_POINTS = ("pre_prompt", "post_response", "pre_tool", "post_tool", "on_error")

Hook = Callable[[Any], Any]


@dataclass
class HookContext:
    agent: str = ""
    blocked: bool = False
    reason: str = ""
    #: Free space for hooks to share state within one event.
    extra: dict[str, Any] = field(default_factory=dict)

    def block(self, reason: str) -> None:
        self.blocked = True
        self.reason = reason


@dataclass
class PromptContext(HookContext):
    prompt: str = ""


@dataclass
class ResponseContext(HookContext):
    prompt: str = ""
    reply: str = ""


@dataclass
class ToolCallContext(HookContext):
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Set for post_tool: the (already shielded) output the model will read.
    result: str | None = None


@dataclass
class ErrorContext(HookContext):
    error: BaseException | None = None


@dataclass
class HookFailure:
    point: str
    hook: str
    error: str


def hook_name(fn: Hook) -> str:
    owner = getattr(fn, "__self__", None)
    base = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__
    if owner is not None and not inspect.isclass(owner):
        plugin = getattr(owner, "name", None)
        if plugin:
            return f"{plugin}.{getattr(fn, '__name__', base)}"
    return base


class HookManager:
    """Holds the hooks for each point and runs them in registration order."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = {point: [] for point in HOOK_POINTS}
        #: Recent hook errors, so the CLI can tell the user which plugin misbehaved.
        self.failures: list[HookFailure] = []

    def register(self, point: str, fn: Hook) -> Hook:
        if point not in self._hooks:
            raise ValueError(f"unknown hook point '{point}'; choose one of: {', '.join(HOOK_POINTS)}")
        if not callable(fn):
            raise TypeError(f"hook for '{point}' must be callable, got {type(fn).__name__}")
        self._hooks[point].append(fn)
        return fn

    def on(self, point: str) -> Callable[[Hook], Hook]:
        """Decorator form: `@hooks.on("pre_tool")`."""
        return lambda fn: self.register(point, fn)

    def unregister(self, point: str, fn: Hook) -> None:
        if fn in self._hooks.get(point, []):
            self._hooks[point].remove(fn)

    def hooks(self, point: str) -> list[Hook]:
        return list(self._hooks.get(point, []))

    def count(self) -> int:
        return sum(len(v) for v in self._hooks.values())

    def __bool__(self) -> bool:
        return self.count() > 0

    async def run(self, point: str, ctx: HookContext) -> HookContext:
        fail_closed = point.startswith("pre_")
        for fn in self.hooks(point):
            try:
                outcome = fn(ctx)
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception as exc:
                name = hook_name(fn)
                self.failures.append(HookFailure(point, name, redact(f"{type(exc).__name__}: {exc}")))
                del self.failures[:-50]
                if fail_closed:
                    ctx.block(f"the {point} hook '{name}' failed ({type(exc).__name__}), so this was stopped")
            if ctx.blocked:
                break
        return ctx


__all__ = [
    "HOOK_POINTS",
    "ErrorContext",
    "HookContext",
    "HookFailure",
    "HookManager",
    "PromptContext",
    "ResponseContext",
    "ToolCallContext",
]
