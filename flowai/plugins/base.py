"""The Plugin class custom extensions subclass.

    from flowai.plugins import Plugin
    from flowai.tools import tool

    class AuditLog(Plugin):
        name = "audit-log"
        description = "Writes every tool call to audit.log"

        def pre_tool(self, ctx):            # any method named after a hook point is a hook
            if ctx.tool == "bash" and "docker" in ctx.arguments.get("command", ""):
                ctx.block("docker is off limits in this project")

        def post_tool(self, ctx):
            with open("audit.log", "a") as f:
                f.write(f"{ctx.tool} {ctx.arguments}\\n")

        def tools(self):
            @tool
            def word_count(text: str) -> int:
                "Count the words in a piece of text."
                return len(text.split())
            return [word_count]

Install it as a package that declares an entry point in the `flowai.plugins`
group, or drop the file into ~/.flowai/plugins/. Plugins are never loaded from
the project you are working in: opening a cloned repository must not run its
code.
"""

from __future__ import annotations

from typing import Any

from flowai.plugins.hooks import HOOK_POINTS, HookManager


class Plugin:
    name: str = ""
    version: str = "0.0.0"
    description: str = ""

    def tools(self) -> list[Any]:
        """Tools this plugin adds: BaseTool instances or plain functions."""
        return []

    def setup(self, hooks: HookManager) -> None:
        """Register hooks by hand. By default every method named after a hook point is registered."""
        for point in HOOK_POINTS:
            method = getattr(self, point, None)
            if callable(method):
                hooks.register(point, method)

    @property
    def display_name(self) -> str:
        return self.name or type(self).__name__
