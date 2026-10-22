"""The interactive interface: a conversation drawn as a current.

Every turn flows down one rail. Your prompt enters at a diamond; the model's
reasoning runs underneath as the undertow; each tool call is a node on the
rail with its timing and outcome; the answer is rendered markdown; the turn
closes with what it cost and how full the context window (the tide) is.
While work is in flight, a swell of block characters moves along the rail.

Nothing here talks to a model or runs a tool. The CLI hands this class the
agent's stream and events, and it decides how they look.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from flowai.cli import theme
from flowai.cli.theme import (
    CORAL,
    DEPTH,
    EMBER,
    FOAM,
    KELP,
    MOSS,
    RISK_COLOURS,
    SHALLOW,
    SUN,
    TIDE,
    YOU,
)
from flowai.security.keystore import flowai_home
from flowai.security.secrets import redact, sanitize_terminal
from flowai.tools.registry import DENIED_MESSAGE

#: Context window sizes by model-name fragment; the first match wins.
CONTEXT_WINDOWS = [
    ("claude", 200_000), ("gemini", 1_000_000), ("gpt-4o", 128_000), ("gpt-4.1", 1_000_000),
    ("o1", 200_000), ("o3", 200_000), ("o4", 200_000), ("llama-4", 1_000_000), ("llama-3", 128_000),
    ("deepseek", 128_000), ("nemotron", 128_000), ("qwen", 32_768), ("mistral", 32_768), ("gemma", 8_192),
]
DEFAULT_CONTEXT = 128_000


def context_window(model: str) -> int:
    name = model.lower()
    return next((size for fragment, size in CONTEXT_WINDOWS if fragment in name), DEFAULT_CONTEXT)


def clean(text: str) -> str:
    """What may reach the screen: no secrets, no escape sequences."""
    return sanitize_terminal(redact(text or ""))


class Railed:
    """Draws any renderable with a coloured rail down its left edge.

    With `tail`, only the last lines are drawn: the live view uses that so a
    long answer scrolls inside the window instead of tearing the screen.
    """

    def __init__(self, renderable: RenderableType, glyph: str, style: str, tail: int | None = None):
        self.renderable = renderable
        self.glyph = glyph
        self.style = style
        self.tail = tail

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(8, options.max_width - 2)
        lines = console.render_lines(self.renderable, options.update(width=width, height=None), pad=False)
        if self.tail and len(lines) > self.tail:
            lines = lines[-self.tail:]
        for line in lines:
            yield Segment(self.glyph + " ", console.get_style(self.style))
            yield from line
            yield Segment.line()


@dataclass
class Turn:
    started: float = field(default_factory=time.monotonic)
    reasoning: list[str] = field(default_factory=list)
    reasoning_started: float | None = None
    text: list[str] = field(default_factory=list)
    header_shown: bool = False
    tool: tuple[str, dict, float] | None = None
    tools_run: int = 0


def summarise_arguments(name: str, args: dict[str, Any]) -> str:
    """The one line that says what a tool call is about."""
    def first_line(value: Any) -> str:
        text = str(value)
        lines = text.splitlines() or [""]
        extra = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        return lines[0] + extra

    if name == "bash":
        return first_line(args.get("command", ""))
    if name == "python_repl":
        return first_line(args.get("code", ""))
    if name in ("read_file", "list_dir", "edit_file"):
        return str(args.get("path", "."))
    if name == "write_file":
        n = len(str(args.get("content", "")).splitlines())
        return f"{args.get('path', '')}  ({n} lines)"
    if name == "grep_files":
        return f"/{args.get('pattern', '')}/  in {args.get('path', '.')}"
    if name == "glob_files":
        return str(args.get("pattern", ""))
    if name == "fetch_url":
        return str(args.get("url", ""))
    if name == "search_web":
        return f"\"{args.get('query', '')}\""
    return ", ".join(f"{k}={first_line(v)}" for k, v in args.items())


def summarise_result(name: str, result: str) -> tuple[bool, str]:
    """(succeeded, short outcome) for the tool line."""
    text = result or ""
    if text.startswith("Blocked by FlowAI shield"):
        return False, "blocked by shield"
    if text == DENIED_MESSAGE:
        return False, "declined"
    body = text
    if body.startswith("<untrusted_content"):
        body = body.split("\n", 1)[-1].rsplit("</untrusted_content>", 1)[0]
    if "(exit code" in body:
        code = body.rsplit("(exit code", 1)[1].split(")")[0].strip()
        return False, f"exit {code}"
    if body.startswith(("Error", "Fetch failed", "Web search failed", "Python execution error", "Failed")):
        return False, body.splitlines()[0][:60]
    lines = [line for line in body.strip().splitlines() if line.strip()]
    if name in ("write_file", "edit_file") and lines:
        return True, lines[0][:60]
    if not lines:
        return True, "no output"
    return True, f"{len(lines)} line{'s' if len(lines) != 1 else ''}"


class _LiveView:
    """Rebuilt on every refresh of the live area, so it always shows the latest state."""

    def __init__(self, interface: Interface):
        self.interface = interface

    def __rich__(self) -> RenderableType:
        return self.interface._live_view()


class Interface:
    """Owns the terminal for an interactive session."""

    def __init__(self, console: Console | None = None, animate: bool | None = None):
        self.console = console or Console(highlight=False, emoji=False)
        self.g = theme.glyphs_for(self.console.encoding)
        if animate is None:
            animate = self.console.is_terminal and not os.getenv("FLOWAI_NO_ANIM")
        self.animate = animate
        self.live: Live | None = None
        self.turn: Turn | None = None
        self.agent_label = "flowai"
        self.last_reasoning = ""
        self.show_thinking = False
        self.context_tokens = 0
        #: Filled in by the CLI, read by the bottom strip: provider, model, shield, approvals.
        self.status: dict[str, Any] = {}
        self._session = None
        self._session_ready = False
        self.commands: dict[str, str] = {}

    # ------------------------------------------------------------- output

    def print(self, renderable: RenderableType = "") -> None:
        (self.live.console if self.live else self.console).print(renderable)

    def info(self, message: str) -> None:
        self.print(Text.assemble((f"{self.g.dot} ", DEPTH), (message, SHALLOW)))

    def notice(self, message: str) -> None:
        self.print(Text.assemble((f"{self.g.notice} ", f"bold {SUN}"), (clean(message), SUN)))

    def error(self, message: str) -> None:
        self.print(Text.assemble((f"{self.g.fail} ", f"bold {EMBER}"), (clean(message), EMBER)))

    def success(self, message: str) -> None:
        self.print(Text.assemble((f"{self.g.ok} ", f"bold {MOSS}"), (message, MOSS)))

    # ------------------------------------------------------------- splash

    def _wordmark(self, phase: float, version: str, tagline: str) -> Group:
        lines = theme.WORDMARK if self.g.unicode else theme.WORDMARK_ASCII
        rows = []
        for i, line in enumerate(lines):
            row = Text("  ")
            row.append_text(theme.current_text(line, phase + i * 0.04))
            side = f"   v{version}" if i == 0 else f"   {tagline}"
            row.append(side, style=DEPTH if i == 0 else SHALLOW)
            rows.append(row)
        if len(lines) == 1:
            rows.append(Text(f"  {tagline}", style=SHALLOW))
        swell = Text("  ")
        swell.append_text(theme.wave(min(46, self.console.width - 4), self.g, t=phase * 3, speed=2.0))
        rows.append(swell)
        return Group(*rows)

    def splash(self, version: str, rows: list[tuple[str, Text | str]], hint: str) -> None:
        tagline = "an agent that moves with your code"
        self.console.print()
        if self.animate:
            with Live(self._wordmark(0, version, tagline), console=self.console, transient=True,
                      refresh_per_second=40) as live:
                for step in range(1, 18):
                    time.sleep(0.03)
                    live.update(self._wordmark(step / 18, version, tagline))
        self.console.print(self._wordmark(1.0, version, tagline))
        self.console.print()
        for label, value in rows:
            line = Text(f"  {label:<10}", style=DEPTH)
            line.append_text(value if isinstance(value, Text) else Text(str(value), style=FOAM))
            self.console.print(line)
        self.console.print()
        self.console.print(Text(f"  {hint}", style=DEPTH))

    # -------------------------------------------------------------- input

    def _build_session(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.styles import Style
        except ImportError:
            return None

        commands = self.commands

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if not text.startswith("/") or " " in text:
                    return
                for name, description in commands.items():
                    if name.startswith(text):
                        yield Completion(name, start_position=-len(text), display_meta=description)

        class RedactingHistory(FileHistory):
            """Prompt history never keeps a key someone pasted."""

            def store_string(self, string: str) -> None:
                super().store_string(redact(string))

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        style = Style.from_dict({
            "you": f"bold {YOU}",
            "rail": DEPTH,
            "bottom-toolbar": "noreverse bg:#0b1220 #94a3b8",
            "tb.key": f"bold {TIDE}",
            "tb.dim": DEPTH,
            "tb.text": "#cbd5e1",
            "tb.ok": MOSS,
            "tb.warn": SUN,
            "tb.bad": EMBER,
            "completion-menu.completion": "bg:#0f172a #e6fffb",
            "completion-menu.completion.current": f"bg:{TIDE} #0b1220",
            "completion-menu.meta.completion": "bg:#0f172a #64748b",
            "completion-menu.meta.completion.current": f"bg:{TIDE} #0b1220",
        })
        try:
            history_path = flowai_home() / "history"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            return PromptSession(
                history=RedactingHistory(str(history_path)),
                completer=SlashCompleter(),
                complete_while_typing=True,
                key_bindings=bindings,
                style=style,
                mouse_support=False,
            )
        except Exception:
            # No real console (some IDE terminals on Windows): plain input still works.
            return None

    def _toolbar(self):
        s = self.status
        g = self.g
        window = context_window(str(s.get("model", "")))
        fraction = self.context_tokens / window if window else 0.0
        block, colour = theme.gauge(fraction, g)
        shield_style = "class:tb.ok" if s.get("approvals") == "ask" else "class:tb.warn"
        parts = [
            ("class:tb.key", f" {s.get('provider', '')} "),
            ("class:tb.dim", f"{g.arrow} "),
            ("class:tb.text", f"{s.get('model', '')}   "),
            ("class:tb.dim", "tide "),
            (f"fg:{colour}", block),
            ("class:tb.text", f" {round(fraction * 100)}%   "),
            ("class:tb.dim", "spent "),
            ("class:tb.text", f"{theme.human_tokens(int(s.get('tokens', 0)))} tok   "),
            ("class:tb.dim", "shield "),
            (shield_style, f"{g.shield} {'armed' if s.get('approvals') == 'ask' else 'armed, yolo'}   "),
            ("class:tb.dim", f"esc{g.arrow}enter newline {g.dot} / commands "),
        ]
        return parts

    def read(self) -> str | None:
        """One prompt from the user; None when they leave (Ctrl-D)."""
        if not self._session_ready:
            self._session = self._build_session()
            self._session_ready = True
        self.console.print()
        if self._session is None:
            try:
                return input(f" {self.g.you} ")
            except EOFError:
                return None
            except KeyboardInterrupt:
                return ""
        try:
            return self._session.prompt(
                [("class:you", f" {self.g.you} ")],
                bottom_toolbar=self._toolbar,
                prompt_continuation=lambda width, line, soft: [("class:rail", f" {self.g.rail} ")],
            )
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""

    async def _ask(self, prompt: str) -> str:
        if self._session is not None:
            from prompt_toolkit import PromptSession

            try:
                return await PromptSession().prompt_async(prompt)
            except (EOFError, KeyboardInterrupt):
                return "n"
        try:
            return await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return "n"

    # --------------------------------------------------------------- turn

    def _live_view(self) -> RenderableType:
        t = self.turn
        g = self.g
        if t is None:
            return Text("")
        now = time.monotonic()
        if t.tool:
            name, args, started = t.tool
            line = Text.assemble((f"{g.tool} ", CORAL), (name, f"bold {CORAL}"), "  ")
            line.append(clean(summarise_arguments(name, args)), style=SHALLOW)
            line.truncate(max(20, self.console.width - 20), overflow="ellipsis")
            line.append("  ")
            line.append_text(theme.wave(6, g))
            line.append(f" {theme.seconds(now - started)}", style=DEPTH)
            return line
        if t.text:
            parts: list[RenderableType] = []
            if not t.header_shown:
                parts.append(self._agent_header())
            body = Markdown(clean("".join(t.text)), code_theme="monokai", hyperlinks=False)
            parts.append(Railed(body, g.rail, TIDE, tail=max(4, self.console.height - 6)))
            swell = Text(f"{g.rail} ", style=TIDE)
            swell.append_text(theme.wave(10, g))
            parts.append(swell)
            return Group(*parts)
        if t.reasoning:
            width = max(10, self.console.width - 30)
            thought = clean("".join(t.reasoning)).replace("\n", " ")[-width:]
            line = Text.assemble((f"{g.think} undertow ", f"bold {KELP}"))
            line.append_text(theme.wave(6, g))
            line.append(f" {thought}", style=f"italic {KELP}")
            return line
        line = theme.wave(12, g)
        line.append(f"  gathering the current  {theme.seconds(now - t.started)}", style=DEPTH)
        return line

    def _agent_header(self) -> Text:
        return Text.assemble((f"{self.g.agent} ", f"bold {TIDE}"), (self.agent_label, f"bold {TIDE}"))

    def _start_live(self) -> None:
        if not self.animate or self.live is not None:
            return
        self.live = Live(_LiveView(self), console=self.console, refresh_per_second=12, transient=True,
                         redirect_stdout=False, redirect_stderr=False)
        self.live.start()

    def _stop_live(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def abort_turn(self) -> None:
        """Called when Ctrl-C cut a turn short: put the terminal back in order."""
        self._stop_live()
        if self.turn:
            self._close_reasoning()
            self._flush_text()
        self.turn = None
        self.notice("interrupted")

    def _close_reasoning(self) -> None:
        t = self.turn
        if not t or not t.reasoning:
            return
        thought = clean("".join(t.reasoning)).strip()
        took = time.monotonic() - (t.reasoning_started or t.started)
        t.reasoning, t.reasoning_started = [], None
        if not thought:
            return
        self.last_reasoning = thought
        words = len(thought.split())
        line = Text.assemble(
            (f"{self.g.think} undertow ", f"bold {KELP}"),
            (f"thought for {theme.seconds(took)} {self.g.dot} {words} words", KELP),
        )
        if not self.show_thinking:
            line.append("   /think to read it", style=DEPTH)
        self.print(line)
        if self.show_thinking:
            self.print(Railed(Text(thought, style=f"italic {KELP}"), self.g.rail, KELP))

    def _flush_text(self) -> None:
        t = self.turn
        if not t or not t.text:
            return
        body = clean("".join(t.text)).strip()
        t.text = []
        if not body:
            return
        if not t.header_shown:
            self.print(self._agent_header())
            t.header_shown = True
        self.print(Railed(Markdown(body, code_theme="monokai", hyperlinks=False), self.g.rail, TIDE))

    def tool_start(self, name: str, arguments: dict) -> None:
        if self.turn is None:
            self.turn = Turn()
        self._close_reasoning()
        self._flush_text()
        self.turn.tool = (name, arguments, time.monotonic())

    def tool_end(self, name: str, result: str, seconds_taken: float | None = None) -> None:
        t = self.turn
        if t is None:
            return
        _, arguments, started = t.tool or (name, {}, time.monotonic())
        t.tool = None
        t.tools_run += 1
        took = seconds_taken if seconds_taken is not None else time.monotonic() - started
        ok, outcome = summarise_result(name, result)
        colour = MOSS if ok else SUN if outcome == "declined" else EMBER

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(justify="right", no_wrap=True)
        left = Text.assemble((f"{self.g.tool} ", CORAL), (name, f"bold {CORAL}"), "  ")
        left.append(clean(summarise_arguments(name, arguments)), style=SHALLOW)
        right = Text(f"{theme.seconds(took)}  ", style=DEPTH)
        right.append(f"{self.g.ok if ok else self.g.fail} {outcome}", style=colour)
        if "[REDACTED:" in (result or ""):
            right.append(f"  {self.g.shield} secrets masked", style=SUN)
        grid.add_row(left, right)
        self.print(grid)

    async def run_turn(self, agent, prompt: str) -> None:
        """Stream one prompt through the agent, drawing the current as it goes."""
        self.agent_label = agent.name.lower()
        self.turn = Turn()
        tokens_before = agent.tracer.total_tokens()
        cost_before = agent.tracer.total_cost()
        self._start_live()
        try:
            async for chunk in agent.stream(prompt):
                t = self.turn
                if chunk.notice:
                    self.notice(chunk.notice)
                if chunk.reasoning_delta:
                    if t.reasoning_started is None:
                        t.reasoning_started = time.monotonic()
                    t.reasoning.append(chunk.reasoning_delta)
                if chunk.text_delta:
                    self._close_reasoning()
                    t.text.append(chunk.text_delta)
                if chunk.usage and chunk.usage.prompt_tokens:
                    self.context_tokens = chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        finally:
            self._stop_live()
        self._close_reasoning()
        self._flush_text()

        t = self.turn
        if not t.header_shown and not t.tools_run:
            self.print(Text.assemble((f"{self.g.agent} ", TIDE), ("(no answer)", DEPTH)))
        if not self.context_tokens:
            # The provider reported no usage: estimate from the transcript, about 4 characters a token.
            self.context_tokens = sum(len(m.content or "") for m in agent.messages) // 4
        tokens = agent.tracer.total_tokens() - tokens_before
        cost = agent.tracer.total_cost() - cost_before
        self.print(self._footer(agent.model, time.monotonic() - t.started, t.tools_run, tokens, cost))
        self.turn = None

    def _footer(self, model: str, took: float, tools: int, tokens: int, cost: float) -> Text:
        from flowai.observability.tracer import MODEL_PRICING

        parts = [theme.seconds(took)]
        if tools:
            parts.append(f"{tools} tool{'s' if tools != 1 else ''}")
        if tokens:
            parts.append(f"{theme.human_tokens(tokens)} tok")
        if cost and model in MODEL_PRICING:
            parts.append(f"~${cost:.4f}")
        fraction = self.context_tokens / context_window(model)
        block, colour = theme.gauge(fraction, self.g)
        line = Text(f"{self.g.end} ", style=DEPTH)
        line.append(f" {self.g.dot} ".join(parts), style=DEPTH)
        line.append("   tide ", style=DEPTH)
        line.append(block, style=colour)
        line.append(f" {round(fraction * 100)}%", style=DEPTH)
        return line

    # --------------------------------------------------------------- gate

    def _gate_body(self, name: str, args: dict[str, Any]) -> RenderableType:
        if name == "bash":
            lexer = "powershell" if sys.platform == "win32" and "powershell" in str(args.get("command", "")).lower() else "bash"
            return Syntax(clean(str(args.get("command", ""))), lexer, theme="monokai",
                          background_color="default", word_wrap=True)
        if name == "python_repl":
            return Syntax(clean(str(args.get("code", ""))), "python", theme="monokai",
                          background_color="default", word_wrap=True, line_numbers=False)
        if name == "edit_file":
            diff = Text(f"{args.get('path', '')}\n", style=f"bold {FOAM}")
            for line in clean(str(args.get("old_text", ""))).splitlines()[:12]:
                diff.append(f"- {line}\n", style=EMBER)
            for line in clean(str(args.get("new_text", ""))).splitlines()[:12]:
                diff.append(f"+ {line}\n", style=MOSS)
            diff.rstrip()
            return diff
        if name == "write_file":
            content = clean(str(args.get("content", "")))
            lines = content.splitlines()
            preview = Text(f"{args.get('path', '')}  ({len(lines)} lines)\n", style=f"bold {FOAM}")
            for line in lines[:10]:
                preview.append(f"  {line}\n", style=SHALLOW)
            if len(lines) > 10:
                preview.append(f"  ... {len(lines) - 10} more lines\n", style=DEPTH)
            preview.rstrip()
            return preview
        body = Text()
        for key, value in args.items():
            body.append(f"{key}: ", style=DEPTH)
            body.append(f"{clean(str(value))[:400]}\n", style=FOAM)
        body.rstrip()
        return body

    async def gate(self, name: str, arguments: dict[str, Any], risk=None) -> str:
        """Ask whether a tool call may run. Returns 'yes', 'no' or 'always'."""
        was_live = self.live is not None
        self._stop_live()
        level = risk.level if risk is not None else "medium"
        critical = level == "critical"
        colour = RISK_COLOURS.get(level, SUN)

        head = Table.grid(expand=True)
        head.add_column(ratio=1)
        head.add_column(justify="right")
        head.add_row(
            Text.assemble(("GATE  ", f"bold {colour}"), (name, f"bold {FOAM}"), ("  wants to run", SHALLOW)),
            Text(f"risk {level.upper()}", style=f"bold {colour}"),
        )
        parts: list[RenderableType] = [head, self._gate_body(name, arguments)]
        for reason in (risk.reasons if risk is not None else []):
            parts.append(Text(f"! {reason}", style=colour))
        if critical:
            choices = "[y] allow once   [n] deny (default)      this call touches secrets, so it asks every time"
        else:
            choices = f"[y] allow once (default)   [a] always allow {name}   [n] deny"
        parts.append(Text(choices, style=DEPTH))

        self.print()
        self.print(Railed(Group(*parts), self.g.gate, colour))
        answer = (await self._ask(f" {self.g.gate} {self.g.arrow} ")).strip().lower()
        if answer in ("a", "always") and not critical:
            verdict = "always"
        elif answer in ("y", "yes") or (answer == "" and not critical):
            verdict = "yes"
        else:
            verdict = "no"
        label = {"yes": "gate opened once", "always": f"gate open for {name} this session", "no": "gate kept shut"}
        self.print(Text(f"{self.g.gate} {label[verdict]}", style=MOSS if verdict != "no" else SUN))
        if was_live and self.turn is not None:
            self._start_live()
        return verdict

    # ---------------------------------------------------------- listings

    def show_table(self, rows: list[tuple[str, str]], key_style: str = TIDE) -> None:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(no_wrap=True)
        grid.add_column()
        for key, value in rows:
            grid.add_row(Text(key, style=f"bold {key_style}"), Text(value, style=SHALLOW))
        self.print(Railed(grid, self.g.rail, DEPTH))

    def show_report(self, title: str, rows: list[tuple[str, str, str]]) -> None:
        """rows: (state, label, detail), state in ok / warn / bad / info."""
        marks = {
            "ok": (self.g.ok, MOSS), "warn": (self.g.notice, SUN), "bad": (self.g.fail, EMBER),
            "info": (self.g.dot, DEPTH),
        }
        self.print(Text.assemble((f"{self.g.shield} ", f"bold {TIDE}"), (title, f"bold {FOAM}")))
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column()
        for state, label, detail in rows:
            mark, colour = marks.get(state, marks["info"])
            grid.add_row(Text(mark, style=colour), Text(label, style=f"bold {FOAM}"), Text(detail, style=SHALLOW))
        self.print(Railed(grid, self.g.rail, DEPTH))

    def show_reasoning(self) -> None:
        if not self.last_reasoning:
            self.info("no reasoning yet: the last answer came from a model that does not show its thinking.")
            return
        self.print(Text(f"{self.g.think} undertow, last turn", style=f"bold {KELP}"))
        self.print(Railed(Text(self.last_reasoning, style=f"italic {KELP}"), self.g.rail, KELP))

    def show_models(self, models: list[str], current: str) -> None:
        if not models:
            self.info("the provider listed no models.")
            return
        grid = Table.grid(padding=(0, 3))
        columns = max(1, min(3, (self.console.width - 4) // 44))
        for _ in range(columns):
            grid.add_column(no_wrap=True, overflow="ellipsis", max_width=44)
        cells = [
            Text(f"{self.g.agent if m == current else ' '} {m}", style=f"bold {TIDE}" if m == current else SHALLOW)
            for m in models
        ]
        for i in range(0, len(cells), columns):
            row = cells[i:i + columns]
            grid.add_row(*row, *[Text("")] * (columns - len(row)))
        self.print(Railed(grid, self.g.rail, DEPTH))
        self.info(f"{len(models)} models. Switch with /model <id>.")
