"""Terminal presentation helpers: colour, framing, and the approval prompt.

Output stays plain ASCII so it renders on every Windows console, and colour
switches itself off when the output is piped or NO_COLOR is set.
"""

from __future__ import annotations

import os
import sys

from flowai.security.secrets import sanitize_terminal

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

MAX_ARG_PREVIEW = 400


def stdin_is_terminal() -> bool:
    """True only when a person can type answers.

    On Windows the NUL device claims to be a terminal (isatty() is True for
    `< NUL`), so the console is asked directly whether this is a real one.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except Exception:
        return True


def colour_enabled() -> bool:
    """Colour only when a human is watching a terminal that wants it."""
    if os.getenv("NO_COLOR") or os.getenv("FLOWAI_NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text: str, *codes: str) -> str:
    if not codes or not colour_enabled():
        return text
    return "".join(codes) + text + RESET


def rule(width: int = 60) -> str:
    return "-" * width


def banner(lines: list[str]) -> str:
    return "\n".join([rule(), *lines, rule()])


def info(message: str) -> None:
    print(paint(message, DIM))


def warn(message: str) -> None:
    print(paint(message, YELLOW))


def error(message: str) -> None:
    print(paint(f"Error: {message}", RED), file=sys.stderr)


def format_arguments(arguments: dict) -> str:
    """One-line preview of tool arguments, shortened to stay readable."""
    parts = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else repr(value)
        text = text.replace("\n", "\\n")
        if len(text) > 120:
            text = text[:120] + "..."
        parts.append(f"{key}={text}")
    line = ", ".join(parts)
    return line[:MAX_ARG_PREVIEW] + ("..." if len(line) > MAX_ARG_PREVIEW else "")


def tool_start(name: str, arguments: dict) -> None:
    print(paint(f"\n  [tool] {name}({sanitize_terminal(format_arguments(arguments))})", CYAN))


def tool_end(name: str, result: str, seconds: float | None = None) -> None:
    lines = (result or "").strip().splitlines()
    summary = sanitize_terminal(lines[0][:160]) if lines else "(no output)"
    extra = f" (+{len(lines) - 1} more lines)" if len(lines) > 1 else ""
    print(paint(f"  [done] {summary}{extra}", DIM))


def ask_approval(name: str, arguments: dict, risk=None) -> str:
    """Show what a tool is about to do and return 'yes', 'no' or 'always'."""
    level = risk.level if risk is not None else "medium"
    critical = level == "critical"
    print(paint(f"\n  {name} wants to run (risk: {level}):", RED if critical else YELLOW))
    for key, value in arguments.items():
        text = sanitize_terminal(value if isinstance(value, str) else repr(value))
        lines = text.splitlines() or [""]
        print(f"    {key}: {lines[0]}")
        for line in lines[1:20]:
            print(f"      {line}")
        if len(lines) > 20:
            print(f"      ... ({len(lines) - 20} more lines)")
    for reason in (risk.reasons if risk is not None else []):
        print(paint(f"    ! {reason}", RED if critical else YELLOW))
    choices = "[y]es / [n]o" if critical else "[y]es / [n]o / [a]lways"
    try:
        answer = input(paint(f"  Allow? {choices}: ", BOLD)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "no"
    if answer in ("a", "always") and not critical:
        return "always"
    return "yes" if answer in ("y", "yes") else "no"


class PlainView:
    """The interface for pipes, CI logs and --plain: the same calls, plain text out.

    The chat loop talks to this or to the full-screen Interface (flowai.cli.tui)
    without knowing which one it has.
    """

    def __init__(self) -> None:
        self.status: dict = {}
        self.commands: dict[str, str] = {}
        self.last_reasoning = ""
        self.show_thinking = False
        self.context_tokens = 0

    def print(self, text: str = "") -> None:
        print(text)

    def info(self, message: str) -> None:
        info(message)

    def notice(self, message: str) -> None:
        warn(message)

    def error(self, message: str) -> None:
        error(message)

    def success(self, message: str) -> None:
        print(paint(message, GREEN))

    def splash(self, version: str, rows: list, hint: str) -> None:
        print(banner([f"  FlowAI {version}"] + [f"  {label}: {value}" for label, value in rows] + [f"  {hint}"]))

    def read(self) -> str | None:
        try:
            return input(paint("\n> ", BOLD))
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""

    def tool_start(self, name: str, arguments: dict) -> None:
        tool_start(name, arguments)

    def tool_end(self, name: str, result: str, seconds_taken: float | None = None) -> None:
        tool_end(name, result, seconds_taken)

    async def gate(self, name: str, arguments: dict, risk=None) -> str:
        return ask_approval(name, arguments, risk)

    async def run_turn(self, agent, prompt: str) -> None:
        print(paint(f"\n{agent.name}: ", BOLD, GREEN), end="", flush=True)
        thinking: list[str] = []
        async for chunk in agent.stream(prompt):
            if chunk.notice:
                warn(f"\n({chunk.notice})")
            if chunk.reasoning_delta:
                thinking.append(chunk.reasoning_delta)
                if self.show_thinking:
                    print(paint(sanitize_terminal(chunk.reasoning_delta), DIM), end="", flush=True)
            if chunk.text_delta:
                print(sanitize_terminal(chunk.text_delta), end="", flush=True)
        print()
        if thinking:
            self.last_reasoning = "".join(thinking).strip()

    def abort_turn(self) -> None:
        warn("\n(interrupted)")

    def show_table(self, rows: list[tuple[str, str]]) -> None:
        width = max((len(k) for k, _ in rows), default=0) + 2
        for key, value in rows:
            print(f"  {key:<{width}}{value}")

    def show_report(self, title: str, rows: list[tuple[str, str, str]]) -> None:
        print(title)
        marks = {"ok": "ok  ", "warn": "WARN", "bad": "FAIL", "info": "    "}
        for state, label, detail in rows:
            print(f"  [{marks.get(state, '    ')}] {label}: {detail}")

    def show_reasoning(self) -> None:
        print(self.last_reasoning or "(no reasoning from the last answer)")

    def show_models(self, models: list[str], current: str) -> None:
        for model in models:
            print(f"{'*' if model == current else ' '} {model}")
