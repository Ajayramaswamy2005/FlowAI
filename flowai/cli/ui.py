"""Terminal presentation helpers: colour, framing, and the approval prompt.

Output stays plain ASCII so it renders on every Windows console, and colour
switches itself off when the output is piped or NO_COLOR is set.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

MAX_ARG_PREVIEW = 400


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
    print(paint(f"\n  [tool] {name}({format_arguments(arguments)})", CYAN))


def tool_end(name: str, result: str) -> None:
    lines = (result or "").strip().splitlines()
    summary = lines[0][:160] if lines else "(no output)"
    extra = f" (+{len(lines) - 1} more lines)" if len(lines) > 1 else ""
    print(paint(f"  [done] {summary}{extra}", DIM))


def ask_approval(name: str, arguments: dict) -> str:
    """Show what a tool is about to do and return 'yes', 'no' or 'always'."""
    print(paint(f"\n  {name} wants to run:", YELLOW))
    for key, value in arguments.items():
        text = value if isinstance(value, str) else repr(value)
        lines = text.splitlines() or [""]
        print(f"    {key}: {lines[0]}")
        for line in lines[1:20]:
            print(f"      {line}")
        if len(lines) > 20:
            print(f"      ... ({len(lines) - 20} more lines)")
    try:
        answer = input(paint("  Allow? [y]es / [n]o / [a]lways: ", BOLD)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "no"
    if answer in ("a", "always"):
        return "always"
    return "yes" if answer in ("y", "yes") else "no"
