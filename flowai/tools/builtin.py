"""Built-in tools and the default toolset the CLI and server hand to an agent."""

from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stdout

from flowai.tools.decorator import tool
from flowai.tools.files import (
    edit_file,
    glob_files,
    grep_files,
    list_dir,
    read_file,
    workspace_root,
    write_file,
)
from flowai.tools.web import fetch_url, search_web

BASH_TIMEOUT_SECONDS = 120


@tool("bash", description="Run a shell command in the workspace directory.", requires_approval=True)
async def bash(command: str) -> str:
    """Run a shell command and return its combined output."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace_root()),
        )
    except OSError as exc:
        return f"Failed to start command: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=BASH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Command timed out after {BASH_TIMEOUT_SECONDS}s and was killed."

    out = stdout.decode(errors="replace").strip()
    err = stderr.decode(errors="replace").strip()
    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"STDERR: {err}")
    if proc.returncode != 0:
        parts.append(f"(exit code {proc.returncode})")
    return "\n".join(parts) or "(no output)"


@tool("python_repl", description="Execute Python code in a fresh namespace and return its output.", requires_approval=True)
def python_repl(code: str) -> str:
    """Execute Python code and capture standard output."""
    buffer = io.StringIO()
    # One namespace for globals and locals: separate ones break comprehensions
    # and any function that refers to a name defined alongside it.
    namespace: dict = {"__name__": "__flowai_repl__"}
    try:
        with redirect_stdout(buffer):
            exec(code, namespace)
    except Exception as exc:
        captured = buffer.getvalue().strip()
        error = f"Python execution error: {type(exc).__name__}: {exc}"
        return f"{captured}\n{error}".strip()

    out = buffer.getvalue().strip()
    if not out:
        user_vars = {k: v for k, v in namespace.items() if not k.startswith("__")}
        if user_vars:
            name, value = list(user_vars.items())[-1]
            out = f"{name} = {value!r}"
    return out or "Executed successfully (no output)."


def default_toolset() -> list:
    """The tools an interactive FlowAI agent gets unless the caller says otherwise."""
    return [
        read_file,
        write_file,
        edit_file,
        list_dir,
        glob_files,
        grep_files,
        bash,
        python_repl,
        search_web,
        fetch_url,
    ]


def default_system_prompt(workspace: str | None = None) -> str:
    """System prompt describing the workspace the built-in tools operate on."""
    root = workspace or str(workspace_root())
    return (
        "You are FlowAI, a terminal coding assistant working inside a user's project.\n"
        f"The workspace root is: {root}\n"
        f"The platform is: {sys.platform} (shell commands must be valid for it).\n\n"
        "Guidelines:\n"
        "- Inspect before you change: use list_dir, glob_files, grep_files and read_file.\n"
        "- Prefer edit_file for targeted changes; use write_file only for new or fully rewritten files.\n"
        "- Paths are relative to the workspace root and must stay inside it.\n"
        "- Run commands with bash when you need to build, test or inspect the system.\n"
        "- Never claim you ran something you did not run, and report failures exactly as they occurred.\n"
        "- Keep answers short and concrete; the user is reading them in a terminal."
    )


__all__ = [
    "bash",
    "python_repl",
    "search_web",
    "fetch_url",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
    "default_toolset",
    "default_system_prompt",
    "BASH_TIMEOUT_SECONDS",
]
