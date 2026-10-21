"""File system tools: the toolkit an agent needs to read and change a project.

Every path is resolved against the workspace root (the directory FlowAI was
started in, unless FLOWAI_WORKSPACE says otherwise) and must stay inside it.
That keeps a confused model from reading ~/.ssh or writing outside the project.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from flowai.tools.decorator import tool

# Files larger than this are refused rather than silently truncating context.
MAX_READ_BYTES = 400_000

# Directories never worth walking into during glob/grep.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".flowai",
}


def workspace_root() -> Path:
    """The directory the agent is allowed to touch."""
    return Path(os.getenv("FLOWAI_WORKSPACE", ".")).resolve()


def resolve_in_workspace(path: str) -> Path:
    """Resolve a user/model supplied path, refusing anything outside the workspace."""
    root = workspace_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path '{path}' is outside the workspace ({root}).")
    return resolved


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(workspace_root()))
    except ValueError:
        return str(path)


def _iter_files(root: Path):
    """Walk the workspace, skipping directories that are never interesting."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            yield Path(dirpath) / filename


@tool("read_file", description="Read a text file from the workspace. Returns numbered lines.")
def read_file(path: str, start_line: int = 1, max_lines: int = 500) -> str:
    """Read a file and return its contents with line numbers."""
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_dir():
        return f"Error: '{path}' is a directory. Use list_dir instead."
    if target.stat().st_size > MAX_READ_BYTES:
        return f"Error: '{path}' is larger than {MAX_READ_BYTES} bytes. Use grep_files to search it instead."

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file."

    lines = text.splitlines()
    start = max(1, start_line)
    window = lines[start - 1: start - 1 + max_lines]
    if not window:
        return f"(no lines at or after line {start}; file has {len(lines)} lines)"

    numbered = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(window))
    shown_to = start + len(window) - 1
    if shown_to < len(lines):
        numbered += f"\n... ({len(lines) - shown_to} more lines; read again with start_line={shown_to + 1})"
    return numbered


@tool("write_file", description="Create or overwrite a text file in the workspace.")
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    target = resolve_in_workspace(path)
    if target.is_dir():
        return f"Error: '{path}' is a directory."
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {_relative(target)} ({len(content.splitlines())} lines)."


@tool("edit_file", description="Replace an exact string in a workspace file. The string must appear exactly once.")
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact occurrence of old_text with new_text."""
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist. Use write_file to create it."

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a UTF-8 text file."

    occurrences = text.count(old_text)
    if occurrences == 0:
        return f"Error: old_text was not found in '{path}'. Read the file again and copy the text exactly."
    if occurrences > 1:
        return f"Error: old_text appears {occurrences} times in '{path}'. Include more surrounding lines to make it unique."

    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    line_no = text[: text.index(old_text)].count("\n") + 1
    return f"Edited {_relative(target)} at line {line_no}."


@tool("list_dir", description="List the entries of a directory in the workspace.")
def list_dir(path: str = ".") -> str:
    """List files and subdirectories, directories first."""
    target = resolve_in_workspace(path)
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not entries:
        return f"{_relative(target)} is empty."

    lines = []
    for entry in entries:
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
    return "\n".join(lines) or f"{_relative(target)} has no visible entries."


@tool("glob_files", description="Find workspace files whose path matches a glob pattern, e.g. '**/*.py'.")
def glob_files(pattern: str, limit: int = 100) -> str:
    """Return paths matching a glob pattern, newest first."""
    root = workspace_root()
    matches = []
    for file_path in _iter_files(root):
        rel = file_path.relative_to(root).as_posix()
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file_path.name, pattern):
            matches.append(file_path)

    if not matches:
        return f"No files match '{pattern}'."

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shown = matches[:limit]
    out = "\n".join(p.relative_to(root).as_posix() for p in shown)
    if len(matches) > limit:
        out += f"\n... ({len(matches) - limit} more matches)"
    return out


@tool("grep_files", description="Search workspace file contents with a regular expression.")
def grep_files(pattern: str, path: str = ".", glob: str = "*", limit: int = 100) -> str:
    """Search files for a regex and return 'path:line: text' hits."""
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"

    root = resolve_in_workspace(path)
    if root.is_file():
        candidates = [root]
    else:
        candidates = [p for p in _iter_files(root) if fnmatch.fnmatch(p.name, glob)]

    hits: list[str] = []
    for file_path in candidates:
        try:
            if file_path.stat().st_size > MAX_READ_BYTES:
                continue
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{_relative(file_path)}:{line_no}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n... (stopped at {limit} matches)"

    return "\n".join(hits) if hits else f"No matches for '{pattern}'."
