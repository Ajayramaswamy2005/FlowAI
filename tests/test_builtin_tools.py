"""Tests for the file, shell and python tools, and the approval gate."""

import sys

import pytest

from flowai.tools import (
    ToolRegistry,
    bash,
    edit_file,
    glob_files,
    grep_files,
    list_dir,
    python_repl,
    read_file,
    write_file,
)
from flowai.tools.registry import DENIED_MESSAGE


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWAI_WORKSPACE", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_write_then_read_with_line_numbers(workspace):
    out = await write_file.execute(path="src/app.py", content="a = 1\nb = 2\n")
    assert "Created src" in out
    text = await read_file.execute(path="src/app.py")
    assert "1\ta = 1" in text and "2\tb = 2" in text


@pytest.mark.asyncio
async def test_read_window_reports_remaining_lines(workspace):
    (workspace / "big.txt").write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")
    text = await read_file.execute(path="big.txt", start_line=3, max_lines=2)
    assert "3\t3" in text and "4\t4" in text and "5\t5" not in text
    assert "start_line=5" in text


@pytest.mark.asyncio
async def test_paths_outside_workspace_are_refused(workspace):
    # The agent reaches tools through a registry, which turns the refusal into a message.
    registry = ToolRegistry([read_file, write_file])
    for path in ["../secret.txt", str(workspace.parent / "x.txt")]:
        out = await registry.execute("read_file", path=path)
        assert "outside the workspace" in out
    out = await registry.execute("write_file", path="../escape.txt", content="x")
    assert "outside the workspace" in out
    assert not (workspace.parent / "escape.txt").exists()


@pytest.mark.asyncio
async def test_edit_requires_a_unique_match(workspace):
    (workspace / "f.py").write_text("x = 1\nx = 1\ny = 2\n", encoding="utf-8")
    assert "appears 2 times" in await edit_file.execute(path="f.py", old_text="x = 1", new_text="x = 3")
    assert "not found" in await edit_file.execute(path="f.py", old_text="z = 9", new_text="")
    out = await edit_file.execute(path="f.py", old_text="y = 2", new_text="y = 20")
    assert "line 3" in out
    assert (workspace / "f.py").read_text(encoding="utf-8").endswith("y = 20\n")


@pytest.mark.asyncio
async def test_list_glob_and_grep(workspace):
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "mod.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    (workspace / "notes.md").write_text("hello world\n", encoding="utf-8")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "junk.py").write_text("def hello(): pass\n", encoding="utf-8")

    listing = await list_dir.execute()
    assert listing.splitlines()[0] == "pkg/" and "notes.md" in listing and "node_modules" not in listing

    assert (await glob_files.execute(pattern="**/*.py")).strip() == "pkg/mod.py"

    hits = await grep_files.execute(pattern=r"def hello", glob="*.py")
    assert "mod.py:1:" in hits and "junk.py" not in hits
    assert "invalid regular expression" in await grep_files.execute(pattern="(")


@pytest.mark.asyncio
async def test_python_repl_shares_one_namespace(workspace):
    # A comprehension referring to a module-level name broke with split globals/locals.
    out = await python_repl.execute(code="n = 3\nprint([i * n for i in range(3)])")
    assert out == "[0, 3, 6]"
    assert await python_repl.execute(code="value = 6 * 7") == "value = 42"
    assert "ZeroDivisionError" in await python_repl.execute(code="print('before')\n1/0")


@pytest.mark.asyncio
async def test_python_and_shell_cannot_see_credentials(workspace, monkeypatch):
    # Both run in child processes that start without credential variables.
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-should-never-reach-a-tool-0123456789")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    out = await python_repl.execute(code="import os\nprint(os.environ.get('NVIDIA_API_KEY'), os.environ.get('HARMLESS_SETTING'))")
    assert out == "None visible"
    cmd = f'"{sys.executable}" -c "import os; print(os.environ.get(\'NVIDIA_API_KEY\'))"'
    assert (await bash.execute(command=cmd)).strip() == "None"


@pytest.mark.asyncio
async def test_python_runs_outside_the_flowai_process(workspace):
    # In-process code could import flowai and read the keys it holds.
    out = await python_repl.execute(code="import sys\nprint('flowai.config' in sys.modules)")
    assert out == "False"


@pytest.mark.asyncio
async def test_bash_reports_exit_code(workspace):
    cmd = f'"{sys.executable}" -c "import sys; print(42); sys.exit(3)"'
    out = await bash.execute(command=cmd)
    assert "42" in out and "exit code 3" in out


@pytest.mark.asyncio
async def test_approval_gate_blocks_and_allows():
    asked = []

    def deny(name, args):
        asked.append(name)
        return False

    registry = ToolRegistry([python_repl, list_dir], approver=deny)
    assert await registry.execute("python_repl", code="print(1)") == DENIED_MESSAGE
    assert asked == ["python_repl"]

    # Tools that only read never ask.
    await registry.execute("list_dir", path=".")
    assert asked == ["python_repl"]

    async def allow(name, args):
        return True

    registry.set_approver(allow)
    assert await registry.execute("python_repl", code="print(1)") == "1"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_not_raised():
    out = await ToolRegistry([list_dir]).execute("no_such_tool")
    assert "does not exist" in out and "list_dir" in out
