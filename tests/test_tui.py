"""The interface: what a turn looks like, and that hostile text cannot hijack the terminal."""

import io

import pytest
from rich.console import Console

from flowai.cli.tui import Interface, Railed, context_window, summarise_arguments, summarise_result
from flowai.observability.tracer import Tracer
from flowai.providers.base import ProviderUsage, StreamChunk
from flowai.security.policy import Risk
from flowai.security.secrets import register_secret
from flowai.tools.registry import DENIED_MESSAGE


def make_ui(width=100):
    console = Console(file=io.StringIO(), width=width, height=40, color_system=None, force_terminal=False)
    return Interface(console=console, animate=False)


def output(ui):
    return ui.console.file.getvalue()


class FakeAgent:
    """Streams a reasoning model's turn: thinking, one tool call, then an answer."""

    name = "FlowAI"
    model = "deepseek-ai/deepseek-r1"

    def __init__(self, ui, answer="The **tests** pass."):
        self.ui = ui
        self.answer = answer
        self.tracer = Tracer()
        self.messages = []

    async def stream(self, prompt):
        yield StreamChunk(notice="example notice")
        yield StreamChunk(reasoning_delta="I should look at the tests first.")
        self.ui.tool_start("bash", {"command": "pytest -q"})
        self.ui.tool_end("bash", "3 passed", 0.42)
        yield StreamChunk(text_delta=self.answer)
        yield StreamChunk(usage=ProviderUsage(prompt_tokens=1200, completion_tokens=300, total_tokens=1500))


@pytest.mark.asyncio
async def test_a_turn_draws_the_current():
    ui = make_ui()
    await ui.run_turn(FakeAgent(ui), "run the tests")
    out = output(ui)
    assert "example notice" in out
    assert "undertow" in out and "thought for" in out
    assert "bash" in out and "pytest -q" in out and "1 line" in out
    assert "flowai" in out and "tests" in out
    assert "tide" in out
    assert ui.last_reasoning == "I should look at the tests first."


@pytest.mark.asyncio
async def test_model_text_cannot_inject_escape_sequences_or_leak_keys():
    key = "nvapi-ZZZZabcdefghijklmnopqrstuvwxyz012345"
    register_secret(key, "NVIDIA_API_KEY")
    ui = make_ui()
    hostile = f"Done.\x1b]52;c;ZXZpbA==\x07\x1b]0;owned\x07 key: {key}"
    await ui.run_turn(FakeAgent(ui, answer=hostile), "x")
    out = output(ui)
    assert "\x1b]" not in out and key not in out and "REDACTED" in out


@pytest.mark.asyncio
async def test_gate_defaults(monkeypatch):
    ui = make_ui()
    answers = iter(["", "", "a", "a"])

    async def fake_ask(prompt):
        return next(answers)

    monkeypatch.setattr(ui, "_ask", fake_ask)
    medium = Risk(level="medium", reasons=["installs packages"])
    critical = Risk(level="critical", reasons=["touches a .env secrets file"])
    # Enter allows an ordinary call, but never a call that touches secrets.
    assert await ui.gate("bash", {"command": "pip install x"}, medium) == "yes"
    assert await ui.gate("bash", {"command": "cat .env"}, critical) == "no"
    assert await ui.gate("bash", {"command": "pip install x"}, medium) == "always"
    # "always" is not offered for secrets.
    assert await ui.gate("bash", {"command": "cat .env"}, critical) == "no"
    out = output(ui)
    assert "GATE" in out and "risk CRITICAL" in out and "touches a .env secrets file" in out


def test_edit_gate_shows_a_diff():
    ui = make_ui()
    body = ui._gate_body("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"})
    ui.print(body)
    assert "- x = 1" in output(ui) and "+ x = 2" in output(ui)


def test_railed_tail_keeps_the_last_lines():
    ui = make_ui(width=40)
    ui.print(Railed("\n".join(f"line {i}" for i in range(10)), "|", "white", tail=3))
    lines = [line for line in output(ui).splitlines() if line.strip()]
    assert lines == ["| line 7", "| line 8", "| line 9"]


@pytest.mark.parametrize("name, args, expected", [
    ("bash", {"command": "ls\npwd"}, "ls  (+1 lines)"),
    ("write_file", {"path": "a.txt", "content": "1\n2\n"}, "a.txt  (2 lines)"),
    ("search_web", {"query": "nim"}, '"nim"'),
])
def test_argument_summaries(name, args, expected):
    assert summarise_arguments(name, args) == expected


@pytest.mark.parametrize("result, ok, outcome", [
    ("a\nb", True, "2 lines"),
    ("out\n(exit code 2)", False, "exit 2"),
    (DENIED_MESSAGE, False, "declined"),
    ("Blocked by FlowAI shield: x", False, "blocked by shield"),
    ("Error: 'x' does not exist.", False, "Error: 'x' does not exist."),
])
def test_result_summaries(result, ok, outcome):
    assert summarise_result("bash", result) == (ok, outcome)


def test_context_windows():
    assert context_window("claude-sonnet-5-5") == 200_000
    assert context_window("meta/llama-3.3-70b-instruct") == 128_000
    assert context_window("something-new") == 128_000


def test_the_real_prompt_reads_lines_and_answers_the_gate(tmp_path, monkeypatch):
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setenv("FLOWAI_HOME", str(tmp_path))
    ui = make_ui()
    ui.commands = {"/help": "show help"}
    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        keys.send_text("fix the build\r")
        assert ui.read() == "fix the build"
        assert ui._session is not None  # the prompt_toolkit session, not the input() fallback
        keys.send_text("a\r")
        verdict = asyncio.run(ui.gate("bash", {"command": "make"}, Risk(level="high", reasons=["x"])))
        assert verdict == "always"
        # Pasted keys never reach the history file.
        keys.send_text("my key is nvapi-HISTORYabcdefghijklmnopqrstuvwxyz0123\r")
        ui.read()
    assert "nvapi-HISTORY" not in (tmp_path / "history").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_the_animated_view_survives_a_gate_mid_turn(monkeypatch):
    console = Console(file=io.StringIO(), width=90, height=30, force_terminal=True, color_system="truecolor")
    ui = Interface(console=console, animate=True)

    async def yes(prompt):
        return "y"

    monkeypatch.setattr(ui, "_ask", yes)

    class GatedAgent(FakeAgent):
        async def stream(self, prompt):
            yield StreamChunk(text_delta="Checking. ")
            self.ui.tool_start("bash", {"command": "make"})
            assert await self.ui.gate("bash", {"command": "make"}, Risk(level="high")) == "yes"
            assert self.ui.live is not None  # restarted after the question
            self.ui.tool_end("bash", "ok", 0.1)
            yield StreamChunk(text_delta="Built.")

    await ui.run_turn(GatedAgent(ui), "build it")
    assert ui.live is None
    out = console.file.getvalue()
    assert "GATE" in out and "Built." in out
