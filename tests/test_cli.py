"""CLI: argument handling, configuration, and an offline end-to-end run."""

import importlib
import json

import pytest

from flowai.config import settings as settings_mod
from flowai.config.settings import Settings
from flowai.providers.base import BaseProvider, ProviderResponse, StreamChunk
from flowai.providers.registry import ProviderRegistry

# flowai.cli re-exports main(), which shadows the module; fetch the module itself.
cli = importlib.import_module("flowai.cli.main")

KEY_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY",
            "OPENROUTER_API_KEY", "FLOWAI_PROVIDER", "FLOWAI_MODEL"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate global config and cwd so tests never read the developer's real setup."""
    cfg_dir = tmp_path / "home" / ".flowai"
    monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG_PATH", cfg_dir / "config.json")
    monkeypatch.setattr(cli, "GLOBAL_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(cli, "GLOBAL_CONFIG_PATH", cfg_dir / "config.json")
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return cfg_dir


@pytest.mark.parametrize("argv, expected", [
    ([], ["chat"]),
    (["fix the bug"], ["run", "fix the bug"]),
    (["chat", "--resume"], ["chat", "--resume"]),
    (["--yolo"], ["chat", "--yolo"]),
    (["--version"], ["--version"]),
    (["tools"], ["tools"]),
])
def test_argv_defaults(argv, expected):
    assert cli.normalise_argv(argv) == expected


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "flowai 0." in capsys.readouterr().out


def test_configure_non_interactive_saves_config(home):
    assert cli.main(["configure", "--provider", "anthropic", "--key", "sk-test"]) == 0
    saved = json.loads((home / "config.json").read_text(encoding="utf-8"))
    assert saved == {"provider": "anthropic", "model": "claude-sonnet-5-5", "ANTHROPIC_API_KEY": "sk-test"}

    cfg = Settings.load()
    assert cfg.provider == "anthropic" and cfg.has_credentials()


def test_configure_rejects_unknown_provider(home):
    assert cli.main(["configure", "--provider", "nope", "--key", "x"]) == 1


def test_project_file_overrides_global_and_is_found_from_subdirs(home, tmp_path):
    cli.main(["configure", "--provider", "openai", "--key", "sk"])
    (tmp_path / "project" / "flowai.yaml").write_text('provider: "ollama"\nmodel: "qwen3"\n', encoding="utf-8")
    sub = tmp_path / "project" / "src" / "deep"
    sub.mkdir(parents=True)
    import os
    os.chdir(sub)
    cfg = Settings.load()
    assert (cfg.provider, cfg.model, cfg.project_name) == ("ollama", "qwen3", "project")


def test_key_in_env_picks_provider_and_default_model(home, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = Settings.load()
    assert (cfg.provider, cfg.model) == ("gemini", "gemini-2.0-flash")


def test_run_without_key_explains_the_fix(home, capsys):
    assert cli.main(["run", "--provider", "openai", "hello"]) == 1
    err = capsys.readouterr().err
    assert "no API key" in err


def test_init_scaffolds_and_refuses_to_overwrite(home, tmp_path):
    assert cli.main(["init", "demo"]) == 0
    assert (tmp_path / "project" / "demo" / "flowai.yaml").exists()
    assert (tmp_path / "project" / "demo" / "main.py").exists()
    assert cli.main(["init", "demo"]) == 1


class EchoProvider(BaseProvider):
    requires_api_key = False

    def __init__(self, model="echo", **kwargs):
        super().__init__(model=model)

    async def generate(self, messages, tools=None, **kwargs):
        return ProviderResponse(content=f"echo: {messages[-1].content}")

    async def stream(self, messages, tools=None, **kwargs):
        yield StreamChunk(text_delta=f"echo: {messages[-1].content}")

    async def embeddings(self, texts, model=None, **kwargs):
        return []


def test_run_end_to_end_offline(home, monkeypatch, capsys):
    ProviderRegistry.register("echo", EchoProvider)
    # "echo" is not a key-bearing provider; treat it like a local one.
    monkeypatch.setattr(Settings, "has_credentials", lambda self, provider=None: True)
    assert cli.main(["run", "--provider", "echo", "-q", "ping", "pong"]) == 0
    assert capsys.readouterr().out.strip() == "echo: ping pong"


def test_approver_refuses_without_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    approver = cli.Approver("ask")
    assert approver("bash", {"command": "rm -rf /"}) is False
    assert cli.Approver("auto")("bash", {"command": "ls"}) is True
