"""CLI: argument handling, configuration, and an offline end-to-end run."""

import asyncio
import importlib
import json

import pytest

from flowai.config.settings import Settings
from flowai.providers.base import BaseProvider, ProviderResponse, StreamChunk
from flowai.providers.registry import ProviderRegistry

# flowai.cli re-exports main(), which shadows the module; fetch the module itself.
cli = importlib.import_module("flowai.cli.main")

KEY_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "FLOWAI_PROVIDER", "FLOWAI_MODEL",
            "NVIDIA_BASE_URL", "NIM_BASE_URL"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate global config and cwd so tests never read the developer's real setup."""
    cfg_dir = tmp_path / "home" / ".flowai"
    monkeypatch.setenv("FLOWAI_HOME", str(cfg_dir))
    # Never touch the developer's real credential store.
    monkeypatch.setenv("FLOWAI_KEYRING", "0")
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
    # Without a usable OS keyring the key lands in the owner-only config file.
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
    assert asyncio.run(approver("bash", {"command": "rm -rf /"})) is False
    assert asyncio.run(cli.Approver("auto")("bash", {"command": "ls"})) is True


def test_configure_nvidia_with_any_key_and_logout(home, capsys):
    assert cli.main(["configure", "--provider", "nvidia", "--key", "nvapi-abcdefghijklmnopqrstuvwxyz0123"]) == 0
    cfg = Settings.load()
    assert (cfg.provider, cfg.model) == ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b")
    assert cfg.has_credentials()
    assert cli.main(["logout", "nvidia"]) == 0
    assert "Removed" in capsys.readouterr().out
    assert not Settings.load().has_credentials("nvidia")


def test_configure_self_hosted_nim_needs_no_key(home):
    assert cli.main(["configure", "--provider", "nvidia", "--base-url", "http://localhost:8000/v1"]) == 0
    cfg = Settings.load()
    assert cfg.provider_options("nvidia")["base_url"] == "http://localhost:8000/v1"
    assert cfg.has_credentials("nvidia")


def test_shield_report_runs_and_flags_plain_text_project_keys(home, tmp_path, capsys):
    cli.main(["configure", "--provider", "openai", "--key", "sk-abcdefghijklmnopqrstuvwxyz"])
    assert cli.main(["shield", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "redaction" in out and "tripwires" in out and "openai key" in out

    (tmp_path / "project" / "flowai.yaml").write_text(
        'providers:\n  openai:\n    api_key: "sk-committed-to-git-abcdefghijkl"\n', encoding="utf-8"
    )
    assert cli.main(["shield", "--plain"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_models_command_lists_and_filters(home, monkeypatch, capsys):
    from flowai.providers.nvidia import NvidiaProvider

    async def fake_list(self):
        return ["deepseek-ai/deepseek-r1", "meta/llama-3.3-70b-instruct", "nvidia/nemotron-4-340b-instruct"]

    monkeypatch.setattr(NvidiaProvider, "list_models", fake_list)
    assert cli.main(["models", "nvidia", "llama", "--plain"]) == 0
    assert capsys.readouterr().out.split() == ["meta/llama-3.3-70b-instruct"]


def test_yolo_still_asks_about_secrets(monkeypatch):
    from flowai.security.policy import Risk

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    asked = []
    approver = cli.Approver("auto", ask=lambda name, args, risk: asked.append(name) or "no")
    assert asyncio.run(approver("bash", {"command": "ls"}, Risk())) is True
    assert asyncio.run(approver("bash", {"command": "env"}, Risk(level="critical"))) is False
    assert asked == ["bash"]
