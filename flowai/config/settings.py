"""Configuration: global user config, per-project file, and environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from flowai.errors import ConfigurationError
from flowai.security import keystore
from flowai.security.secrets import register_environment_secrets, register_secret

#: Project config filenames, in the order they are looked for.
PROJECT_CONFIG_NAMES = ("flowai.yaml", "flowai.yml", "flowai.toml", "flowai.json")

#: Environment variable `flowai configure` saves each provider's key under.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

#: Every name a provider's key is accepted under, first match wins. NVIDIA
#: hands out keys from build.nvidia.com and from NGC, and people export them
#: under all of these.
PROVIDER_KEY_ALIASES = {
    "nvidia": ("NVIDIA_API_KEY", "NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

#: Model used when a provider is chosen but no model is named.
PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
    "nvidia": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gemini": "gemini-2.0-flash",
    "openrouter": "deepseek/deepseek-r1",
    "ollama": "llama3.2",
}

#: Order in which a provider is picked from whatever keys are present.
PROVIDER_PREFERENCE = ("anthropic", "openai", "nvidia", "gemini", "openrouter")


def provider_key_envs(provider: str) -> tuple[str, ...]:
    """All environment names a provider's key may be found under."""
    if provider in PROVIDER_KEY_ALIASES:
        return PROVIDER_KEY_ALIASES[provider]
    primary = PROVIDER_KEY_ENV.get(provider)
    return (primary,) if primary else ()


def nvidia_is_self_hosted(base_url: str | None) -> bool:
    """A NIM running on your own machine or cluster needs no NVIDIA key."""
    url = base_url or os.getenv("NVIDIA_BASE_URL") or os.getenv("NIM_BASE_URL") or ""
    return bool(url) and "api.nvidia.com" not in url


class ProviderConfig(BaseModel):
    name: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class MemoryConfig(BaseModel):
    backend: str = "in_memory"  # in_memory, qdrant, chroma
    url: str = "http://localhost:6333"
    collection_name: str = "flowai_memory"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1


class GuardrailConfig(BaseModel):
    """What the guardrails do with prompts and replies (see flowai.security.guardrails)."""
    #: "guard" reminds the model of its rules, "block" refuses the prompt, "off" does nothing.
    jailbreak: str = "guard"
    #: Mask API keys pasted into a prompt before it is sent to the provider.
    redact_prompt_secrets: bool = True
    max_prompt_chars: int = 400_000
    #: The same tool with the same arguments, more often than this in one turn, is refused. 0 = no limit.
    max_identical_tool_calls: int = 3


class MCPServerConfig(BaseModel):
    """An MCP server to start and borrow tools from."""
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class Settings(BaseModel):
    """Global FlowAI configuration."""
    project_name: str = "flowai-project"
    provider: str = "openai"
    model: str = "gpt-4o"
    max_iterations: int = 10
    temperature: float = 0.7
    #: "ask" prompts before a tool changes the machine; "auto" never prompts.
    approval: str = "ask"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    #: Plugins (by name) not to load.
    disabled_plugins: list[str] = Field(default_factory=list)
    #: MCP servers defined by the project file rather than by the user; they need the user's trust to start.
    project_mcp_servers: list[str] = Field(default_factory=list, exclude=True, repr=False)
    project_file: str | None = Field(default=None, exclude=True, repr=False)
    #: Things the user should know about their setup (a key in a project file, say).
    notices: list[str] = Field(default_factory=list, exclude=True, repr=False)

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        """Load settings: global user config, then project file, then environment."""
        data: dict[str, Any] = {}

        data.update(cls._load_global())
        project_data, project_path = cls._load_project(config_path)
        restricted, trust_notices = _restrict_project(project_data, data, project_path)
        data.update(restricted)
        if project_path and "project_name" not in project_data:
            data["project_name"] = project_path.parent.name
        cls._apply_environment(data)

        settings = cls(**data)
        if project_path:
            settings.project_file = str(project_path)
            settings.project_mcp_servers = sorted((project_data.get("mcp_servers") or {}).keys())
            settings.notices += trust_notices
            settings.notices += _plaintext_key_notices(project_data, project_path)
        settings._register_secrets()
        return settings

    @staticmethod
    def _load_global() -> dict[str, Any]:
        """Read ~/.flowai/config.json, leaving the keys it names where they are.

        Keys are not copied into os.environ: anything the agent starts would
        inherit them from there. Providers resolve them on demand instead.
        """
        return {k: v for k, v in keystore.read_config().items() if not _is_key_name(k)}

    @staticmethod
    def _load_project(config_path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
        """Find the nearest project config, walking up from the working directory."""
        path: Path | None = None
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            for directory in [Path.cwd(), *Path.cwd().parents]:
                for name in PROJECT_CONFIG_NAMES:
                    candidate = directory / name
                    if candidate.is_file():
                        path = candidate
                        break
                if path:
                    break

        if not path:
            return {}, None

        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml
            parsed = yaml.safe_load(content) or {}
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError as exc:  # Python 3.10
                raise ConfigurationError("flowai.toml needs Python 3.11+; use flowai.yaml instead.") from exc
            parsed = tomllib.loads(content)
        else:
            parsed = json.loads(content)

        if not isinstance(parsed, dict):
            return {}, path
        parsed.pop("version", None)
        return parsed, path

    @staticmethod
    def _apply_environment(data: dict[str, Any]) -> None:
        """Environment variables win, and a lone API key is enough to pick a provider."""
        if os.getenv("FLOWAI_PROVIDER"):
            data["provider"] = os.getenv("FLOWAI_PROVIDER")
            data.pop("model", None)
        elif "provider" not in data:
            for provider in PROVIDER_PREFERENCE:
                if keystore.resolve_api_key(*provider_key_envs(provider)):
                    data["provider"] = provider
                    break

        provider = data.get("provider")
        if provider and not data.get("model"):
            data["model"] = PROVIDER_DEFAULT_MODEL.get(provider, "gpt-4o")

        if os.getenv("FLOWAI_MODEL"):
            data["model"] = os.getenv("FLOWAI_MODEL")

    def _register_secrets(self) -> None:
        """Teach the redactor every key this process could end up holding."""
        register_environment_secrets()
        for provider in PROVIDER_KEY_ENV:
            key = self.provider_options(provider).get("api_key")
            if key:
                register_secret(key, PROVIDER_KEY_ENV[provider])

    # ------------------------------------------------------------- helpers

    def provider_options(self, provider: str | None = None) -> dict[str, Any]:
        """Keyword arguments for building this provider: its key and base URL, if any."""
        name = provider or self.provider
        block = self.providers.get(name)
        options: dict[str, Any] = {}
        key = None
        if block:
            if block.base_url:
                options["base_url"] = block.base_url
            key = block.api_key or (os.getenv(block.api_key_env) if block.api_key_env else None)
        key = key or keystore.resolve_api_key(*provider_key_envs(name))
        if key:
            options["api_key"] = key
        return options

    def has_credentials(self, provider: str | None = None) -> bool:
        """True when this provider can actually be called right now."""
        name = provider or self.provider
        if name == "ollama":
            return True
        options = self.provider_options(name)
        if name == "nvidia" and nvidia_is_self_hosted(options.get("base_url")):
            return True
        return bool(options.get("api_key"))

    def key_source(self, provider: str | None = None) -> str | None:
        """Where this provider's key comes from, in words, or None if there is none."""
        name = provider or self.provider
        block = self.providers.get(name)
        if block and block.api_key:
            return "project config file (plain text)"
        for env_name in provider_key_envs(name):
            location = keystore.key_location(env_name)
            if location == "environment":
                return f"environment variable {env_name}"
            if location == "keyring":
                return keystore.keyring_name() or "system keyring"
            if location == "file":
                return f"{keystore.config_path()} (plain text, owner-only)"
        return None


def _is_key_name(name: str) -> bool:
    return name.endswith("_API_KEY") or name in ("NVAPI_KEY", "NGC_API_KEY", "NIM_API_KEY", "GOOGLE_API_KEY")


def _plaintext_key_notices(project_data: dict[str, Any], path: Path) -> list[str]:
    """Warn about keys written into a project file, which tends to get committed."""
    found = [k for k in project_data if _is_key_name(k)]
    for name, block in (project_data.get("providers") or {}).items():
        if isinstance(block, dict) and block.get("api_key"):
            found.append(f"providers.{name}.api_key")
    if not found:
        return []
    return [
        f"{path.name} holds an API key in plain text ({', '.join(found)}). Project files get committed "
        "and shared; move the key out with 'flowai configure' and use api_key_env instead."
    ]


_JAILBREAK_STRICTNESS = {"off": 0, "guard": 1, "block": 2}


def _restrict_project(
    project: dict[str, Any], user: dict[str, Any], path: Path | None
) -> tuple[dict[str, Any], list[str]]:
    """Let a project file tighten security settings but never loosen them.

    A project file arrives with whatever repository you cloned, so it is not
    allowed to switch approvals off, weaken the guardrails or re-enable a
    plugin you disabled. Those settings belong in ~/.flowai/config.json.
    """
    if not project:
        return project, []
    data = dict(project)
    notices: list[str] = []
    name = path.name if path else "the project file"
    loosen = f"{name} tries to {{what}}; ignored, because project files may tighten security but not loosen it."

    if str(data.get("approval", "ask")).lower() == "auto" and str(user.get("approval", "ask")).lower() != "auto":
        data.pop("approval")
        notices.append(loosen.format(what="turn approvals off (approval: auto)")
                       + " Use --yolo for one session if you mean it.")

    if "guardrails" in data:
        theirs = data.pop("guardrails")
        mine = GuardrailConfig(**(user.get("guardrails") or {}))
        merged = mine.model_dump()
        refused: list[str] = []
        if isinstance(theirs, dict):
            mode = theirs.get("jailbreak")
            if mode is not None:
                if _JAILBREAK_STRICTNESS.get(str(mode), -1) >= _JAILBREAK_STRICTNESS[mine.jailbreak]:
                    merged["jailbreak"] = mode
                else:
                    refused.append(f"jailbreak: {mode}")
            if theirs.get("redact_prompt_secrets") is False and mine.redact_prompt_secrets:
                refused.append("redact_prompt_secrets: false")
            for key in ("max_prompt_chars", "max_identical_tool_calls"):
                value = theirs.get(key)
                if value is None:
                    continue
                current = merged[key]
                if isinstance(value, int) and 0 < value and (current == 0 or value <= current):
                    merged[key] = value
                else:
                    refused.append(f"{key}: {value}")
        data["guardrails"] = merged
        if refused:
            notices.append(loosen.format(what=f"weaken the guardrails ({', '.join(refused)})"))

    if "disabled_plugins" in data:
        data["disabled_plugins"] = sorted({*(user.get("disabled_plugins") or []), *(data["disabled_plugins"] or [])})
    return data, notices
