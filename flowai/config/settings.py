"""Configuration: global user config, per-project file, and environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from flowai.errors import ConfigurationError

#: Where `flowai configure` keeps the provider and key chosen once per machine.
GLOBAL_CONFIG_DIR = Path.home() / ".flowai"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.json"

#: Project config filenames, in the order they are looked for.
PROJECT_CONFIG_NAMES = ("flowai.yaml", "flowai.yml", "flowai.toml", "flowai.json")

#: Environment variable holding each provider's key.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

#: Model used when a provider is chosen but no model is named.
PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
    "nvidia": "deepseek-ai/deepseek-r1",
    "gemini": "gemini-2.0-flash",
    "openrouter": "deepseek/deepseek-r1",
    "ollama": "llama3.2",
}

#: Order in which a provider is picked from whatever keys are present.
PROVIDER_PREFERENCE = ("anthropic", "openai", "nvidia", "gemini", "openrouter")


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

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        """Load settings: global user config, then project file, then environment."""
        data: dict[str, Any] = {}

        data.update(cls._load_global())
        project_data, project_path = cls._load_project(config_path)
        data.update(project_data)
        if project_path and "project_name" not in project_data:
            data["project_name"] = project_path.parent.name
        cls._apply_environment(data)

        return cls(**data)

    @staticmethod
    def _load_global() -> dict[str, Any]:
        """Read ~/.flowai/config.json and export any keys it holds to the environment."""
        if not GLOBAL_CONFIG_PATH.exists():
            return {}
        try:
            data = json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        # Provider SDKs and our own providers both read keys from the environment.
        for key, value in data.items():
            if key.endswith("_API_KEY") and isinstance(value, str) and not os.getenv(key):
                os.environ[key] = value
        return data

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
                if os.getenv(PROVIDER_KEY_ENV[provider]):
                    data["provider"] = provider
                    break

        provider = data.get("provider")
        if provider and not data.get("model"):
            data["model"] = PROVIDER_DEFAULT_MODEL.get(provider, "gpt-4o")

        if os.getenv("FLOWAI_MODEL"):
            data["model"] = os.getenv("FLOWAI_MODEL")

    # ------------------------------------------------------------- helpers

    def provider_options(self, provider: str | None = None) -> dict[str, Any]:
        """Keyword arguments for building this provider (key and base URL, if set)."""
        name = provider or self.provider
        block = self.providers.get(name)
        options: dict[str, Any] = {}
        if not block:
            return options
        if block.base_url:
            options["base_url"] = block.base_url
        key = block.api_key or (os.getenv(block.api_key_env) if block.api_key_env else None)
        if key:
            options["api_key"] = key
        return options

    def has_credentials(self, provider: str | None = None) -> bool:
        """True when this provider can actually be called right now."""
        name = provider or self.provider
        if name == "ollama":
            return True
        if self.provider_options(name).get("api_key"):
            return True
        env_name = PROVIDER_KEY_ENV.get(name)
        return bool(env_name and os.getenv(env_name))
