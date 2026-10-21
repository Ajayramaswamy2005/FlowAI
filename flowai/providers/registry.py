"""Provider Registry and Dynamic Factory."""

from __future__ import annotations

from flowai.providers.anthropic import AnthropicProvider
from flowai.providers.base import BaseProvider
from flowai.providers.gemini import GeminiProvider
from flowai.providers.nvidia import NvidiaProvider
from flowai.providers.ollama import OllamaProvider
from flowai.providers.openai import OpenAIProvider, OpenRouterProvider


class ProviderRegistry:
    """Registry maintaining active provider classes and factory instantiations."""

    _registry: dict[str, type[BaseProvider]] = {
        "nvidia": NvidiaProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseProvider]) -> None:
        """Register a custom provider class."""
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def get(cls, name: str) -> type[BaseProvider]:
        """Fetch provider class by name."""
        provider_name = name.lower()
        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
        return cls._registry[provider_name]

    @classmethod
    def create(cls, name: str, model: str, **kwargs) -> BaseProvider:
        """Instantiate a provider by name."""
        provider_cls = cls.get(name)
        return provider_cls(model=model, **kwargs)


def get_provider(provider_or_name: str | BaseProvider, model: str = "gpt-4o", **kwargs) -> BaseProvider:
    """Convenience helper to resolve provider instance or name."""
    if isinstance(provider_or_name, BaseProvider):
        return provider_or_name
    return ProviderRegistry.create(provider_or_name, model=model, **kwargs)
