"""FlowAI Plugin interfaces and dynamic entry_points loader."""

from __future__ import annotations

import importlib.metadata

from flowai.providers.base import BaseProvider
from flowai.providers.registry import ProviderRegistry


class PluginLoader:
    """Discovers and registers external plugins via Python entry points."""

    @staticmethod
    def load_all() -> dict[str, list[str]]:
        loaded = {"providers": [], "tools": [], "memory": []}

        # Discover entry_points in 'flowai.plugins'
        try:
            entry_points = importlib.metadata.entry_points()
            flowai_eps = entry_points.select(group="flowai.plugins") if hasattr(entry_points, "select") else entry_points.get("flowai.plugins", [])
            for ep in flowai_eps:
                plugin_obj = ep.load()
                name = ep.name
                if issubclass(plugin_obj, BaseProvider):
                    ProviderRegistry.register(name, plugin_obj)
                    loaded["providers"].append(name)
        except Exception:
            pass

        return loaded
