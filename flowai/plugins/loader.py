"""Finding and loading plugins, one at a time, so one broken plugin breaks nothing else.

Two places are searched, both chosen by the user rather than by a project:

- installed packages that declare an entry point in the `flowai.plugins` group
  (the entry point may name a Plugin subclass or instance, or a BaseProvider
  subclass, which is registered as a provider under the entry point's name);
- `*.py` files in ~/.flowai/plugins/, which may define Plugin subclasses or a
  module-level `plugin` object.

FLOWAI_NO_PLUGINS=1 (or `--no-plugins`) skips both; `disabled_plugins` in the
user's config skips plugins by name.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flowai.plugins.base import Plugin
from flowai.plugins.hooks import HookManager
from flowai.providers.base import BaseProvider
from flowai.providers.registry import ProviderRegistry
from flowai.security.keystore import flowai_home
from flowai.security.secrets import redact
from flowai.tools.base import BaseTool
from flowai.tools.decorator import tool as as_tool

ENTRY_POINT_GROUP = "flowai.plugins"


def plugins_dir() -> Path:
    return flowai_home() / "plugins"


def plugins_disabled() -> bool:
    return os.getenv("FLOWAI_NO_PLUGINS", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LoadedPlugin:
    name: str
    source: str
    version: str = ""
    description: str = ""
    tools: list[str] = field(default_factory=list)
    hooks: int = 0


@dataclass
class PluginSet:
    plugins: list[LoadedPlugin] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    tools: list[BaseTool] = field(default_factory=list)
    hooks: HookManager = field(default_factory=HookManager)
    #: (plugin or file, what went wrong) — shown to the user, never raised.
    errors: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _error(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")


class PluginLoader:
    """Discovers plugins and collects their tools and hooks into one PluginSet."""

    def __init__(self, disabled: set[str] | list[str] | None = None, directory: Path | None = None):
        self.disabled = {d.lower() for d in (disabled or [])}
        self.directory = directory

    # ------------------------------------------------------------- discovery

    @staticmethod
    def _entry_points() -> list[Any]:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINT_GROUP))
        return list(eps.get(ENTRY_POINT_GROUP, []))  # Python < 3.10 shape

    @staticmethod
    def _from_module(module: Any) -> list[Any]:
        found = [obj for obj in vars(module).values()
                 if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin
                 and obj.__module__ == module.__name__]
        explicit = getattr(module, "plugin", None)
        if explicit is not None:
            found = [explicit]
        return found

    def _import_file(self, path: Path) -> Any:
        module_name = f"flowai_user_plugins.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    # ---------------------------------------------------------------- loading

    def load(self, existing_tools: set[str] | None = None) -> PluginSet:
        result = PluginSet()
        if plugins_disabled():
            result.skipped.append("all plugins (FLOWAI_NO_PLUGINS)")
            return result
        taken = set(existing_tools or ())

        try:
            entry_points = self._entry_points()
        except Exception as exc:
            entry_points = []
            result.errors.append(("entry points", _error(exc)))
        for ep in entry_points:
            if ep.name.lower() in self.disabled:
                result.skipped.append(ep.name)
                continue
            try:
                obj = ep.load()
                if inspect.isclass(obj) and issubclass(obj, BaseProvider):
                    ProviderRegistry.register(ep.name, obj)
                    result.providers.append(ep.name)
                    continue
                self._add(obj, f"package {ep.value}", result, taken)
            except Exception as exc:
                result.errors.append((ep.name, _error(exc)))

        directory = self.directory or plugins_dir()
        if directory.is_dir():
            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_") or path.stem.lower() in self.disabled:
                    continue
                try:
                    module = self._import_file(path)
                    candidates = self._from_module(module)
                    if not candidates:
                        result.errors.append((path.name, "defines no Plugin subclass and no `plugin` object"))
                    for obj in candidates:
                        self._add(obj, str(path), result, taken)
                except Exception as exc:
                    result.errors.append((path.name, _error(exc)))
        return result

    def _add(self, obj: Any, source: str, result: PluginSet, taken: set[str]) -> None:
        plugin = obj() if inspect.isclass(obj) else obj
        if not isinstance(plugin, Plugin):
            raise TypeError(f"{type(plugin).__name__} is not a flowai Plugin")
        name = plugin.display_name
        if name.lower() in self.disabled:
            result.skipped.append(name)
            return
        if any(p.name == name for p in result.plugins):
            raise ValueError(f"a plugin named '{name}' is already loaded")

        # Collect everything first, so a plugin that fails half way adds nothing.
        tools: list[BaseTool] = []
        for t in plugin.tools() or []:
            instance = t if isinstance(t, BaseTool) else as_tool(t)
            if instance.name in taken or any(x.name == instance.name for x in tools):
                # A plugin must not be able to replace bash or read_file with its own version.
                raise ValueError(f"tool '{instance.name}' already exists; plugin tools cannot replace other tools")
            tools.append(instance)
        staged = HookManager()
        plugin.setup(staged)

        for t in tools:
            taken.add(t.name)
            result.tools.append(t)
        hooks = 0
        for point, fns in staged._hooks.items():
            for fn in fns:
                result.hooks.register(point, fn)
                hooks += 1
        result.plugins.append(LoadedPlugin(
            name=name,
            source=source,
            version=str(getattr(plugin, "version", "")),
            description=str(getattr(plugin, "description", "")),
            tools=[t.name for t in tools],
            hooks=hooks,
        ))

    @staticmethod
    def load_all() -> dict[str, list[str]]:
        """Older interface: load everything and report names by kind."""
        loaded = PluginLoader().load()
        return {
            "providers": loaded.providers,
            "tools": [t.name for t in loaded.tools],
            "plugins": [p.name for p in loaded.plugins],
        }
