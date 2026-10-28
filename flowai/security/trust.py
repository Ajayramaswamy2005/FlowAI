"""Remembering which project-defined commands the user agreed to run.

An MCP server in a project's flowai.yaml is a program FlowAI would start the
moment you open that project, so a cloned repository could use one to run
anything. Servers from your own config start as before; a server from a
project file starts only after you approve that exact command for that
project. Change the command and it asks again.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from flowai.security.keystore import ensure_home, flowai_home, secure_write


def _store():
    return flowai_home() / "trusted.json"


def fingerprint(project_file: str, name: str, server: Any) -> str:
    material = json.dumps({
        "project": str(project_file),
        "name": name,
        "command": server.command,
        "args": list(server.args),
        "env": sorted(server.env),
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def _load() -> set[str]:
    try:
        data = json.loads(_store().read_text(encoding="utf-8"))
        return set(data.get("mcp", []))
    except (OSError, ValueError, AttributeError):
        return set()


def is_trusted(project_file: str, name: str, server: Any) -> bool:
    return fingerprint(project_file, name, server) in _load()


def trust(project_file: str, name: str, server: Any) -> None:
    known = _load()
    known.add(fingerprint(project_file, name, server))
    ensure_home()
    secure_write(_store(), json.dumps({"mcp": sorted(known)}, indent=2))
