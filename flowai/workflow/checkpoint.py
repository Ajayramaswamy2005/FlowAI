"""State checkpointing and resumable execution persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from flowai.workflow.state import AgentState


class BaseCheckpointer(ABC):
    """Abstract interface for storing state checkpoints."""

    @abstractmethod
    async def save(self, state: AgentState) -> str:
        """Persist state and return checkpoint ID."""
        ...

    @abstractmethod
    async def load(self, checkpoint_id: str) -> AgentState | None:
        """Retrieve state from checkpoint."""
        ...

    @abstractmethod
    async def list_checkpoints(self, session_id: str) -> list[str]:
        """List checkpoint IDs for a given session."""
        ...


class InMemoryCheckpointer(BaseCheckpointer):
    """Ephemeral in-memory checkpoint store."""

    def __init__(self):
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._session_index: dict[str, list[str]] = {}

    async def save(self, state: AgentState) -> str:
        checkpoint_id = f"{state.session_id}_step_{state.step}"
        self._checkpoints[checkpoint_id] = state.model_dump()
        self._session_index.setdefault(state.session_id, []).append(checkpoint_id)
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> AgentState | None:
        data = self._checkpoints.get(checkpoint_id)
        return AgentState(**data) if data else None

    async def list_checkpoints(self, session_id: str) -> list[str]:
        return self._session_index.get(session_id, [])


class FileCheckpointer(BaseCheckpointer):
    """Filesystem JSON checkpointer for local resumable workflows."""

    def __init__(self, directory: str = ".flowai/checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, state: AgentState) -> str:
        checkpoint_id = f"{state.session_id}_step_{state.step}"
        path = self.directory / f"{checkpoint_id}.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> AgentState | None:
        path = self.directory / f"{checkpoint_id}.json"
        if not path.exists():
            return None
        return AgentState.model_validate_json(path.read_text(encoding="utf-8"))

    async def list_checkpoints(self, session_id: str) -> list[str]:
        files = self.directory.glob(f"{session_id}_step_*.json")
        return sorted([f.stem for f in files])
