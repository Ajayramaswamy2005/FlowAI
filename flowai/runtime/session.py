"""Session management and persistent conversation storage on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flowai.providers.base import Message
from flowai.security.keystore import flowai_home, harden_permissions, secure_write
from flowai.security.secrets import redact


class SessionStore:
    """Manages persistent conversation sessions on disk (stored in ~/.flowai/sessions/)."""

    def __init__(self, storage_dir: Path | str | None = None):
        if storage_dir is None:
            self.storage_dir = flowai_home() / "sessions"
        else:
            self.storage_dir = Path(storage_dir)
        fresh = not self.storage_dir.exists()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            harden_permissions(self.storage_dir, directory=True)

    def _get_path(self, session_id: str) -> Path:
        # Session ids come from the command line; keep them from naming paths elsewhere.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "session"
        return self.storage_dir / f"{safe}.json"

    def save(self, session_id: str, messages: list[Message], metadata: dict[str, Any] | None = None) -> Path:
        """Persist message history and metadata to disk."""
        data = {
            "session_id": session_id,
            "metadata": metadata or {},
            "messages": [m.model_dump() for m in messages],
        }
        path = self._get_path(session_id)
        # A transcript can hold a key the user pasted; it is written redacted and owner-only.
        secure_write(path, redact(json.dumps(data, indent=2)), harden=False)

        # Update a 'latest' pointer for instant resumption
        latest_path = self.storage_dir / "latest.txt"
        secure_write(latest_path, session_id, harden=False)
        return path

    def load(self, session_id: str) -> list[Message]:
        """Load messages from an existing disk session."""
        path = self._get_path(session_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [Message(**m) for m in raw.get("messages", [])]
        except Exception:
            return []

    def get_latest_session_id(self) -> str | None:
        """Retrieve ID of the most recent session."""
        latest_path = self.storage_dir / "latest.txt"
        if latest_path.exists():
            sid = latest_path.read_text(encoding="utf-8").strip()
            if self._get_path(sid).exists():
                return sid
        # Fallback to the newest json file in the directory
        files = sorted(self.storage_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0].stem if files else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with summary statistics."""
        sessions = []
        for p in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", p.stem),
                    "turns": len(data.get("messages", [])),
                    "path": str(p),
                    "modified": p.stat().st_mtime,
                })
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s["modified"], reverse=True)
