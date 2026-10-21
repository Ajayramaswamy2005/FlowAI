"""Tests for persistent session storage."""

from pathlib import Path

from flowai.providers.base import Message
from flowai.runtime.session import SessionStore


def test_session_store_save_and_load(tmp_path: Path):
    store = SessionStore(storage_dir=tmp_path)
    session_id = "test_ses_001"

    msgs = [
        Message(role="user", content="Hello, remember my secret number is 777."),
        Message(role="assistant", content="I will remember 777."),
    ]

    saved_path = store.save(session_id, msgs)
    assert saved_path.exists()

    loaded = store.load(session_id)
    assert len(loaded) == 2
    assert loaded[0].content == "Hello, remember my secret number is 777."
    assert loaded[1].content == "I will remember 777."

    assert store.get_latest_session_id() == session_id
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
