"""Tests for FlowAI Memory subsystem."""

import pytest

from flowai.memory import ConversationMemory, InMemory, ShortTermMemory


@pytest.mark.asyncio
async def test_in_memory_crud():
    mem = InMemory()
    rec_id = await mem.add("The secret code is 4242", topic="security")
    assert rec_id is not None

    hits = await mem.search("secret")
    assert len(hits) == 1
    assert "4242" in hits[0].content

    hits_none = await mem.search("nonexistent")
    assert len(hits_none) == 0


@pytest.mark.asyncio
async def test_short_term_sliding_window():
    mem = ShortTermMemory(capacity=3)
    for i in range(5):
        await mem.add(f"Message {i}")

    recent = await mem.get_recent(10)
    assert len(recent) == 3
    assert recent[0].content == "Message 2"
    assert recent[2].content == "Message 4"


@pytest.mark.asyncio
async def test_conversation_memory_history():
    mem = ConversationMemory()
    await mem.add("Hi there", role="user")
    await mem.add("Hello! How can I assist you?", role="assistant")

    formatted = mem.format_history()
    assert "User: Hi there" in formatted
    assert "Assistant: Hello!" in formatted
