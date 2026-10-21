"""Tests for FlowAI Observability: Tracer, Spans, Cost, and EventBus."""

import pytest

from flowai.observability import EventBus, StreamEvent, Tracer


@pytest.mark.asyncio
async def test_event_bus():
    bus = EventBus()
    received = []

    async def on_event(ev: StreamEvent):
        received.append(ev)

    bus.subscribe(on_event)
    await bus.publish("test_event", {"value": 123})

    assert len(received) == 1
    assert received[0].event == "test_event"
    assert received[0].data["value"] == 123


def test_tracer_cost_and_latency():
    tracer = Tracer()
    span = tracer.start_span("agent_turn", model="gpt-4o")

    tracer.record_usage(span, model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
    span.finish()

    assert span.duration_ms >= 0.0
    assert span.total_tokens == 1500
    assert span.cost_usd > 0.0

    summary = tracer.summary()
    assert summary["total_spans"] == 1
    assert summary["total_tokens"] == 1500
    assert summary["total_cost_usd"] == span.cost_usd
