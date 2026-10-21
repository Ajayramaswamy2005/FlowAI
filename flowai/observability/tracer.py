"""Execution Tracer with Token Usage, Cost, and Latency tracking."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

# Pricing per million tokens (estimates in USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "meta/llama-3.1-70b-instruct": {"input": 0.90, "output": 0.90},
    "deepseek-ai/deepseek-r1": {"input": 0.55, "output": 2.19},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

# Used when a model is not in the table above: a visible estimate, not a quote.
DEFAULT_PRICING = {"input": 1.0, "output": 2.0}


class Span(BaseModel):
    """Represents a single traced execution operation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def finish(self) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


class Tracer:
    """Manages spans and calculates total resource spend."""

    def __init__(self):
        self.spans: list[Span] = []

    def start_span(self, name: str, **attributes: Any) -> Span:
        span = Span(name=name, attributes=attributes)
        self.spans.append(span)
        return span

    def record_usage(self, span: Span, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Add one model call's usage to a span.

        Usage accumulates: an agent turn that calls tools makes several model
        calls inside a single span, and all of them cost money.
        """
        span.prompt_tokens += prompt_tokens
        span.completion_tokens += completion_tokens
        span.total_tokens = span.prompt_tokens + span.completion_tokens

        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000.0
        span.cost_usd = round(span.cost_usd + cost, 6)

    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def summary(self) -> dict[str, Any]:
        return {
            "total_spans": len(self.spans),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": self.total_cost(),
            "spans": [s.model_dump() for s in self.spans],
        }
