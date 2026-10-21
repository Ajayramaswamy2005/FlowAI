"""OpenTelemetry and LangSmith integration adapters."""

from __future__ import annotations

import os
from typing import Any

from flowai.observability.tracer import Span


class OpenTelemetryExporter:
    """Exports FlowAI spans to standard OpenTelemetry collector / APM."""

    def __init__(self, service_name: str = "flowai-agent", endpoint: str | None = None):
        self.service_name = service_name
        self.endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._tracer = None

    def export(self, span: Span) -> None:
        """Export completed span to OTel."""
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("flowai", "0.1.0")
            with tracer.start_as_current_span(span.name) as otel_span:
                otel_span.set_attribute("gen_ai.usage.prompt_tokens", span.prompt_tokens)
                otel_span.set_attribute("gen_ai.usage.completion_tokens", span.completion_tokens)
                otel_span.set_attribute("gen_ai.usage.cost_usd", span.cost_usd)
                for k, v in span.attributes.items():
                    otel_span.set_attribute(f"flowai.{k}", str(v))
        except ImportError:
            pass


class LangSmithExporter:
    """Exports agent traces to LangSmith platform."""

    def __init__(self, project_name: str | None = None):
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "flowai-default")
        self.api_key = os.getenv("LANGCHAIN_API_KEY")

    def export_run(self, name: str, run_type: str, inputs: dict[str, Any], outputs: dict[str, Any], **kwargs: Any) -> None:
        if not self.api_key:
            return
        try:
            from langsmith import Client
            client = Client(api_key=self.api_key)
            client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                project_name=self.project_name,
                extra=kwargs,
            )
        except Exception:
            pass
