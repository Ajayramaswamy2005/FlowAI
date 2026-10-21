"""@tool decorator with automatic Pydantic v2 schema generation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import create_model

from flowai.tools.base import BaseTool, ToolDefinition, ToolParameterSchema


def tool(
    name_or_fn: str | Callable | None = None,
    description: str | None = None,
    requires_approval: bool = False,
):
    """Decorator to turn any standard or async Python function into a FlowAI tool.

    Example:
        @tool
        def calculate_tax(amount: float, rate: float = 0.15) -> float:
            '''Calculate total tax for an amount.'''
            return amount * rate

        @tool("web_search", description="Search the web for query")
        async def fetch_results(query: str, max_results: int = 5):
            ...

    Pass requires_approval=True for a tool that changes the machine; a registry
    with an approver attached will ask the user before each call.
    """

    _requires_approval = requires_approval

    def decorator(fn: Callable[..., Any]) -> BaseTool:
        tool_name = (name_or_fn if isinstance(name_or_fn, str) else None) or fn.__name__
        raw_doc = fn.__doc__ or ""
        tool_desc = (description or raw_doc.strip()) or f"Tool {tool_name}"

        # Inspect parameters and types
        sig = inspect.signature(fn)
        type_hints = {}
        try:
            type_hints = get_type_hints(fn)
        except Exception:
            pass

        fields: dict[str, Any] = {}
        required_fields: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = type_hints.get(param_name, Any)
            default = ... if param.default == inspect.Parameter.empty else param.default
            if default is ...:
                required_fields.append(param_name)
            fields[param_name] = (param_type, default)

        # Build dynamic Pydantic model for validation & schema generation
        model_name = f"{tool_name.capitalize()}Args"
        ArgsModel = create_model(model_name, **fields)
        json_schema = ArgsModel.model_json_schema()

        param_schema = ToolParameterSchema(
            type="object",
            properties=json_schema.get("properties", {}),
            required=required_fields,
        )

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=param_schema,
        )

        class DecoratedTool(BaseTool):
            name = tool_name
            description = tool_desc
            schema_definition = tool_def
            is_async = inspect.iscoroutinefunction(fn)
            requires_approval = _requires_approval

            def __init__(self):
                self._fn = fn
                self._args_model = ArgsModel

            async def execute(self, **kwargs: Any) -> Any:
                validated_args = self._args_model(**kwargs).model_dump()
                if self.is_async:
                    return await self._fn(**validated_args)
                return self._fn(**validated_args)

            def __repr__(self) -> str:
                return f"<Tool {self.name}: {self.description}>"

        return DecoratedTool()

    if callable(name_or_fn):
        return decorator(name_or_fn)
    return decorator
