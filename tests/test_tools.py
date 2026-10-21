"""Tests for FlowAI @tool decorator and ToolRegistry."""

import asyncio

import pytest

from flowai.tools import BaseTool, ToolRegistry, tool


def test_tool_decorator_sync():
    @tool
    def add(a: int, b: int = 10) -> int:
        """Add two numbers together."""
        return a + b

    assert isinstance(add, BaseTool)
    assert add.name == "add"
    assert "Add two numbers" in add.description
    assert add.is_async is False

    schema = add.schema_definition.parameters.model_dump()
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    assert "a" in schema["required"]
    assert "b" not in schema["required"]


@pytest.mark.asyncio
async def test_tool_execution_async():
    @tool("async_multiplier")
    async def multiply(x: float, y: float) -> float:
        await asyncio.sleep(0.01)
        return x * y

    assert multiply.is_async is True
    res = await multiply.execute(x=3.5, y=2.0)
    assert res == 7.0


@pytest.mark.asyncio
async def test_tool_registry():
    @tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry = ToolRegistry([greet])
    assert len(registry.list_tools()) == 1
    assert registry.get("greet") is not None

    output = await registry.execute("greet", name="FlowAI")
    assert output == "Hello, FlowAI!"

    # Surface-don't-crash behavior
    err_output = await registry.execute("greet")
    assert "Error executing greet" in err_output
