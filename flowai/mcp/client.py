"""Model Context Protocol (MCP) Client and Tool Discovery."""

from __future__ import annotations

from typing import Any

from flowai.errors import ToolError
from flowai.security.secrets import scrubbed_env
from flowai.tools.base import BaseTool, ToolDefinition, ToolParameterSchema


class MCPClient:
    """MCP client for stdio servers: lists their tools and adapts them to FlowAI tools.

    Each tool call starts the server, makes the call, and stops it again. That
    is slower than a held-open session but needs no lifecycle management.
    """

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.command = command
        self.args = args or []
        self.env = env or {}

    async def connect_and_discover_tools(self) -> list[BaseTool]:
        """Discover tools exposed by the MCP server and adapt them to FlowAI BaseTools."""
        tools: list[BaseTool] = []
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                # The server gets this process's environment minus its credentials,
                # plus whatever its own config entry hands it on purpose.
                env=scrubbed_env(self.env),
            )

            # We create an MCP bridge tool wrapper
            class MCPRemoteTool(BaseTool):
                def __init__(self, tool_name: str, desc: str, schema: dict):
                    self.name = tool_name
                    self.description = desc
                    self.schema_definition = ToolDefinition(
                        name=tool_name,
                        description=desc,
                        parameters=ToolParameterSchema(
                            type="object",
                            properties=schema.get("properties", {}),
                            required=schema.get("required", []),
                        ),
                    )
                    self.is_async = True
                    # An external server can do anything, so its tools ask first,
                    # and the shield treats what they return as untrusted.
                    self.requires_approval = True
                    self.external = True

                async def execute(self, **kwargs: Any) -> Any:
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            res = await session.call_tool(self.name, arguments=kwargs)
                            # MCP returns content blocks; hand the agent their text.
                            texts = [getattr(block, "text", None) or str(block) for block in res.content]
                            text = "\n".join(texts)
                            return f"Error from MCP tool: {text}" if getattr(res, "isError", False) else text

            # Attempt listing tools
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = await session.list_tools()
                    for t in mcp_tools.tools:
                        tools.append(MCPRemoteTool(t.name, t.description or "", t.inputSchema))

        except ImportError as exc:
            raise ToolError("MCP support needs the mcp package: pip install 'flowai-cli[mcp]'") from exc

        return tools
