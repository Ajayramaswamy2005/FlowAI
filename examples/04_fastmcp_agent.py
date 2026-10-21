"""Example 04: FastMCP / Model Context Protocol Integration."""

import asyncio

from flowai import Agent
from flowai.mcp import MCPClient


async def main():
    print("=== FlowAI MCP Integration Example ===")

    # Initialize client to an MCP server (e.g. SQLite, GitHub, or custom FastMCP server)
    mcp_client = MCPClient(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    )

    print("Connecting to MCP server and discovering exposed tools...")
    tools = await mcp_client.connect_and_discover_tools()
    print(f"Discovered {len(tools)} MCP tools:")
    for t in tools:
        print(f" - {t.name}: {t.description}")

    # Plug discovered MCP tools directly into a FlowAI agent
    agent = Agent(
        name="MCPWorker",
        provider="nvidia",
        model="deepseek-ai/deepseek-r1",
        tools=tools,
    )

    reply = await agent.run("List the tools you have and pick one to demonstrate.")
    print(f"\n{agent.name}: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
