"""Example 02: NVIDIA API Catalog with DeepSeek-R1 / DeepSeek-V4."""

import asyncio

from flowai import Agent, tool


@tool
def check_server_health(host: str) -> dict:
    """Check system resource status and latency."""
    return {"host": host, "status": "healthy", "cpu_percent": 24.5, "latency_ms": 12.4}


async def main():
    # Make sure NVIDIA_API_KEY is set in your environment
    agent = Agent(
        name="NvidiaDeepSeekAgent",
        provider="nvidia",
        model="deepseek-ai/deepseek-r1",
        system_prompt="You are a DevOps autonomous triage engineer.",
        tools=[check_server_health],
    )

    print("=== Invoking DeepSeek on NVIDIA API Catalog ===")
    prompt = "Check the health of 'prod-cluster-us-east-1' and summarize your findings."

    # Non-streaming invocation
    result = await agent.run(prompt)
    print("\n--- Agent Conclusion ---")
    print(result)

    # Token streaming invocation
    print("\n--- Streaming Response ---")
    async for chunk in agent.stream("Recommend 2 optimization actions."):
        print(chunk.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
