"""Example 02: an agent on NVIDIA's API catalog, with streamed reasoning.

Needs an NVIDIA key in any of NVIDIA_API_KEY, NVAPI_KEY, NGC_API_KEY or
NIM_API_KEY (or saved with `flowai configure`). For a NIM you run yourself,
set NVIDIA_BASE_URL=http://localhost:8000/v1 instead; no key is needed.
List the models your key can use with: flowai models nvidia
"""

import asyncio

from flowai import Agent, tool


@tool
def check_server_health(host: str) -> dict:
    """Check system resource status and latency."""
    return {"host": host, "status": "healthy", "cpu_percent": 24.5, "latency_ms": 12.4}


async def main():
    agent = Agent(
        name="NvidiaAgent",
        provider="nvidia",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        system_prompt="You are a DevOps autonomous triage engineer.",
        tools=[check_server_health],
    )

    print("=== NVIDIA API catalog ===")
    prompt = "Check the health of 'prod-cluster-us-east-1' and summarize your findings."

    # Non-streaming invocation
    result = await agent.run(prompt)
    print("\n--- Agent Conclusion ---")
    print(result)

    # Streaming: reasoning models send their thinking separately from the answer.
    print("\n--- Streaming Response ---")
    async for chunk in agent.stream("Recommend 2 optimization actions."):
        if chunk.reasoning_delta:
            print(f"\033[2m{chunk.reasoning_delta}\033[0m", end="", flush=True)
        print(chunk.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
