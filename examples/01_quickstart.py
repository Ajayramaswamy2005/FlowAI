"""Example 01: Quickstart with FlowAI."""

import asyncio

from flowai import Agent, tool
from flowai.memory import ConversationMemory


@tool
def calculate_growth(initial: float, rate: float, periods: int) -> float:
    """Calculate compound interest growth."""
    return round(initial * ((1 + rate) ** periods), 2)


async def main():
    # Initialize an agent with in-memory conversation memory
    memory = ConversationMemory()
    agent = Agent(
        name="FinanceAgent",
        provider="openai",  # or "nvidia", "ollama", "anthropic", "gemini"
        model="gpt-4o-mini",
        tools=[calculate_growth],
        memory=memory,
    )

    print("=== FlowAI Quickstart Agent ===")
    response = await agent.run("If I invest $10,000 at 7% for 5 years, what will it be worth?")
    print(f"Agent Reply:\n{response}\n")

    # Conversation history is preserved
    print("=== Preserved Conversation History ===")
    print(memory.format_history())


if __name__ == "__main__":
    asyncio.run(main())
