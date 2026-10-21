"""Example 03: Multi-agent Workflow DAG with State Passing and Checkpoints."""

import asyncio

from flowai import Agent, Workflow
from flowai.workflow.checkpoint import InMemoryCheckpointer


async def main():
    # 1. Instantiate specialized agents
    researcher = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        system_prompt="You are a Technical Researcher. Provide facts, architecture patterns, and benchmarks.",
    )
    coder = Agent(
        name="Coder",
        model="gpt-4o-mini",
        system_prompt="You are a Principal Software Engineer. Write clean, idiomatic Python code based on research.",
    )
    reviewer = Agent(
        name="Reviewer",
        model="gpt-4o-mini",
        system_prompt="You are a Code Reviewer. Inspect code for bugs, edge cases, and performance bottlenecks.",
    )

    # 2. Construct DAG Workflow with Checkpointing
    checkpointer = InMemoryCheckpointer()
    wf = Workflow(name="CodeGenerationPipeline", checkpointer=checkpointer)

    wf.add_agent(researcher, name="research_node")
    wf.add_agent(coder, name="coding_node")
    wf.add_agent(reviewer, name="review_node")

    # Connect nodes in a linear sequential pipeline
    wf.add_edge("research_node", "coding_node")
    wf.add_edge("coding_node", "review_node")

    # 3. Execute workflow
    task = "Design and implement a thread-safe token bucket rate limiter in Python."
    print(f"Starting Workflow for task: '{task}' ...\n")
    final_state = await wf.run(task, entry_point="research_node")

    print("=== Final Pipeline State Output ===")
    print(f"Total Steps Executed: {final_state.step}")
    print(f"Researcher Output:\n{final_state.get('research_node_output')[:300]}...\n")
    print(f"Coder Output:\n{final_state.get('coding_node_output')[:300]}...\n")
    print(f"Reviewer Output:\n{final_state.get('review_node_output')}\n")

    # Verify checkpoints
    checkpoints = await checkpointer.list_checkpoints(final_state.session_id)
    print(f"Checkpoints persisted: {checkpoints}")


if __name__ == "__main__":
    asyncio.run(main())
