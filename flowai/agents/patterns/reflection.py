"""Reflection and Plan-and-Execute multi-agent patterns."""

from __future__ import annotations

from typing import Any

from flowai.agents.specialized import CriticAgent, PlannerAgent
from flowai.agents.standard import Agent


class ReflectionPattern:
    """Actor-Critic reflection pattern: generate candidate -> critique -> refine."""

    def __init__(self, actor: Agent, critic: CriticAgent, max_rounds: int = 3):
        self.actor = actor
        self.critic = critic
        self.max_rounds = max_rounds

    async def execute(self, task: str) -> dict[str, Any]:
        current_solution = await self.actor.run(f"Task: {task}\nProvide an initial solution:")
        history = []

        for round_idx in range(self.max_rounds):
            evaluation = await self.critic.evaluate(task, current_solution)
            history.append({
                "round": round_idx + 1,
                "solution": current_solution,
                "critique": evaluation.get("critique"),
                "is_approved": evaluation.get("is_approved"),
            })

            if evaluation.get("is_approved"):
                break

            # Refine solution based on critique
            refine_prompt = (
                f"Your previous solution received this critique:\n{evaluation.get('critique')}\n\n"
                f"Please produce an improved and corrected version of the solution for:\n{task}"
            )
            current_solution = await self.actor.run(refine_prompt)

        return {
            "final_solution": current_solution,
            "rounds_completed": len(history),
            "history": history,
        }


class PlanAndExecutePattern:
    """Planner decomposes task, Worker executes each step sequentially."""

    def __init__(self, planner: PlannerAgent, worker: Agent):
        self.planner = planner
        self.worker = worker

    async def execute(self, goal: str) -> dict[str, Any]:
        plan = await self.planner.create_plan(goal)
        step_results = []
        cumulative_context = ""

        for i, step in enumerate(plan, 1):
            step_prompt = f"Goal: {goal}\nPrevious context:\n{cumulative_context}\nExecute Step {i}: {step}"
            result = await self.worker.run(step_prompt)
            step_results.append({"step": step, "output": result})
            cumulative_context += f"\nStep {i} ({step}) output:\n{result}\n"

        summary_prompt = f"Goal: {goal}\nReview all step outputs and provide a final answer:\n{cumulative_context}"
        final_answer = await self.worker.run(summary_prompt)

        return {
            "plan": plan,
            "step_results": step_results,
            "final_answer": final_answer,
        }
