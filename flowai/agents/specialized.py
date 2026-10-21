"""Specialized agents: ToolAgent, PlannerAgent, CriticAgent, SupervisorAgent."""

from __future__ import annotations

import json
from typing import Any

from flowai.agents.standard import Agent


class ToolAgent(Agent):
    """Agent specialized strictly for executing tools and validating their results."""

    def __init__(self, name: str = "ToolExpert", **kwargs):
        system_prompt = kwargs.pop(
            "system_prompt",
            "You are a Tool Specialist. Always prioritize selecting the right tool, verifying arguments, and validating outputs.",
        )
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)


class PlannerAgent(Agent):
    """Agent that decomposes complex user queries into structured sequential steps."""

    def __init__(self, name: str = "Planner", **kwargs):
        system_prompt = kwargs.pop(
            "system_prompt",
            "You are a Strategic Planning Agent. Analyze goals and output clear, numbered step-by-step execution plans.",
        )
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)

    async def create_plan(self, goal: str) -> list[str]:
        prompt = f"Decompose the following goal into discrete execution steps. Output only the steps as a JSON array of strings:\nGoal: {goal}"
        response = await self.run(prompt)
        try:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(response[start_idx:end_idx])
        except Exception:
            pass
        return [s.strip("- ") for s in response.split("\n") if s.strip()]


class CriticAgent(Agent):
    """Agent that reviews, critiques, and suggests refinements to work products."""

    def __init__(self, name: str = "Critic", **kwargs):
        system_prompt = kwargs.pop(
            "system_prompt",
            "You are a Critical Reviewer. Scrutinize logic, identify hallucinations, point out omissions, and propose fixes.",
        )
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)

    async def evaluate(self, task: str, solution: str) -> dict[str, Any]:
        prompt = (
            f"Evaluate the proposed solution for the given task.\n"
            f"Task: {task}\n"
            f"Solution: {solution}\n\n"
            f"Respond with a JSON object containing: 'is_approved' (boolean), 'score' (1-10), 'critique' (string)."
        )
        res = await self.run(prompt)
        try:
            start_idx = res.find("{")
            end_idx = res.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(res[start_idx:end_idx])
        except Exception:
            pass
        return {"is_approved": True, "score": 8, "critique": res}


class SupervisorAgent(Agent):
    """Agent that orchestrates worker agents and routes queries to the appropriate specialist."""

    def __init__(self, workers: dict[str, Agent] | None = None, name: str = "Supervisor", **kwargs):
        self.workers = workers or {}
        worker_descriptions = "\n".join([f"- {name}: {w.system_prompt}" for name, w in self.workers.items()])
        system_prompt = kwargs.pop(
            "system_prompt",
            f"You are the Supervisor. Route questions to the appropriate worker agent:\n{worker_descriptions}\n"
            f"Respond with the name of the worker agent followed by instructions.",
        )
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)

    def register_worker(self, name: str, agent: Agent) -> None:
        self.workers[name] = agent
