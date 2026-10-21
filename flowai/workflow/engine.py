"""Async Workflow Engine with DAG waves, branching, loops, and parallel execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from flowai.agents.base import BaseAgent
from flowai.observability.events import EventBus
from flowai.workflow.checkpoint import BaseCheckpointer
from flowai.workflow.state import AgentState

NodeFn = Callable[[AgentState], AgentState | Coroutine[Any, Any, AgentState]]
RouterFn = Callable[[AgentState], str | Coroutine[Any, Any, str]]

START = "START"
END = "END"


class Workflow:
    """Async Graph Workflow Engine supporting branching, loops, and parallel nodes."""

    def __init__(
        self,
        name: str = "Workflow",
        checkpointer: BaseCheckpointer | None = None,
        event_bus: EventBus | None = None,
        max_steps: int = 50,
    ):
        self.name = name
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[str]] = {}
        self.routers: dict[str, RouterFn] = {}
        self.checkpointer = checkpointer
        self.event_bus = event_bus or EventBus()
        self.max_steps = max_steps

    def add_node(self, name: str, fn_or_agent: NodeFn | BaseAgent) -> Workflow:
        """Register a node function or agent."""
        if isinstance(fn_or_agent, BaseAgent):
            agent = fn_or_agent

            async def agent_node(state: AgentState) -> AgentState:
                prompt = state.get("task", "") or (state.messages[-1].content if state.messages else "")
                reply = await agent.run(prompt)
                return state.set(f"{name}_output", reply).append_message({"role": "assistant", "content": reply})

            self.nodes[name] = agent_node
        else:
            self.nodes[name] = fn_or_agent
        self.edges.setdefault(name, [])
        return self

    def add_agent(self, agent: BaseAgent, name: str | None = None) -> Workflow:
        """Convenience method to register an agent as a workflow node."""
        node_name = name or agent.name
        return self.add_node(node_name, agent)

    def add_edge(self, from_node: str, to_node: str) -> Workflow:
        """Add a directed edge from one node to another."""
        self.edges.setdefault(from_node, []).append(to_node)
        return self

    def add_conditional_edge(self, from_node: str, router: RouterFn) -> Workflow:
        """Add a dynamic routing function to branch or loop based on state."""
        self.routers[from_node] = router
        return self

    async def execute_parallel(self, node_names: list[str], state: AgentState) -> AgentState:
        """Run multiple ready nodes in parallel and merge disjoint state updates."""
        tasks = []
        for name in node_names:
            fn = self.nodes[name]
            if asyncio.iscoroutinefunction(fn):
                tasks.append(fn(state))
            else:
                tasks.append(asyncio.to_thread(fn, state))

        results: list[AgentState] = await asyncio.gather(*tasks)
        merged = state
        for res in results:
            merged = merged.update_values(res.values)
            for m in res.messages[len(state.messages):]:
                merged = merged.append_message(m)
        return merged

    async def run(self, initial_input: str | dict[str, Any] | AgentState, entry_point: str = "START") -> AgentState:
        """Execute the workflow starting from entry_point until END or max_steps."""
        if isinstance(initial_input, AgentState):
            state = initial_input
        elif isinstance(initial_input, str):
            state = AgentState(values={"task": initial_input}).append_message({"role": "user", "content": initial_input})
        else:
            state = AgentState(values=initial_input)

        current_node = entry_point
        # If START is not an explicit node, jump to first edge from START or first registered node
        if current_node == START and START not in self.nodes:
            if START in self.edges and self.edges[START]:
                current_node = self.edges[START][0]
            elif self.nodes:
                current_node = list(self.nodes.keys())[0]
            else:
                return state

        step_count = 0
        while current_node != END and step_count < self.max_steps:
            step_count += 1
            await self.event_bus.publish("node_start", {"node": current_node, "step": step_count})

            if self.checkpointer:
                await self.checkpointer.save(state)

            fn = self.nodes.get(current_node)
            if fn:
                if asyncio.iscoroutinefunction(fn):
                    state = await fn(state)
                else:
                    state = await asyncio.to_thread(fn, state)

            await self.event_bus.publish("node_end", {"node": current_node, "step": step_count})

            # Check dynamic router
            if current_node in self.routers:
                router = self.routers[current_node]
                next_node = await router(state) if asyncio.iscoroutinefunction(router) else router(state)
                current_node = next_node
            # Check static edges
            elif self.edges.get(current_node):
                targets = self.edges[current_node]
                if len(targets) > 1:
                    # Parallel branch execution
                    state = await self.execute_parallel(targets, state)
                    current_node = END
                else:
                    current_node = targets[0]
            else:
                current_node = END

        if self.checkpointer:
            await self.checkpointer.save(state)

        return state
