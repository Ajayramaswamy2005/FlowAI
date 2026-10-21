"""Tests for FlowAI Workflow DAG, State, and Checkpointing."""

import pytest

from flowai.workflow import AgentState, Workflow
from flowai.workflow.checkpoint import InMemoryCheckpointer


@pytest.mark.asyncio
async def test_workflow_sequential_execution():
    checkpointer = InMemoryCheckpointer()
    wf = Workflow("SequentialTest", checkpointer=checkpointer)

    def node_a(state: AgentState) -> AgentState:
        val = state.get("count", 0)
        return state.set("count", val + 1)

    def node_b(state: AgentState) -> AgentState:
        val = state.get("count", 0)
        return state.set("count", val * 2)

    wf.add_node("step_a", node_a)
    wf.add_node("step_b", node_b)
    wf.add_edge("step_a", "step_b")

    final_state = await wf.run({"count": 5}, entry_point="step_a")
    assert final_state.get("count") == 12  # (5 + 1) * 2

    # Check that checkpoints were created
    cps = await checkpointer.list_checkpoints(final_state.session_id)
    assert len(cps) >= 2


@pytest.mark.asyncio
async def test_workflow_conditional_branching():
    wf = Workflow("BranchTest")

    def evaluator(state: AgentState) -> AgentState:
        return state

    def router(state: AgentState) -> str:
        score = state.get("score", 0)
        return "pass_node" if score >= 80 else "fail_node"

    def pass_node(state: AgentState) -> AgentState:
        return state.set("status", "PASSED")

    def fail_node(state: AgentState) -> AgentState:
        return state.set("status", "FAILED")

    wf.add_node("evaluator", evaluator)
    wf.add_node("pass_node", pass_node)
    wf.add_node("fail_node", fail_node)

    wf.add_conditional_edge("evaluator", router)

    res_pass = await wf.run({"score": 95}, entry_point="evaluator")
    assert res_pass.get("status") == "PASSED"

    res_fail = await wf.run({"score": 60}, entry_point="evaluator")
    assert res_fail.get("status") == "FAILED"
