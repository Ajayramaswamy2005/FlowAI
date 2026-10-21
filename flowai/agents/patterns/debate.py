"""Debate and Swarm multi-agent interaction topologies."""

from __future__ import annotations

from typing import Any

from flowai.agents.standard import Agent


class DebatePattern:
    """Multi-agent adversarial debate with a judge/moderator."""

    def __init__(self, debater_a: Agent, debater_b: Agent, judge: Agent, rounds: int = 2):
        self.debater_a = debater_a
        self.debater_b = debater_b
        self.judge = judge
        self.rounds = rounds

    async def execute(self, topic: str) -> dict[str, Any]:
        transcript = []
        last_arg_a = await self.debater_a.run(f"Argue IN FAVOR of: {topic}")
        transcript.append(f"{self.debater_a.name}: {last_arg_a}")

        last_arg_b = await self.debater_b.run(f"Argue AGAINST: {topic}\nOpponent argument: {last_arg_a}")
        transcript.append(f"{self.debater_b.name}: {last_arg_b}")

        for r in range(1, self.rounds):
            last_arg_a = await self.debater_a.run(f"Rebut: {last_arg_b}\nTopic: {topic}")
            transcript.append(f"{self.debater_a.name} (Round {r+1}): {last_arg_a}")

            last_arg_b = await self.debater_b.run(f"Rebut: {last_arg_a}\nTopic: {topic}")
            transcript.append(f"{self.debater_b.name} (Round {r+1}): {last_arg_b}")

        full_transcript = "\n\n".join(transcript)
        verdict = await self.judge.run(
            f"You are the impartial judge of this debate on '{topic}'. Review the transcript and determine the winner and reasoning:\n\n{full_transcript}"
        )

        return {
            "transcript": transcript,
            "verdict": verdict,
        }


class Swarm:
    """Dynamic agent handoff network where agents can pass execution to peers."""

    def __init__(self, agents: list[Agent], initial_agent: Agent | None = None):
        self.agents = {a.name: a for a in agents}
        self.current_agent = initial_agent or agents[0]

    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    async def execute(self, prompt: str, max_handoffs: int = 5) -> str:
        current = self.current_agent
        current_prompt = prompt

        for _ in range(max_handoffs):
            # inject peers into context
            peer_names = [name for name in self.agents.keys() if name != current.name]
            directive = (
                f"\n[SYSTEM NOTICE]: If another agent is better suited, respond with 'HANDOFF_TO: <AgentName> <Instructions>'. "
                f"Available agents: {', '.join(peer_names)}"
            )
            response = await current.run(current_prompt + directive)

            if "HANDOFF_TO:" in response:
                parts = response.split("HANDOFF_TO:", 1)[1].strip().split(" ", 1)
                target_name = parts[0].strip()
                handoff_instruction = parts[1].strip() if len(parts) > 1 else current_prompt
                if target_name in self.agents:
                    current = self.agents[target_name]
                    current_prompt = handoff_instruction
                    continue

            return response

        return f"Completed with agent {current.name}: {response}"
