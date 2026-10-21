"""Example 05: Multi-agent Debate and Swarm Patterns."""

import asyncio

from flowai import Agent, DebatePattern, Swarm
from flowai.providers.base import BaseProvider, ProviderResponse


class ScriptedProvider(BaseProvider):
    """Deterministic provider for simulation."""

    def __init__(self, script: dict[str, str]):
        super().__init__(model="scripted")
        self.script = script

    async def generate(self, messages, tools=None, **kwargs):
        last_msg = messages[-1].content
        for k, reply in self.script.items():
            if k in last_msg:
                return ProviderResponse(content=reply)
        return ProviderResponse(content="Default response.")

    async def stream(self, messages, tools=None, **kwargs):
        yield

    async def embeddings(self, texts, model=None, **kwargs):
        return []


async def test_debate():
    print("=== Multi-Agent Debate Pattern ===")
    provider_pro = ScriptedProvider({"IN FAVOR": "Remote work increases productivity and mental wellbeing."})
    provider_con = ScriptedProvider({"AGAINST": "In-office work fosters spontaneous collaboration and mentorship."})
    provider_judge = ScriptedProvider({"impartial judge": "Both presented strong points; hybrid mode is optimal."})

    debater_a = Agent("ProRemote", provider=provider_pro)
    debater_b = Agent("ProOffice", provider=provider_con)
    judge = Agent("Judge", provider=provider_judge)

    debate = DebatePattern(debater_a, debater_b, judge, rounds=1)
    result = await debate.execute("Should companies mandate 5 days in-office?")
    print("Debate Transcript:")
    for entry in result["transcript"]:
        print(f" - {entry}")
    print(f"\nVerdict:\n{result['verdict']}\n")


async def test_swarm():
    print("=== Multi-Agent Swarm Dynamic Handoff ===")
    triage_script = {
        "Python": "HANDOFF_TO: PythonExpert Please write the Python implementation.",
        "SQL": "HANDOFF_TO: SQLExpert Please optimize the SQL query.",
    }
    triage = Agent("TriageAgent", provider=ScriptedProvider(triage_script))
    py_agent = Agent("PythonExpert", provider=ScriptedProvider({"write": "def solve(): return 42"}))

    swarm = Swarm(agents=[triage, py_agent], initial_agent=triage)
    result = await swarm.execute("I need a Python function to compute answer to life.")
    print(f"Swarm Result:\n{result}")


async def main():
    await test_debate()
    await test_swarm()


if __name__ == "__main__":
    asyncio.run(main())
