"""Multi-agent execution patterns and topologies."""

from flowai.agents.patterns.debate import DebatePattern, Swarm
from flowai.agents.patterns.reflection import PlanAndExecutePattern, ReflectionPattern

__all__ = [
    "ReflectionPattern",
    "PlanAndExecutePattern",
    "DebatePattern",
    "Swarm",
]
