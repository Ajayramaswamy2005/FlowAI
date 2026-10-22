"""Standard Agent and AssistantAgent implementations."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from flowai.agents.base import BaseAgent
from flowai.providers.base import BaseProvider, Message, StreamChunk, ToolCall


class Agent(BaseAgent):
    """Core autonomous Agent capable of reasoning, tool execution, and memory recall."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.messages: list[Message] = [
            Message(role="system", content=self.system_prompt)
        ]

    # ---------------------------------------------------------------- helpers

    def switch_model(self, model: str, provider: str | BaseProvider | None = None) -> None:
        """Switch provider or model mid-conversation while preserving full context history."""
        from flowai.providers.registry import get_provider
        self.model = model
        if provider:
            self.provider = get_provider(provider, model=model) if isinstance(provider, str) else provider
        else:
            # Re-instantiate current provider with new model
            provider_name = self.provider.__class__.__name__.replace("Provider", "").lower()
            self.provider = get_provider(provider_name, model=model, api_key=self.provider.api_key, base_url=self.provider.base_url)

    def reset_session(self) -> None:
        """Clear conversation history back to initial system prompt."""
        self.messages = [Message(role="system", content=self.system_prompt)]

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        if not self.tools or not self.tools.list_tools():
            return None
        return self.tools.get_schemas(format="openai")

    async def _recall(self, prompt: str) -> None:
        """Pull relevant long-term memories into context, if a memory backend is attached."""
        if not self.memory:
            return
        past = await self.memory.search(prompt, limit=5)
        if past:
            context_str = "\n".join(f"- [{p.role}] {p.content}" for p in past)
            self.messages.append(
                Message(role="system", content=f"Relevant background memories:\n{context_str}")
            )

    async def _run_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute every requested tool call and append its result to the transcript."""
        for tc in tool_calls:
            await self.event_bus.publish("tool_start", {"tool": tc.name, "arguments": tc.arguments})
            started = time.perf_counter()
            output = await self.tools.execute(tc.name, **tc.arguments)
            output_str = output if isinstance(output, str) else str(output)
            await self.event_bus.publish("tool_end", {
                "tool": tc.name,
                "result": output_str,
                "seconds": time.perf_counter() - started,
            })
            self.messages.append(
                Message(role="tool", name=tc.name, tool_call_id=tc.id, content=output_str)
            )

    # ------------------------------------------------------------------- run

    async def run(self, prompt: str, **kwargs: Any) -> str:
        span = self.tracer.start_span(f"{self.name}.run", prompt=prompt, model=self.model)
        await self.event_bus.publish("agent_start", {"agent": self.name, "prompt": prompt, "model": self.model})

        await self._recall(prompt)
        self.messages.append(Message(role="user", content=prompt))

        final_reply = ""
        hit_limit = True
        for iteration in range(1, self.max_iterations + 1):
            await self.event_bus.publish("agent_iteration", {"iteration": iteration, "agent": self.name})

            response = await self.provider.generate(
                messages=self.messages,
                tools=self._tool_schemas(),
                temperature=self.temperature,
            )

            self.tracer.record_usage(
                span,
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

            # No tool calls means the model has reached its conclusion.
            if not response.tool_calls:
                final_reply = response.content
                self.messages.append(Message(role="assistant", content=final_reply))
                hit_limit = False
                break

            self.messages.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            await self._run_tool_calls(response.tool_calls)

        span.finish()
        await self.event_bus.publish("agent_end", {"agent": self.name, "reply": final_reply})

        if self.memory and final_reply:
            await self.memory.add(f"User: {prompt}\nAssistant: {final_reply}")

        if hit_limit and not final_reply:
            return (
                f"(Stopped after {self.max_iterations} tool-calling iterations without a final answer. "
                "Raise max_iterations or simplify the task.)"
            )
        return final_reply or "(No output generated)"

    # ---------------------------------------------------------------- stream

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream the answer, running any tools the model asks for along the way.

        Text deltas are yielded as they arrive. When the model asks for tools,
        they run, their results are appended to the transcript, and the loop
        continues streaming the next turn, so a streamed session is as capable
        as a non-streamed one.
        """
        span = self.tracer.start_span(f"{self.name}.stream", prompt=prompt, model=self.model)
        await self.event_bus.publish("agent_start", {"agent": self.name, "prompt": prompt, "model": self.model})

        await self._recall(prompt)
        self.messages.append(Message(role="user", content=prompt))

        for iteration in range(1, self.max_iterations + 1):
            await self.event_bus.publish("agent_iteration", {"iteration": iteration, "agent": self.name})

            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            async for chunk in self.provider.stream(
                messages=self.messages,
                tools=self._tool_schemas(),
                temperature=self.temperature,
            ):
                if chunk.text_delta:
                    text_parts.append(chunk.text_delta)
                    await self.event_bus.publish("token", {"delta": chunk.text_delta})
                if chunk.tool_call_delta is not None:
                    tool_calls.append(chunk.tool_call_delta)
                if chunk.usage:
                    self.tracer.record_usage(
                        span, self.model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                    )
                yield chunk

            reply_text = "".join(text_parts)
            # A provider may correct the model id mid-call (NVIDIA's bare-name lookup).
            self.model = self.provider.model or self.model

            if not tool_calls:
                if reply_text:
                    self.messages.append(Message(role="assistant", content=reply_text))
                break

            self.messages.append(
                Message(role="assistant", content=reply_text, tool_calls=tool_calls)
            )
            await self._run_tool_calls(tool_calls)

        span.finish()
        await self.event_bus.publish("agent_end", {"agent": self.name, "reply": ""})


class AssistantAgent(Agent):
    """Personalized assistant agent with customizable persona and role."""

    def __init__(self, name: str = "Assistant", persona: str = "Expert AI Consultant", **kwargs):
        system_prompt = kwargs.pop("system_prompt", f"You are {name}, a {persona}. Provide rigorous, concise, and helpful responses.")
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)
