"""Standard Agent and AssistantAgent implementations."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from flowai.agents.base import BaseAgent
from flowai.plugins.hooks import ErrorContext, PromptContext, ResponseContext
from flowai.providers.base import BaseProvider, Message, StreamChunk, ToolCall
from flowai.security.secrets import redact

INTERRUPTED_TOOL_RESULT = "(This call did not finish: the turn was interrupted before it returned a result.)"


class Agent(BaseAgent):
    """Core autonomous Agent capable of reasoning, tool execution, and memory recall."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.messages: list[Message] = [
            Message(role="system", content=self.system_prompt)
        ]
        self._call_counts: dict[str, int] = {}

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

    def _repair_transcript(self) -> None:
        """Give every tool call in the transcript a result.

        A turn interrupted while a tool ran (Ctrl-C, a crash) leaves an
        assistant message whose tool calls never got answers, and providers
        reject every later request in that conversation. The missing results
        are filled in so the session carries on.
        """
        repaired: list[Message] = []
        pending: list[ToolCall] = []

        def close_pending() -> None:
            for tc in pending:
                repaired.append(Message(role="tool", name=tc.name, tool_call_id=tc.id,
                                        content=INTERRUPTED_TOOL_RESULT))
            pending.clear()

        for m in self.messages:
            if m.role == "tool":
                pending[:] = [tc for tc in pending if tc.id != m.tool_call_id]
            else:
                close_pending()
            repaired.append(m)
            if m.role == "assistant" and m.tool_calls:
                pending = list(m.tool_calls)
        close_pending()
        if len(repaired) != len(self.messages):
            self.messages = repaired

    def _recent_prompts(self) -> list[str]:
        return [m.content or "" for m in self.messages if m.role == "user"][-4:]

    async def _begin_turn(self, prompt: str) -> tuple[str | None, list[str], Message | None]:
        """Run pre_prompt hooks and the guardrails on a prompt.

        Returns (the prompt to send, or None when the turn is refused; notices
        for the user; a reminder message to drop once the turn ends). Hooks run
        first, so the guardrails judge the prompt as a plugin left it.
        """
        self._repair_transcript()
        self._call_counts = {}
        ctx = await self.hooks.run("pre_prompt", PromptContext(agent=self.name, prompt=prompt))
        if ctx.blocked:
            return None, [f"Stopped by a FlowAI plugin: {ctx.reason}"], None
        prompt = ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt)
        if self.guardrails is None:
            return prompt, [], None

        verdict = self.guardrails.check_prompt(prompt, history=self._recent_prompts())
        if verdict.jailbreak:
            await self.event_bus.publish("guardrail", {
                "agent": self.name, "reasons": verdict.jailbreak, "blocked": verdict.blocked,
            })
        if verdict.blocked:
            return None, [*verdict.notices, verdict.message], None
        reminder = Message(role="system", content=verdict.reminder) if verdict.reminder else None
        return verdict.prompt, verdict.notices, reminder

    def _end_turn(self, reminder: Message | None) -> None:
        # The reminder was about this prompt only. Left in, providers that fold system
        # messages into one system prompt would keep repeating it on every later turn.
        if reminder is not None:
            self.messages = [m for m in self.messages if m is not reminder]

    async def _finish_reply(self, prompt: str, reply: str) -> str:
        ctx = await self.hooks.run("post_response", ResponseContext(agent=self.name, prompt=prompt, reply=reply))
        if isinstance(ctx.reply, str):
            reply = ctx.reply
        return self.guardrails.clean_output(reply) if self.guardrails else reply

    def _as_flowai_error(self, exc: Exception) -> Exception:
        """A network failure from any provider (plugins' included) becomes one clear ProviderError."""
        if isinstance(exc, httpx.HTTPError):
            return self.provider._transport_error(exc)
        return exc

    async def _report_error(self, exc: BaseException) -> None:
        await self.hooks.run("on_error", ErrorContext(agent=self.name, error=exc))
        await self.event_bus.publish("error", {"agent": self.name, "error": redact(f"{type(exc).__name__}: {exc}")})

    def _repeat_limit_message(self, tc: ToolCall) -> str | None:
        """Stops a model stuck calling the same tool with the same arguments over and over."""
        limit = self.guardrails.max_identical_tool_calls if self.guardrails else 0
        if not limit:
            return None
        key = tc.name + json.dumps(tc.arguments, sort_keys=True, default=str)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        if self._call_counts[key] <= limit:
            return None
        return (
            f"Not run: you already called {tc.name} with exactly these arguments {limit} times this turn. "
            "Repeating it will not give a different result. Try another approach, or tell the user what is blocking you."
        )

    async def _run_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute every requested tool call and append its result to the transcript."""
        for tc in tool_calls:
            await self.event_bus.publish("tool_start", {"tool": tc.name, "arguments": tc.arguments})
            started = time.perf_counter()
            output = self._repeat_limit_message(tc)
            if output is None:
                arguments = tc.arguments if isinstance(tc.arguments, dict) else {"value": tc.arguments}
                output = await self.tools.execute(tc.name, **arguments)
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

        original = prompt
        prompt, notices, reminder = await self._begin_turn(prompt)
        for notice in notices:
            await self.event_bus.publish("notice", {"agent": self.name, "message": notice})
        if prompt is None:
            span.finish()
            await self.event_bus.publish("agent_end", {"agent": self.name, "reply": ""})
            return notices[-1]

        final_reply = ""
        hit_limit = True
        try:
            await self._recall(prompt)
            self.messages.append(Message(role="user", content=prompt))
            if reminder:
                self.messages.append(reminder)

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
                    final_reply = await self._finish_reply(original, response.content)
                    self.messages.append(Message(role="assistant", content=final_reply))
                    hit_limit = False
                    break

                self.messages.append(
                    Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
                )
                await self._run_tool_calls(response.tool_calls)
        except Exception as exc:
            failure = self._as_flowai_error(exc)
            await self._report_error(failure)
            if failure is exc:
                raise
            raise failure from exc
        finally:
            self._end_turn(reminder)
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

        Text deltas are yielded as they arrive, redacted, holding back only the
        word in progress so a key cannot slip out split across two chunks.
        When the model asks for tools, they run, their results are appended to
        the transcript, and the loop continues streaming the next turn, so a
        streamed session is as capable as a non-streamed one. post_response
        hooks see the answer after it was streamed: here they can change what
        is kept in the transcript, not what was shown.
        """
        span = self.tracer.start_span(f"{self.name}.stream", prompt=prompt, model=self.model)
        await self.event_bus.publish("agent_start", {"agent": self.name, "prompt": prompt, "model": self.model})

        original = prompt
        prompt, notices, reminder = await self._begin_turn(prompt)
        for notice in notices:
            yield StreamChunk(notice=notice)
        if prompt is None:
            span.finish()
            await self.event_bus.publish("agent_end", {"agent": self.name, "reply": ""})
            return

        try:
            await self._recall(prompt)
            self.messages.append(Message(role="user", content=prompt))
            if reminder:
                self.messages.append(reminder)

            for iteration in range(1, self.max_iterations + 1):
                await self.event_bus.publish("agent_iteration", {"iteration": iteration, "agent": self.name})

                text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                text_guard = self.guardrails.stream_redactor() if self.guardrails else None
                thought_guard = self.guardrails.stream_redactor() if self.guardrails else None

                async for chunk in self.provider.stream(
                    messages=self.messages,
                    tools=self._tool_schemas(),
                    temperature=self.temperature,
                ):
                    if chunk.text_delta:
                        text_parts.append(chunk.text_delta)
                    if chunk.tool_call_delta is not None:
                        tool_calls.append(chunk.tool_call_delta)
                    if chunk.usage:
                        self.tracer.record_usage(
                            span, self.model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                        )
                    if text_guard and (chunk.text_delta or chunk.reasoning_delta):
                        chunk = chunk.model_copy(update={
                            "text_delta": text_guard.feed(chunk.text_delta),
                            "reasoning_delta": thought_guard.feed(chunk.reasoning_delta),
                        })
                        if not (chunk.text_delta or chunk.reasoning_delta or chunk.tool_call_delta
                                or chunk.notice or chunk.usage or chunk.finish_reason):
                            continue  # everything in it is held back for now
                    if chunk.text_delta:
                        await self.event_bus.publish("token", {"delta": chunk.text_delta})
                    yield chunk
                if text_guard:
                    rest, rest_thought = text_guard.flush(), thought_guard.flush()
                    if rest or rest_thought:
                        if rest:
                            await self.event_bus.publish("token", {"delta": rest})
                        yield StreamChunk(text_delta=rest, reasoning_delta=rest_thought)

                reply_text = "".join(text_parts)
                # A provider may correct the model id mid-call (NVIDIA's bare-name lookup).
                self.model = self.provider.model or self.model

                if not tool_calls:
                    if reply_text:
                        reply_text = await self._finish_reply(original, reply_text)
                        self.messages.append(Message(role="assistant", content=reply_text))
                    break

                self.messages.append(
                    Message(role="assistant", content=reply_text, tool_calls=tool_calls)
                )
                await self._run_tool_calls(tool_calls)
        except Exception as exc:
            failure = self._as_flowai_error(exc)
            await self._report_error(failure)
            if failure is exc:
                raise
            raise failure from exc
        finally:
            self._end_turn(reminder)
            span.finish()

        await self.event_bus.publish("agent_end", {"agent": self.name, "reply": ""})


class AssistantAgent(Agent):
    """Personalized assistant agent with customizable persona and role."""

    def __init__(self, name: str = "Assistant", persona: str = "Expert AI Consultant", **kwargs):
        system_prompt = kwargs.pop("system_prompt", f"You are {name}, a {persona}. Provide rigorous, concise, and helpful responses.")
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)
