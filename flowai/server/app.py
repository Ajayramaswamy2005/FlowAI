"""FastAPI REST and SSE Streaming API Server."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError:
    FastAPI = None

from flowai import __version__
from flowai.agents.standard import Agent
from flowai.config.settings import Settings
from flowai.errors import FlowAIError
from flowai.tools.builtin import default_system_prompt, default_toolset
from flowai.tools.registry import ToolRegistry


class ChatRequest(BaseModel):
    message: str
    agent: str = "default"
    provider: str | None = None
    model: str | None = None
    stream: bool = False


class AgentCreateRequest(BaseModel):
    name: str
    provider: str = "openai"
    model: str = "gpt-4o"
    system_prompt: str = "You are a helpful AI assistant."


def create_app(settings: Settings | None = None) -> Any:
    if FastAPI is None:
        raise ImportError("FastAPI is required to run the server. Install with: pip install 'flowai-cli[server]'")

    cfg = settings or Settings.load()
    app = FastAPI(title="FlowAI Agent Harness API", version=__version__)

    # Agents here can run shell commands, so only local pages may call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Active runtime registries
    tool_registry = ToolRegistry(default_toolset())
    agents_store: dict[str, Agent] = {
        "default": Agent(
            name="DefaultAgent",
            provider=cfg.provider,
            model=cfg.model,
            system_prompt=default_system_prompt(),
            tools=tool_registry,
        )
    }

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "project": cfg.project_name}

    @app.get("/tools")
    async def list_tools():
        return {"tools": tool_registry.get_schemas(format="openai")}

    @app.get("/agents")
    async def list_agents():
        return {
            "agents": [
                {
                    "name": a.name,
                    "model": a.model,
                    "provider": a.provider.__class__.__name__,
                    "system_prompt": a.system_prompt,
                }
                for a in agents_store.values()
            ]
        }

    @app.post("/agents")
    async def create_agent(req: AgentCreateRequest):
        agent = Agent(
            name=req.name,
            provider=req.provider,
            model=req.model,
            system_prompt=req.system_prompt,
            tools=tool_registry,
        )
        agents_store[req.name] = agent
        return {"message": f"Agent '{req.name}' created successfully."}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        agent = agents_store.get(req.agent)
        if not agent:
            # instantiate on demand
            agent = Agent(
                name=req.agent,
                provider=req.provider or cfg.provider,
                model=req.model or cfg.model,
                tools=tool_registry,
            )
            agents_store[req.agent] = agent

        if not req.stream:
            try:
                reply = await agent.run(req.message)
            except FlowAIError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"reply": reply, "agent": req.agent}

        async def sse_generator():
            try:
                async for chunk in agent.stream(req.message):
                    if chunk.text_delta:
                        yield f"data: {json.dumps({'token': chunk.text_delta})}\n\n"
            except FlowAIError as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    return app
