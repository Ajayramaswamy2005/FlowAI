"""Tests for FlowAI Server."""

import pytest

from flowai.config.settings import Settings
from flowai.server.app import create_app

try:
    from starlette.testclient import TestClient
    HAS_TEST_CLIENT = True
except ImportError:
    HAS_TEST_CLIENT = False


@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="FastAPI / Starlette test client not available")
def test_server_routes():
    cfg = Settings(project_name="test_proj")
    app = create_app(cfg)
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # List tools
    tools_res = client.get("/tools")
    assert tools_res.status_code == 200
    assert "tools" in tools_res.json()

    # List agents
    agents_res = client.get("/agents")
    assert agents_res.status_code == 200
    assert len(agents_res.json()["agents"]) >= 1

    # Create agent
    new_agent_res = client.post("/agents", json={"name": "TestBot", "provider": "openai", "model": "gpt-4o"})
    assert new_agent_res.status_code == 200
