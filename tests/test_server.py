"""The HTTP server: it works for a client with the token, and for no one else."""

import pytest

from flowai.config.settings import Settings
from flowai.server.app import create_app

try:
    from starlette.testclient import TestClient
    HAS_TEST_CLIENT = True
except ImportError:
    HAS_TEST_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TEST_CLIENT, reason="FastAPI / Starlette test client not available")

TOKEN = "test-token-0123456789"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWAI_HOME", str(tmp_path / ".flowai"))
    monkeypatch.setenv("FLOWAI_KEYRING", "0")
    app = create_app(Settings(project_name="test_proj"), token=TOKEN)
    return TestClient(app, base_url="http://localhost")


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_routes_work_with_the_token(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "tools" in client.get("/tools", headers=auth()).json()
    assert len(client.get("/agents", headers=auth()).json()["agents"]) >= 1
    res = client.post("/agents", headers=auth(), json={"name": "TestBot", "provider": "openai", "model": "gpt-4o"})
    assert res.status_code == 200


def test_requests_without_the_right_token_are_refused(client):
    assert client.get("/tools").status_code == 401
    assert client.get("/tools", headers=auth("wrong")).status_code == 401
    assert client.post("/chat", json={"message": "hi"}).status_code == 401


def test_dns_rebinding_hosts_are_refused(client):
    # A malicious page that rebinds its own domain to 127.0.0.1 still sends its Host header.
    res = client.get("/health", headers={"Host": "evil.example.com"})
    assert res.status_code == 403


def test_unknown_provider_is_a_client_error(client):
    res = client.post("/agents", headers=auth(), json={"name": "x", "provider": "nope"})
    assert res.status_code == 400


def test_a_server_without_a_token_refuses_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOWAI_HOME", str(tmp_path / ".flowai"))
    c = TestClient(create_app(Settings(), token=None), base_url="http://localhost")
    assert c.get("/tools").status_code == 503
