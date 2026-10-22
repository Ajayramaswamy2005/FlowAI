"""Every test runs against a throwaway FlowAI home and never the real keyring."""

import pytest


@pytest.fixture(autouse=True)
def isolated_flowai_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("FLOWAI_HOME", str(tmp_path_factory.mktemp("flowai-home")))
    monkeypatch.setenv("FLOWAI_KEYRING", "0")
