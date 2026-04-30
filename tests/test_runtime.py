"""Tests for agent_core.runtime.run_daemon."""
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_core.agent import Agent
from agent_core.config import BaseConfig
from agent_core.runtime import run_daemon


class _StubAgent(Agent):
    name = "test"
    setup_was_called: bool = False
    setup_saw_managers: dict = {}

    def setup(self):
        type(self).setup_was_called = True
        type(self).setup_saw_managers = {
            "config": self.config,
            "profile": self.profile,
            "wisdom": self.wisdom,
            "channels": self.channels,
            "inference": self.inference,
        }


def test_run_daemon_populates_managers_then_calls_setup(monkeypatch, tmp_path):
    """run_daemon wires every framework manager before invoking setup."""
    monkeypatch.setenv("TEST_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("TEST_SOCKET_PATH", str(tmp_path / "test.sock"))

    # Reset class-level state between tests.
    _StubAgent.setup_was_called = False
    _StubAgent.setup_saw_managers = {}

    # Stub out asyncio.run so the daemon doesn't actually serve.
    with patch("agent_core.runtime.asyncio.run") as mock_run:
        agent = _StubAgent()
        run_daemon(agent)
        mock_run.assert_called_once()

    assert _StubAgent.setup_was_called
    seen = _StubAgent.setup_saw_managers
    assert seen["config"] is not None
    assert seen["profile"] is not None
    assert seen["wisdom"] is not None
    assert seen["channels"] is not None
    assert seen["inference"] is not None


def test_run_daemon_uses_subclassed_config(monkeypatch, tmp_path):
    """run_daemon accepts a subclass of BaseConfig and reads its extra fields."""
    from dataclasses import dataclass

    @dataclass
    class MyConfig(BaseConfig):
        my_extra: str = "default"

    monkeypatch.setenv("TEST_MY_EXTRA", "from-env")
    monkeypatch.setenv("TEST_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("TEST_SOCKET_PATH", str(tmp_path / "test.sock"))

    with patch("agent_core.runtime.asyncio.run"):
        agent = _StubAgent()
        run_daemon(agent, config_cls=MyConfig)

    assert agent.config.my_extra == "from-env"
