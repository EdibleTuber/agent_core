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


# ---------------------------------------------------------------------------
# Task 17: URLFetcher wiring and _attach_registries
# ---------------------------------------------------------------------------

def test_run_daemon_wires_fetcher(monkeypatch, tmp_path):
    """run_daemon constructs URLFetcher onto agent.fetcher with config-driven settings."""
    from agent_core.utils.fetcher import URLFetcher

    captured = {}

    class _ProbeAgent(Agent):
        name = "probe"

        def setup(self):
            captured["fetcher"] = self.fetcher
            captured["max_bytes"] = self.fetcher.max_bytes
            captured["timeout"] = self.fetcher.timeout

    monkeypatch.setenv("PROBE_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("PROBE_SOCKET_PATH", str(tmp_path / "probe.sock"))

    with patch("agent_core.runtime.asyncio.run"):
        run_daemon(_ProbeAgent())

    assert isinstance(captured["fetcher"], URLFetcher)
    assert captured["max_bytes"] == BaseConfig().fetch_max_bytes   # default 2_000_000
    assert captured["timeout"] == BaseConfig().fetch_timeout        # default 30


def test_run_daemon_attaches_registries_after_setup(monkeypatch, tmp_path):
    """After v0.6.1, _attach_registries runs after agent.setup() so tool/command
    `requires` declarations can validate against domain managers constructed in
    setup. Setup itself does NOT see the registries — they are attached immediately
    after setup returns. The registries ARE present on the agent after run_daemon."""
    from agent_core.tools.base import Tool

    class _ProbeTool(Tool):
        name = "probe_tool"
        description = "a probe"
        parameters = {"type": "object", "properties": {}}

        async def run(self, args, ctx):
            return "ok"

    captured = {}

    class _ProbeAgent(Agent):
        name = "probeat"
        tools = [_ProbeTool]

        def setup(self):
            # At setup() time, registries have NOT been attached yet.
            captured["setup_saw_executor"] = hasattr(self, "tool_executor")
            captured["setup_saw_registry"] = hasattr(self, "command_registry")
            captured["setup_saw_prompt_builder"] = hasattr(self, "prompt_builder")

    monkeypatch.setenv("PROBEAT_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("PROBEAT_SOCKET_PATH", str(tmp_path / "probeat.sock"))

    with patch("agent_core.runtime.asyncio.run"):
        agent = _ProbeAgent()
        run_daemon(agent)

    # setup() ran before _attach_registries, so it saw none of them.
    assert captured["setup_saw_executor"] is False
    assert captured["setup_saw_registry"] is False
    assert captured["setup_saw_prompt_builder"] is False

    # After run_daemon returns, all three are attached and the tool is registered.
    assert hasattr(agent, "tool_executor")
    assert hasattr(agent, "command_registry")
    assert hasattr(agent, "prompt_builder")
    assert "probe_tool" in agent.tool_executor.names()


def test_attach_registries_validates_domain_managers_set_in_setup(monkeypatch, tmp_path):
    """A tool requiring a domain manager constructed in setup() now validates
    cleanly. Before v0.6.1 this would have raised RuntimeError at registration
    because _attach_registries ran before setup() had a chance to set the attr."""
    from unittest.mock import MagicMock

    from agent_core.tools.base import Tool

    class _NeedsDomain(Tool):
        name = "needs_domain"
        description = ""
        parameters = {}
        requires = ("compiler",)  # constructed in setup, not by run_daemon

        async def run(self, args, ctx):
            return "ok"

    captured = {}

    class _DomainAgent(Agent):
        name = "domain-agent"
        tools = [_NeedsDomain]

        def setup(self):
            self.compiler = MagicMock()
            captured["setup_set_compiler"] = True

    monkeypatch.setenv("DOMAIN_AGENT_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("DOMAIN_AGENT_SOCKET_PATH", str(tmp_path / "domain-agent.sock"))

    with patch("agent_core.runtime.asyncio.run"):
        agent = _DomainAgent()
        # Should not raise — _attach_registries now sees self.compiler.
        run_daemon(agent)

    assert captured["setup_set_compiler"] is True
    assert "needs_domain" in agent.tool_executor.names()


def test_attach_registries_fails_when_required_attr_missing():
    """If a tool requires an attr the agent doesn't have, _attach_registries
    raises before setup runs."""
    from unittest.mock import MagicMock

    from agent_core.tools.base import Tool
    from agent_core.runtime import _attach_registries

    class _NeedsXyz(Tool):
        name = "needs_xyz"
        description = ""
        parameters = {}
        requires = ("xyz",)

        async def run(self, args, ctx):
            return ""

    class _BadAgent(Agent):
        name = "bad-agent"
        tools = [_NeedsXyz]

    a = _BadAgent()
    # Stub framework managers _attach_registries reads during ToolExecutor.build
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())
    # Deliberately do NOT set a.xyz

    with pytest.raises(RuntimeError, match="needs_xyz"):
        _attach_registries(a)


def test_attach_registries_includes_builtins():
    """A minimal agent gets all 12 builtin tools and 12 builtin commands via
    _attach_registries."""
    from unittest.mock import MagicMock

    from agent_core.runtime import _attach_registries

    class _Empty(Agent):
        name = "empty"

    a = _Empty()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())

    _attach_registries(a)

    tool_names = a.tool_executor.names()
    assert len(tool_names) == 12
    assert "cat" in tool_names
    assert "fetch_url" in tool_names

    cmd_names = a.command_registry.names()
    assert len(cmd_names) == 12
    assert "help" in cmd_names
    assert "quit" in cmd_names

    assert hasattr(a, "prompt_builder")
