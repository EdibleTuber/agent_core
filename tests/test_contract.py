"""Contract tests: pinning the agent_core public API surface.

These tests verify the API guarantee. They live in agent_core forever.
A change that breaks one of these tests is by definition a breaking change.
"""
import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from agent_core.agent import Agent, HandlerContext
from agent_core.channels import ChannelStore
from agent_core.config import BaseConfig
from agent_core.daemon import Daemon
from agent_core.protocol import ChatMessage, ResponseMessage, decode_message, encode_message


class _MinimalAgent(Agent):
    """One-line agent: implements the minimum required to handle a chat."""
    name = "minimal"

    async def handle_chat(self, msg, ctx) -> AsyncIterator[object]:
        yield ResponseMessage(text=f"got: {msg.text}")


@pytest.mark.asyncio
async def test_minimal_agent_boots(tmp_path):
    """A trivial Agent subclass with manually-wired channels boots and serves."""
    agent = _MinimalAgent()
    cfg = BaseConfig()
    cfg.vault_path = tmp_path
    cfg.socket_path = tmp_path / "minimal.sock"
    agent.config = cfg
    agent.channels = ChannelStore(
        vault_path=tmp_path, agent_name="minimal", history_depth=10,
    )

    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(cfg.socket_path),
    )
    try:
        # Verify the socket file exists and is reachable.
        reader, writer = await asyncio.open_unix_connection(path=str(cfg.socket_path))
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_agent_receives_chat(tmp_path):
    """Sending a ChatMessage to the daemon causes Agent.handle_chat to run."""
    agent = _MinimalAgent()
    cfg = BaseConfig()
    cfg.vault_path = tmp_path
    cfg.socket_path = tmp_path / "minimal.sock"
    agent.config = cfg
    agent.channels = ChannelStore(
        vault_path=tmp_path, agent_name="minimal", history_depth=10,
    )

    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(cfg.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(cfg.socket_path))
        writer.write(encode_message(ChatMessage(text="hello", channel_id="C1")))
        await writer.drain()
        line = await reader.readline()
        msg = decode_message(line.rstrip(b"\n"))
        assert isinstance(msg, ResponseMessage)
        assert msg.text == "got: hello"
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_agent_handle_chat_yields_responses(tmp_path):
    """A handler yielding multiple messages results in multiple responses."""
    class MultiResponseAgent(Agent):
        name = "multi"

        async def handle_chat(self, msg, ctx) -> AsyncIterator[object]:
            yield ResponseMessage(text="first")
            yield ResponseMessage(text="second")

    agent = MultiResponseAgent()
    cfg = BaseConfig()
    cfg.vault_path = tmp_path
    cfg.socket_path = tmp_path / "multi.sock"
    agent.config = cfg
    agent.channels = ChannelStore(
        vault_path=tmp_path, agent_name="multi", history_depth=10,
    )

    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(cfg.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(cfg.socket_path))
        writer.write(encode_message(ChatMessage(text="ping")))
        await writer.drain()

        responses = []
        for _ in range(2):
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            responses.append(decode_message(line.rstrip(b"\n")))
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert [r.text for r in responses] == ["first", "second"]


# ---------------------------------------------------------------------------
# Task 19: tool/command registration API contracts
# ---------------------------------------------------------------------------

def test_minimal_agent_with_tools_registers_them():
    """An agent that lists `tools = [...]` gets a populated tool_executor
    after _attach_registries runs (the same wiring run_daemon performs).
    The agent's named tool is reachable, alongside the 12 builtins."""
    from unittest.mock import MagicMock

    from agent_core.agent import Agent
    from agent_core.tools.base import Tool
    from agent_core.runtime import _attach_registries

    class _Probe(Tool):
        name = "contract_probe"
        description = "Contract test probe"
        parameters = {}
        async def run(self, args, ctx): return "ok"

    class _A(Agent):
        name = "contract-a"
        tools = [_Probe]

    a = _A()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())

    _attach_registries(a)

    names = a.tool_executor.names()
    assert "contract_probe" in names      # agent-supplied
    assert "cat" in names                  # builtin
    assert "fetch_url" in names            # framework-backed builtin


def test_disabled_builtins_excludes_from_executor():
    """disabled_builtins removes named tools from the executor (works for
    builtin and agent-supplied alike)."""
    from unittest.mock import MagicMock

    from agent_core.agent import Agent
    from agent_core.runtime import _attach_registries

    class _A(Agent):
        name = "contract-disabled-tools"
        disabled_builtins = frozenset({"grep"})

    a = _A()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())

    _attach_registries(a)

    assert "grep" not in a.tool_executor.names()
    assert "cat" in a.tool_executor.names()


def test_disabled_builtins_excludes_from_commands():
    """disabled_builtins removes named commands from the registry."""
    from unittest.mock import MagicMock

    from agent_core.agent import Agent
    from agent_core.runtime import _attach_registries

    class _A(Agent):
        name = "contract-disabled-cmds"
        disabled_builtins = frozenset({"quit"})

    a = _A()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())

    _attach_registries(a)

    assert "quit" not in a.command_registry.names()
    assert "help" in a.command_registry.names()


def test_missing_dep_fails_at_attach_not_at_runtime():
    """If a tool requires an attribute the agent doesn't have, the failure
    surfaces inside _attach_registries — before agent.setup() runs, before
    any user message is processed."""
    import pytest
    from unittest.mock import MagicMock

    from agent_core.agent import Agent
    from agent_core.tools.base import Tool
    from agent_core.runtime import _attach_registries

    class _NeedsXyz(Tool):
        name = "needs_xyz"
        description = ""
        parameters = {}
        requires = ("xyz",)
        async def run(self, args, ctx): return ""

    class _A(Agent):
        name = "contract-missing-dep"
        tools = [_NeedsXyz]

    a = _A()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())
    # Deliberately do NOT set a.xyz

    with pytest.raises(RuntimeError, match="needs_xyz.*xyz"):
        _attach_registries(a)


@pytest.mark.asyncio
async def test_handler_context_carries_agent_and_emit_through_dispatch():
    """A tool dispatched via tool_executor sees ctx.agent and ctx.emit at
    runtime. Constructs HandlerContext manually (no full daemon harness needed)
    since the contract is from the tool's perspective, not the transport's."""
    from unittest.mock import AsyncMock, MagicMock

    from agent_core.agent import Agent, HandlerContext
    from agent_core.tools.base import Tool
    from agent_core.runtime import _attach_registries
    from agent_core.conversation import Conversation

    captured: dict = {}

    class _SeesContext(Tool):
        name = "sees_ctx"
        description = ""
        parameters = {}
        async def run(self, args, ctx):
            captured["agent"] = ctx.agent
            captured["emit_callable"] = callable(ctx.emit)
            return "ok"

    class _A(Agent):
        name = "contract-ctx"
        tools = [_SeesContext]

    a = _A()
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(a, attr, MagicMock())

    _attach_registries(a)

    emit_fn = AsyncMock()
    ctx = HandlerContext(
        conversation=Conversation(history_depth=10),
        channel_id="C1",
        writer=MagicMock(),
        agent=a,
        emit=emit_fn,
    )

    result = await a.tool_executor.run("sees_ctx", {}, ctx)

    assert result == "ok"
    assert captured["agent"] is a
    assert captured["emit_callable"] is True
