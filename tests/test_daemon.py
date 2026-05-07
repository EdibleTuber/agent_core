"""Tests for agent_core.daemon.Daemon."""
import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from agent_core.agent import Agent, HandlerContext
from agent_core.channels import ChannelStore
from agent_core.config import BaseConfig
from agent_core.daemon import Daemon, resolve_channel_id
from agent_core.protocol import (
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    ResponseMessage,
    decode_message,
    encode_message,
)


def test_resolve_channel_id_default_for_none():
    assert resolve_channel_id(None) == "cli-default"


def test_resolve_channel_id_default_for_empty():
    assert resolve_channel_id("") == "cli-default"


def test_resolve_channel_id_passes_valid():
    assert resolve_channel_id("C1") == "C1"


def test_resolve_channel_id_falls_back_for_invalid():
    assert resolve_channel_id("../etc/passwd") == "cli-default"


class _StubAgent(Agent):
    """Minimal Agent that records dispatched messages."""
    name = "test"

    def __init__(self):
        self.chat_msgs: list = []
        self.command_msgs: list = []

    async def handle_chat(self, msg, ctx) -> AsyncIterator[object]:
        self.chat_msgs.append((msg, ctx.channel_id))
        yield ResponseMessage(text=f"handled: {msg.text}")

    async def handle_command(self, msg, ctx) -> AsyncIterator[object]:
        self.command_msgs.append((msg, ctx.channel_id))
        yield ResponseMessage(text=f"cmd: {msg.name}")


def _wire_minimal_agent(tmp_path: Path) -> _StubAgent:
    """Construct a stub agent with the minimum framework attrs the daemon needs."""
    agent = _StubAgent()
    cfg = BaseConfig()
    cfg.vault_path = tmp_path
    cfg.socket_path = tmp_path / "test.sock"
    cfg.history_depth = 50
    agent.config = cfg
    agent.channels = ChannelStore(
        vault_path=tmp_path, agent_name="test", history_depth=50,
    )
    return agent


@pytest.mark.asyncio
async def test_daemon_dispatches_chat(tmp_path):
    agent = _wire_minimal_agent(tmp_path)
    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(ChatMessage(text="hello", channel_id="C1")))
        await writer.drain()

        line = await reader.readline()
        msg = decode_message(line.rstrip(b"\n"))
        assert isinstance(msg, ResponseMessage)
        assert msg.text == "handled: hello"

        writer.close()
        await writer.wait_closed()
        # Allow handler task to record before assertion.
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()

    assert len(agent.chat_msgs) == 1
    assert agent.chat_msgs[0][1] == "C1"


@pytest.mark.asyncio
async def test_daemon_dispatches_command(tmp_path):
    agent = _wire_minimal_agent(tmp_path)
    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(CommandMessage(
            name="help", args="", channel_id="C1",
        )))
        await writer.drain()

        line = await reader.readline()
        msg = decode_message(line.rstrip(b"\n"))
        assert isinstance(msg, ResponseMessage)
        assert msg.text == "cmd: help"

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()

    assert len(agent.command_msgs) == 1


@pytest.mark.asyncio
async def test_daemon_dispatches_unknown_message_to_handle_other(tmp_path):
    """Non-Chat/non-Command messages route to agent.handle_other."""
    from dataclasses import dataclass

    from agent_core.protocol import register_message

    @dataclass
    class CustomApprovalMessage:
        proposal_id: str
        choice: str
        type: str = "custom_approval_test"

    register_message(CustomApprovalMessage)

    agent = _wire_minimal_agent(tmp_path)
    received: list = []

    async def custom_handle_other(msg, ctx):
        received.append((msg, ctx.channel_id))

    agent.handle_other = custom_handle_other  # type: ignore[assignment]

    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(CustomApprovalMessage(
            proposal_id="abc", choice="approve",
        )))
        await writer.drain()

        # No response expected from the daemon for handle_other (synchronous, no yield).
        # Allow handler to run.
        await asyncio.sleep(0.05)

        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()

    assert len(received) == 1
    assert received[0][0].proposal_id == "abc"


@pytest.mark.asyncio
async def test_daemon_emits_error_on_decode_failure(tmp_path):
    agent = _wire_minimal_agent(tmp_path)
    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(b"not-json-at-all\n")
        await writer.drain()

        line = await reader.readline()
        msg = decode_message(line.rstrip(b"\n"))
        assert isinstance(msg, ErrorMessage)
        assert "decode failed" in msg.error

        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Task 18: ctx.agent and ctx.emit population
# ---------------------------------------------------------------------------

class _ProbeAgentCtx(Agent):
    """Agent that captures ctx.agent for the populate-ctx-agent test."""
    name = "probe-agent-ctx"
    env_prefix = "PROBE_AGENT_CTX"

    def __init__(self):
        self.captured: dict = {}

    async def handle_chat(self, msg, ctx) -> AsyncIterator[object]:
        self.captured["agent_is_self"] = ctx.agent is self
        yield ResponseMessage(text="probe-agent-ctx-ok")

    async def handle_command(self, msg, ctx) -> AsyncIterator[object]:
        yield ResponseMessage(text="noop")

    def system_prompt(self, ctx):
        return "p"


class _ProbeEmit(Agent):
    """Agent that calls ctx.emit() from inside handle_chat."""
    name = "probe-emit"
    env_prefix = "PROBE_EMIT"

    def __init__(self):
        self.emit_raised: list[str] = []

    async def handle_chat(self, msg, ctx) -> AsyncIterator[object]:
        assert callable(ctx.emit)
        await ctx.emit(ResponseMessage(text="from-emit"))
        yield ResponseMessage(text="probe-emit-done")

    async def handle_command(self, msg, ctx) -> AsyncIterator[object]:
        yield ResponseMessage(text="noop")

    def system_prompt(self, ctx):
        return "p"


def _wire_probe_agent(tmp_path: Path, agent: Agent) -> Agent:
    """Wire framework attrs onto any probe agent instance."""
    cfg = BaseConfig()
    cfg.vault_path = tmp_path
    cfg.socket_path = tmp_path / "probe.sock"
    cfg.history_depth = 50
    agent.config = cfg
    agent.channels = ChannelStore(
        vault_path=tmp_path, agent_name=agent.name, history_depth=50,
    )
    return agent


@pytest.mark.asyncio
async def test_daemon_populates_ctx_agent(tmp_path):
    """HandlerContext.agent is the daemon's agent instance (verified by is-self check)."""
    agent = _wire_probe_agent(tmp_path, _ProbeAgentCtx())
    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(ChatMessage(text="hi", channel_id="C1")))
        await writer.drain()

        line = await reader.readline()
        resp = decode_message(line.rstrip(b"\n"))
        assert isinstance(resp, ResponseMessage)
        assert resp.text == "probe-agent-ctx-ok"

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()

    assert agent.captured.get("agent_is_self") is True


@pytest.mark.asyncio
async def test_daemon_populates_ctx_emit(tmp_path):
    """ctx.emit writes an NDJSON-encoded message to the connection."""
    agent = _wire_probe_agent(tmp_path, _ProbeEmit())
    daemon = Daemon(agent)
    server = await asyncio.start_unix_server(
        daemon._handle_connection, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(ChatMessage(text="hi", channel_id="C1")))
        await writer.drain()

        # Two responses: one from ctx.emit("from-emit"), one from the yield.
        line1 = await reader.readline()
        line2 = await reader.readline()

        msg1 = decode_message(line1.rstrip(b"\n"))
        msg2 = decode_message(line2.rstrip(b"\n"))

        texts = {msg1.text, msg2.text}
        assert "from-emit" in texts
        assert "probe-emit-done" in texts

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_daemon_emit_triggers_writer_drain(tmp_path):
    """ctx.emit() calls writer.drain() — backpressure is honored.

    We wrap the real writer's drain method to count calls, then verify that
    at least one drain call came from ctx.emit (i.e., drain count > 0 after
    the emit-only message arrives).
    """
    agent = _wire_probe_agent(tmp_path, _ProbeEmit())
    drain_calls: list[int] = []

    original_handle = agent  # captured for server start

    daemon = Daemon(agent)

    # Patch asyncio.StreamWriter.drain by intercepting at the daemon level.
    # The simplest approach: wrap _handle_connection to instrument the writer.
    original_handle_conn = daemon._handle_connection

    async def _instrumented(reader, writer):
        original_drain = writer.drain

        async def counting_drain():
            drain_calls.append(1)
            return await original_drain()

        writer.drain = counting_drain
        await original_handle_conn(reader, writer)

    server = await asyncio.start_unix_server(
        _instrumented, path=str(agent.config.socket_path),
    )
    try:
        reader, writer = await asyncio.open_unix_connection(
            path=str(agent.config.socket_path),
        )
        writer.write(encode_message(ChatMessage(text="hi", channel_id="C1")))
        await writer.drain()

        # Drain both response lines.
        await reader.readline()
        await reader.readline()

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()

    # At minimum: 1 drain from ctx.emit + 1 drain from the yield response.
    assert len(drain_calls) >= 2
