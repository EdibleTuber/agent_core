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
