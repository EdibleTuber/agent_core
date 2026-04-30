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
