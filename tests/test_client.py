"""Tests for agent_core.client.DaemonConnection."""
import asyncio
from pathlib import Path

import pytest

from agent_core.client import DaemonConnection
from agent_core.protocol import (
    ChatMessage,
    ResponseMessage,
    encode_message,
)


@pytest.mark.asyncio
async def test_connection_round_trip(tmp_path):
    """Connect to a fake unix server, send a chat, receive a response."""
    socket_path = tmp_path / "test.sock"

    async def fake_server(reader, writer):
        line = await reader.readline()
        # Echo back a ResponseMessage.
        writer.write(encode_message(ResponseMessage(text="echo")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        await conn.send(ChatMessage(text="hi"))

        responses = []
        async for msg in conn.receive():
            responses.append(msg)
        await conn.close()

        assert len(responses) == 1
        assert isinstance(responses[0], ResponseMessage)
        assert responses[0].text == "echo"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connection_send_before_connect_raises(tmp_path):
    conn = DaemonConnection(tmp_path / "missing.sock")
    with pytest.raises(AssertionError):
        await conn.send(ChatMessage(text="hi"))


@pytest.mark.asyncio
async def test_connection_receive_streams_multiple(tmp_path):
    """A server sending N messages results in N items from receive()."""
    socket_path = tmp_path / "multi.sock"

    async def fake_server(reader, writer):
        for i in range(3):
            writer.write(encode_message(ResponseMessage(text=f"msg{i}")))
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        msgs = [m async for m in conn.receive()]
        await conn.close()
        assert [m.text for m in msgs] == ["msg0", "msg1", "msg2"]
    finally:
        server.close()
        await server.wait_closed()
