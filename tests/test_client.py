"""Tests for agent_core.client.DaemonConnection."""
import asyncio
from pathlib import Path

import pytest

from agent_core.client import DaemonConnection
from agent_core.protocol import (
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    ResponseMessage,
    StreamChunkMessage,
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


@pytest.mark.asyncio
async def test_is_connected_true_after_connect(tmp_path):
    socket_path = tmp_path / "conn.sock"

    async def fake_server(reader, writer):
        # Just hold open until client side closes; don't block on read.
        try:
            await reader.readline()
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        assert conn.is_connected is True
        await conn.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_is_connected_false_before_connect(tmp_path):
    conn = DaemonConnection(tmp_path / "nope.sock")
    assert conn.is_connected is False


@pytest.mark.asyncio
async def test_is_connected_false_after_close(tmp_path):
    socket_path = tmp_path / "close.sock"

    async def fake_server(reader, writer):
        try:
            await reader.readline()
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        await conn.close()
        assert conn.is_connected is False
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_chat_yields_until_response_message(tmp_path):
    """chat() yields stream chunks then breaks on ResponseMessage."""
    socket_path = tmp_path / "chat.sock"

    async def fake_server(reader, writer):
        # Read the request line.
        await reader.readline()
        writer.write(encode_message(StreamChunkMessage(token="hel")))
        writer.write(encode_message(StreamChunkMessage(token="lo")))
        writer.write(encode_message(ResponseMessage(text="hello")))
        # Send a trailing message that should NOT be yielded (loop already broke).
        writer.write(encode_message(StreamChunkMessage(token="trail")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        msgs = [m async for m in conn.chat("hi", channel_id="c1")]
        await conn.close()
        assert len(msgs) == 3
        assert isinstance(msgs[0], StreamChunkMessage)
        assert msgs[0].token == "hel"
        assert isinstance(msgs[1], StreamChunkMessage)
        assert msgs[1].token == "lo"
        assert isinstance(msgs[2], ResponseMessage)
        assert msgs[2].text == "hello"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_command_returns_final_response_message(tmp_path):
    """command() consumes intermediate messages and returns the ResponseMessage."""
    socket_path = tmp_path / "cmd.sock"
    received_request: list[bytes] = []

    async def fake_server(reader, writer):
        line = await reader.readline()
        received_request.append(line)
        writer.write(encode_message(StreamChunkMessage(token="working...")))
        writer.write(encode_message(ResponseMessage(text="done", command="ping")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        result = await conn.command("ping", "args", channel_id="c2")
        await conn.close()
        assert isinstance(result, ResponseMessage)
        assert result.text == "done"
        assert result.command == "ping"
        # Verify the request payload included the channel_id and args.
        assert b"\"name\": \"ping\"" in received_request[0] or b'"name":"ping"' in received_request[0]
        assert b"c2" in received_request[0]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_command_stream_yields_until_terminator(tmp_path):
    """command_stream() yields each message and stops at ResponseMessage."""
    socket_path = tmp_path / "cmdstream.sock"

    async def fake_server(reader, writer):
        await reader.readline()
        writer.write(encode_message(StreamChunkMessage(token="a")))
        writer.write(encode_message(StreamChunkMessage(token="b")))
        writer.write(encode_message(ResponseMessage(text="final", command="x")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        msgs = [m async for m in conn.command_stream("x")]
        await conn.close()
        assert len(msgs) == 3
        assert isinstance(msgs[-1], ResponseMessage)
        assert msgs[-1].text == "final"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_command_raises_on_error_message(tmp_path):
    """command() raises RuntimeError if the daemon sends an ErrorMessage."""
    socket_path = tmp_path / "cmderr.sock"

    async def fake_server(reader, writer):
        await reader.readline()
        writer.write(encode_message(ErrorMessage(error="boom")))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()
        with pytest.raises(RuntimeError, match="boom"):
            await conn.command("bad")
        await conn.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_chat_before_connect_raises(tmp_path):
    """chat() raises RuntimeError (not AssertionError) when not connected."""
    conn = DaemonConnection(tmp_path / "nope.sock")
    with pytest.raises(RuntimeError, match="Not connected"):
        async for _ in conn.chat("hi"):
            pass


@pytest.mark.asyncio
async def test_concurrent_receivers_serialized_by_lock(tmp_path):
    """Two concurrent chat() calls on one connection complete without crashing.

    The server handles each request in turn (single connection, serial reads).
    The client's _read_lock ensures the two coroutines do not interleave their
    reads of the daemon's response stream.
    """
    socket_path = tmp_path / "concurrent.sock"

    async def fake_server(reader, writer):
        # Handle two requests on the same connection. For each, send 2 messages.
        for i in range(2):
            line = await reader.readline()
            if not line:
                break
            writer.write(encode_message(StreamChunkMessage(token=f"chunk{i}")))
            writer.write(encode_message(ResponseMessage(text=f"resp{i}")))
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(fake_server, path=str(socket_path))
    try:
        conn = DaemonConnection(socket_path)
        await conn.connect()

        async def consume(text: str) -> list:
            return [m async for m in conn.chat(text)]

        # Launch two concurrently. The lock should serialize them.
        results = await asyncio.gather(consume("a"), consume("b"))
        await conn.close()

        # Each call should see exactly one chunk + one response.
        assert len(results) == 2
        for r in results:
            assert len(r) == 2
            assert isinstance(r[0], StreamChunkMessage)
            assert isinstance(r[1], ResponseMessage)
        # And the two response texts together cover the two server iterations.
        response_texts = sorted(r[1].text for r in results)
        assert response_texts == ["resp0", "resp1"]
    finally:
        server.close()
        await server.wait_closed()
