"""Socket client for connecting to an agent_core daemon."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from agent_core.protocol import (
    STREAM_BUFFER_LIMIT,
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    ResponseMessage,
    decode_message,
    encode_message,
)


class DaemonConnection:
    """Async unix-socket connection to an agent_core daemon.

    Low-level usage: `await conn.connect()`, then `await conn.send(msg)` and
    `async for msg in conn.receive()`. Call `await conn.close()` when done.

    High-level helpers (`chat`, `command`, `command_stream`) wrap a request
    plus its streaming response in a single call, serialized by an internal
    asyncio.Lock so concurrent callers do not interleave reads on the same
    connection.
    """

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._read_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the unix socket connection."""
        self.reader, self.writer = await asyncio.open_unix_connection(
            path=str(self.socket_path), limit=STREAM_BUFFER_LIMIT,
        )

    @property
    def is_connected(self) -> bool:
        """True if the writer is open and not closing."""
        return self.writer is not None and not self.writer.is_closing()

    async def send(self, msg: object) -> None:
        """Send one message (NDJSON-encoded) to the daemon."""
        assert self.writer is not None, "connect() before send()"
        self.writer.write(encode_message(msg))
        await self.writer.drain()

    async def receive(self) -> AsyncIterator[object]:
        """Yield messages from the daemon until the connection closes."""
        assert self.reader is not None, "connect() before receive()"
        while not self.reader.at_eof():
            line = await self.reader.readline()
            if not line:
                break
            yield decode_message(line.rstrip(b"\n"))

    async def close(self) -> None:
        """Close the writer half cleanly. Safe to call multiple times."""
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def chat(
        self, text: str, *, channel_id: str | None = None,
    ) -> AsyncIterator[object]:
        """Send a ChatMessage and yield streaming chunks until terminator.

        Yields every message as it arrives, breaking after a ResponseMessage
        or ErrorMessage. Acquires the read lock so concurrent chat/command
        calls on the same connection do not interleave.
        """
        if self.writer is None or self.reader is None:
            raise RuntimeError("Not connected")
        async with self._read_lock:
            self.writer.write(
                encode_message(ChatMessage(text=text, channel_id=channel_id))
            )
            await self.writer.drain()
            while True:
                line = await self.reader.readline()
                if not line:
                    break
                decoded = decode_message(line.rstrip(b"\n"))
                yield decoded
                if isinstance(decoded, (ResponseMessage, ErrorMessage)):
                    break

    async def command(
        self, name: str, args: str = "", *, channel_id: str | None = None,
    ) -> ResponseMessage:
        """Send a CommandMessage and return the final ResponseMessage.

        Raises RuntimeError on ErrorMessage or ConnectionError on EOF.
        """
        if self.writer is None or self.reader is None:
            raise RuntimeError("Not connected")
        async with self._read_lock:
            self.writer.write(
                encode_message(
                    CommandMessage(name=name, args=args, channel_id=channel_id)
                )
            )
            await self.writer.drain()
            while True:
                line = await self.reader.readline()
                if not line:
                    raise ConnectionError("Connection closed")
                decoded = decode_message(line.rstrip(b"\n"))
                if isinstance(decoded, ResponseMessage):
                    return decoded
                if isinstance(decoded, ErrorMessage):
                    raise RuntimeError(decoded.error)

    async def command_stream(
        self, name: str, args: str = "", *, channel_id: str | None = None,
    ) -> AsyncIterator[object]:
        """Send a CommandMessage and yield streaming chunks until terminator."""
        if self.writer is None or self.reader is None:
            raise RuntimeError("Not connected")
        async with self._read_lock:
            self.writer.write(
                encode_message(
                    CommandMessage(name=name, args=args, channel_id=channel_id)
                )
            )
            await self.writer.drain()
            while True:
                line = await self.reader.readline()
                if not line:
                    break
                decoded = decode_message(line.rstrip(b"\n"))
                yield decoded
                if isinstance(decoded, (ResponseMessage, ErrorMessage)):
                    break
