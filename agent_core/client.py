"""Socket client for connecting to an agent_core daemon."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from agent_core.protocol import (
    STREAM_BUFFER_LIMIT,
    decode_message,
    encode_message,
)


class DaemonConnection:
    """Async unix-socket connection to an agent_core daemon.

    Use as `await conn.connect()`, then `await conn.send(msg)` and
    `async for msg in conn.receive()`. Call `await conn.close()` when done.
    """

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Open the unix socket connection."""
        self.reader, self.writer = await asyncio.open_unix_connection(
            path=str(self.socket_path), limit=STREAM_BUFFER_LIMIT,
        )

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
