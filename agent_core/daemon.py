"""Generic agent daemon: unix socket server with NDJSON message protocol.

Transport-only. Connection lifecycle, message decode, dispatch to agent
handlers, message encode, disconnect cleanup. The agent owns chat and command
logic; the daemon does not.
"""
from __future__ import annotations

import asyncio
import logging

from agent_core.agent import Agent, HandlerContext
from agent_core.channels import validate_channel_id
from agent_core.protocol import (
    STREAM_BUFFER_LIMIT,
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    decode_message,
    encode_message,
)

logger = logging.getLogger(__name__)


def resolve_channel_id(raw: str | None, default: str = "cli-default") -> str:
    """Validate channel_id, falling back to a default if missing or invalid."""
    if not raw:
        return default
    if not validate_channel_id(raw):
        logger.warning(
            "invalid channel_id %r received; falling back to %s", raw, default,
        )
        return default
    return raw


class Daemon:
    """Transport-only daemon. Owns the socket and dispatches to an Agent."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        # Reserved for the deferred per-channel preemption safety fix; not
        # used by Phase E.
        self._chat_tasks: dict[str, asyncio.Task] = {}

    async def serve(self) -> None:
        """Bind the socket and accept connections forever."""
        socket_path = self.agent.config.socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(socket_path),
            limit=STREAM_BUFFER_LIMIT,
        )
        logger.info("agent %s listening on %s", self.agent.name, socket_path)
        async with server:
            await server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Per-connection loop: read NDJSON, dispatch, write responses."""
        owned_tasks: list[asyncio.Task] = []
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = decode_message(line.rstrip(b"\n"))
                except Exception as exc:
                    err = ErrorMessage(error=f"decode failed: {exc}")
                    writer.write(encode_message(err))
                    await writer.drain()
                    continue

                channel_id = resolve_channel_id(getattr(msg, "channel_id", None))
                conv = await self.agent.channels.get_or_create(channel_id)
                ctx = HandlerContext(
                    conversation=conv, channel_id=channel_id, writer=writer,
                )

                if isinstance(msg, ChatMessage):
                    task = asyncio.create_task(
                        self._run_handler(self.agent.handle_chat, msg, ctx, writer),
                    )
                    owned_tasks.append(task)
                elif isinstance(msg, CommandMessage):
                    task = asyncio.create_task(
                        self._run_handler(self.agent.handle_command, msg, ctx, writer),
                    )
                    owned_tasks.append(task)
                else:
                    # Synchronous dispatch for agent-specific message types
                    # (approval responses, batch fallback choices, etc.).
                    # The agent's default no-op makes unknown messages silent;
                    # PAL overrides handle_other to route approval messages.
                    try:
                        await self.agent.handle_other(msg, ctx)
                    except Exception as exc:
                        logger.exception("handle_other failed: %s", exc)

        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception as exc:
            logger.exception("connection handler error: %s", exc)
        finally:
            for t in owned_tasks:
                if not t.done():
                    t.cancel()
            for t in owned_tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _run_handler(self, handler, msg, ctx: HandlerContext, writer) -> None:
        """Invoke an Agent handler, encode each yielded message, write to socket."""
        try:
            async for response in handler(msg, ctx):
                writer.write(encode_message(response))
                await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("handler error: %s", exc)
            try:
                err = ErrorMessage(error=f"{type(exc).__name__}: {exc}")
                writer.write(encode_message(err))
                await writer.drain()
            except Exception:
                pass
