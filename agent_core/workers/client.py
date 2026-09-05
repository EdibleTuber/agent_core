"""MCPClient: thin async wrapper over the official mcp SDK.

Supports two transports — Streamable HTTP (existing) and stdio (new).

Lifecycle (Streamable HTTP):
    client = MCPClient(endpoint="http://host:port/mcp")
    await client.connect()
    await client.initialize()
    tools = await client.list_tools()
    result = await client.call_tool(name, arguments)
    await client.close()

Lifecycle (stdio):
    client = MCPClient(command="python", args=["-m", "my_mcp_server"])
    await client.connect()
    await client.initialize()
    tools = await client.list_tools()
    result = await client.call_tool(name, arguments)
    await client.close()

The wrapper exposes the methods Phase 2's discovery driver needs.
MCP error objects are returned unchanged — translation into agent_core
error semantics happens at the call site (tool_factory.py, Task 7).
"""
from __future__ import annotations

import contextlib
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.workers.types import WorkerSpec

import httpx
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

logger = logging.getLogger(__name__)

DEFAULT_HTTP_CONNECT_TIMEOUT = 30.0
"""httpx connect/write/pool bound. Matches the SDK's own default so an
unconfigured worker behaves exactly as it did before these fields existed."""

DEFAULT_HTTP_READ_TIMEOUT = 300.0
"""httpx read bound, also the SDK's default. Five minutes is far too long to
wait on a tool call over a tailnet, which is why WorkerSpec.read_timeout
exists -- but changing the default silently would change every existing
deployment, so the override is opt-in."""


class MCPClient:
    """One MCP client connection to one worker endpoint.

    Accepts either a Streamable HTTP endpoint or a stdio command, not both.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        if not endpoint and not command:
            raise ValueError(
                "MCPClient requires either endpoint (streamable_http) or "
                "command (stdio)"
            )
        if endpoint and command:
            raise ValueError(
                "MCPClient cannot accept both endpoint and command — choose one transport"
            )
        self.endpoint = endpoint
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._transport: str = "stdio" if command else "streamable_http"
        self._session: ClientSession | None = None
        self._transport_ctx: object | None = None
        self._http_client: httpx.AsyncClient | None = None

    @classmethod
    def from_spec(cls, spec: "WorkerSpec") -> "MCPClient":
        """Construct an MCPClient from a WorkerSpec, dispatching transport.

        WorkerSpec.transport determines which constructor path to use:
            stdio           → cls(command=..., args=..., env=..., cwd=...)
            streamable_http → cls(endpoint=...)
        """
        if spec.transport == "stdio":
            return cls(
                command=spec.command,
                args=list(spec.args),
                env=dict(spec.env) if spec.env else None,
                cwd=spec.cwd,
                connect_timeout=spec.connect_timeout,
                read_timeout=spec.read_timeout,
            )
        return cls(
            endpoint=spec.endpoint,
            connect_timeout=spec.connect_timeout,
            read_timeout=spec.read_timeout,
        )

    async def connect(self) -> None:
        """Open the configured transport and wrap it in a ClientSession.

        Every scope this enters is released again if a later stage fails --
        see _release(). A half-entered connect that keeps its transport is how
        an unreachable HTTP endpoint used to leak an httpx.AsyncClient per
        attempt, finalised much later by the GC in an arbitrary task.
        """
        if self._transport == "stdio":
            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env or None,
                cwd=self.cwd,
            )
            self._transport_ctx = stdio_client(params)
        else:
            # Two bounds, not one. httpx.Timeout's positional argument covers
            # connect/write/pool; the RESPONSE read is governed separately and
            # defaults to 300s, so a single field cannot bound a slow worker.
            self._http_client = create_mcp_http_client(
                timeout=httpx.Timeout(
                    self.connect_timeout
                    if self.connect_timeout is not None
                    else DEFAULT_HTTP_CONNECT_TIMEOUT,
                    read=self.read_timeout
                    if self.read_timeout is not None
                    else DEFAULT_HTTP_READ_TIMEOUT,
                )
            )
            # Passing http_client makes US its owner: streamable_http_client
            # only manages the lifecycle of a client it created itself.
            self._transport_ctx = streamable_http_client(
                self.endpoint, http_client=self._http_client
            )

        try:
            read_stream, write_stream, *_ = await self._transport_ctx.__aenter__()
            session = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=(
                    timedelta(seconds=self.read_timeout)
                    if self.read_timeout is not None
                    else None
                ),
            )
            await session.__aenter__()
        except BaseException:
            # Release in this task, now, while we are still inside the task
            # that entered the scopes. Secondary failures are suppressed so the
            # original cause is what the caller sees.
            with contextlib.suppress(Exception):
                await self._release()
            raise
        # Published only once fully entered: a session whose __aenter__ raised
        # must never reach _release(), which would __aexit__ a scope that was
        # never entered.
        self._session = session

    async def _release(self) -> None:
        """Unwind session, transport and http client -- every stage runs.

        Chained finally rather than sequential awaits: if the session's
        __aexit__ raises (routine for an HTTP endpoint that was never
        reachable, since the client is lazy and the failure surfaces inside
        initialize()), the transport and the httpx client must still be
        closed. The first exception propagates; the rest still get cleaned up.
        """
        session, self._session = self._session, None
        ctx, self._transport_ctx = self._transport_ctx, None
        http, self._http_client = self._http_client, None
        try:
            if session is not None:
                await session.__aexit__(None, None, None)
        finally:
            try:
                if ctx is not None:
                    await ctx.__aexit__(None, None, None)
            finally:
                if http is not None:
                    await http.aclose()

    async def close(self) -> None:
        await self._release()

    async def initialize(self):
        """Send the MCP initialize request. Returns the server's InitializeResult."""
        assert self._session is not None, "call connect() before initialize()"
        return await self._session.initialize()

    async def list_tools(self):
        """Send the MCP tools/list request. Returns ListToolsResult."""
        assert self._session is not None, "call connect() before list_tools()"
        return await self._session.list_tools()

    async def call_tool(self, name: str, arguments: dict | None = None):
        """Send the MCP tools/call request. Returns CallToolResult.

        Raises mcp.McpError (or the SDK's equivalent) on protocol errors;
        the caller decides how to map those to agent_core error semantics.
        """
        assert self._session is not None, "call connect() before call_tool()"
        return await self._session.call_tool(name, arguments or {})
