"""MCPClient: thin async wrapper over the official mcp SDK's
Streamable HTTP transport.

Lifecycle:
    client = MCPClient(endpoint="http://host:port/mcp")
    await client.connect()
    await client.initialize()          # MCP handshake (Task 3)
    tools = await client.list_tools()  # tools/list (Task 4)
    result = await client.call_tool(name, arguments)  # tools/call (Task 5)
    await client.close()

The wrapper exposes the methods Phase 2's discovery driver needs.
MCP error objects are returned unchanged — translation into agent_core
error semantics happens at the call site (tool_factory.py, Task 7).
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    """One MCP client connection to one worker endpoint."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._session: ClientSession | None = None
        self._transport_ctx: Any = None  # held to keep the streams open

    async def connect(self) -> None:
        """Open the Streamable HTTP transport and wrap it in a ClientSession.

        The transport context manager is held on the instance; close() releases it."""
        self._transport_ctx = streamablehttp_client(self.endpoint)
        read_stream, write_stream, *_ = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None

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
