"""Tests for agent_core.workers.client.MCPClient against a live FastMCP fixture."""
import pytest

from agent_core.workers.client import MCPClient


@pytest.mark.asyncio
async def test_initialize_completes_against_live_fixture(streamable_http_fixture):
    """initialize() returns the server's InitializeResult without raising."""
    client = MCPClient(streamable_http_fixture)
    try:
        await client.connect()
        result = await client.initialize()
        # MCP InitializeResult exposes serverInfo.name; the stub names itself.
        assert result is not None
        # Be permissive about the attribute name — MCP SDK may use snake_case
        # or camelCase depending on version.
        info = getattr(result, "server_info", None) or getattr(result, "serverInfo", None)
        assert info is not None, f"InitializeResult has neither server_info nor serverInfo: {result!r}"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_tools_returns_stub_tools(streamable_http_fixture):
    """list_tools() returns the two tools the FastMCP stub registered."""
    client = MCPClient(streamable_http_fixture)
    try:
        await client.connect()
        await client.initialize()
        tools = await client.list_tools()
        # mcp.types.ListToolsResult.tools is a list of Tool objects.
        names = {t.name for t in tools.tools}
        assert "noop_low" in names
        assert "risky_high" in names
    finally:
        await client.close()
