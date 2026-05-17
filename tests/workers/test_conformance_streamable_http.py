"""Streamable HTTP conformance suite — runs against a live MCP worker."""
import pytest

from agent_core.workers.conformance import assert_streamable_http_conformance


@pytest.mark.asyncio
async def test_streamable_http_fixture_passes_conformance(streamable_http_fixture):
    """The FastMCP fixture satisfies the live-transport conformance checks."""
    await assert_streamable_http_conformance(streamable_http_fixture)


@pytest.mark.asyncio
async def test_unreachable_endpoint_fails_conformance():
    """A nonexistent endpoint fails the conformance checks (raises AssertionError)."""
    with pytest.raises(AssertionError):
        await assert_streamable_http_conformance("http://127.0.0.1:1/mcp")
