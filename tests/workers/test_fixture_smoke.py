"""Sanity check that the streamable_http_fixture starts and serves."""
import pytest
import httpx


@pytest.mark.asyncio
async def test_fixture_serves_http(streamable_http_fixture):
    """The fixture URL responds to a basic HTTP request (any status; just verifies
    the server is alive and bound)."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        # MCP Streamable HTTP responds to POST/GET on the /mcp path; any
        # response (even a 405 or 400) means the server is alive.
        resp = await client.get(streamable_http_fixture)
        assert resp.status_code < 500
