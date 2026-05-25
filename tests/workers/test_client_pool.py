"""Tests for MCPClientPool — lazy connect, reuse, close_all."""
import pytest

from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.types import WorkerSpec


@pytest.mark.asyncio
async def test_pool_lazy_connects_and_reuses(streamable_http_fixture):
    """The pool doesn't connect until first call; second call reuses the same client."""
    spec = WorkerSpec(
        name="stub",
        endpoint=streamable_http_fixture,
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([spec])
    try:
        # First call triggers connect + initialize.
        tools = await pool.list_tools("stub")
        assert len(tools.tools) >= 2
        # Second call reuses the same connection — no error, returns same shape.
        tools2 = await pool.list_tools("stub")
        assert {t.name for t in tools.tools} == {t.name for t in tools2.tools}
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_pool_call_tool_via_pool(streamable_http_fixture):
    """call_tool through the pool round-trips arguments to the fixture."""
    spec = WorkerSpec(
        name="stub",
        endpoint=streamable_http_fixture,
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([spec])
    try:
        result = await pool.call_tool("stub", "noop_low", {"message": "via-pool"})
        text_blocks = [b for b in result.content if getattr(b, "type", None) == "text"]
        assert any("via-pool" in b.text for b in text_blocks)
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_pool_unknown_worker_raises(streamable_http_fixture):
    """Asking the pool for a worker name not in its specs raises KeyError."""
    spec = WorkerSpec(
        name="stub",
        endpoint=streamable_http_fixture,
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([spec])
    try:
        with pytest.raises(KeyError):
            await pool.list_tools("nonexistent")
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_pool_works_with_stdio_spec(stdio_fixture_spec):
    """MCPClientPool drives a stdio worker the same way as Streamable HTTP."""
    pool = MCPClientPool([stdio_fixture_spec])
    try:
        tools = await pool.list_tools("stdio_stub")
        names = {t.name for t in tools.tools}
        assert "noop_low" in names

        result = await pool.call_tool("stdio_stub", "risky_high", {"target": "via-stdio"})
        text_blocks = [b for b in result.content if getattr(b, "type", None) == "text"]
        assert any("via-stdio" in b.text for b in text_blocks)
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_pool_mixed_transports(streamable_http_fixture, stdio_fixture_spec):
    """A pool with both an HTTP worker and a stdio worker handles each correctly."""
    http_spec = WorkerSpec(
        name="http_stub",
        endpoint=streamable_http_fixture,
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([http_spec, stdio_fixture_spec])
    try:
        http_tools = await pool.list_tools("http_stub")
        stdio_tools = await pool.list_tools("stdio_stub")
        # Both workers expose the same toy surface in their fixtures.
        http_names = {t.name for t in http_tools.tools}
        stdio_names = {t.name for t in stdio_tools.tools}
        assert "noop_low" in http_names
        assert "noop_low" in stdio_names
    finally:
        await pool.close_all()
