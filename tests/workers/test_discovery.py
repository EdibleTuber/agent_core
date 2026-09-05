"""Tests for discover_and_register — top-level worker discovery driver."""
import pytest

from agent_core.workers.discovery import discover_and_register
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.types import WorkerSpec


@pytest.mark.asyncio
async def test_discover_returns_tool_classes_from_live_fixture(streamable_http_fixture):
    """discover_and_register against the FastMCP fixture returns Tool subclasses
    named after the fixture's two stub tools."""
    spec = WorkerSpec(
        name="stub",
        endpoint=streamable_http_fixture,
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([spec])
    try:
        tool_classes = await discover_and_register([spec], pool)
        names = {cls.name for cls in tool_classes}
        assert "stub_noop_low" in names
        assert "stub_risky_high" in names
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_discover_skips_unreachable_workers(caplog):
    """A worker whose endpoint refuses connection is logged and skipped, not raised."""
    import logging
    caplog.set_level(logging.WARNING, logger="agent_core.workers.discovery")

    bad_spec = WorkerSpec(
        name="bogus",
        endpoint="http://127.0.0.1:1/mcp",  # nothing listens here
        transport="streamable_http",
        risk_default="low",
    )
    pool = MCPClientPool([bad_spec])
    try:
        tool_classes = await discover_and_register([bad_spec], pool)
        assert tool_classes == []  # nothing registered
        assert any("bogus" in rec.message for rec in caplog.records)
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_discover_against_stdio_fixture(stdio_fixture_spec):
    """discover_and_register returns prefixed Tool subclasses for a stdio
    worker, same as for an HTTP worker."""
    pool = MCPClientPool([stdio_fixture_spec])
    try:
        tool_classes = await discover_and_register([stdio_fixture_spec], pool)
        names = {cls.name for cls in tool_classes}
        assert "stdio_stub_noop_low" in names
        assert "stdio_stub_risky_high" in names
    finally:
        await pool.close_all()


@pytest.mark.asyncio
async def test_discovery_does_not_swallow_cancellation():
    """`except (TimeoutError, CancelledError, Exception)` absorbed a cancellation
    aimed at the discovery task and marched on to the next worker inside a task
    that was already cancelling -- the best available explanation for the
    'one dead worker killed its siblings' report in PARE's workers.yaml."""
    import asyncio
    from agent_core.workers.discovery import discover_and_register
    from agent_core.workers.types import WorkerSpec

    class _Pool:
        async def list_tools(self, worker):
            raise asyncio.CancelledError()

    spec = WorkerSpec(name="w", transport="stdio", risk_default="low",
                      command="/bin/true")
    with pytest.raises(asyncio.CancelledError):
        await discover_and_register([spec], _Pool())
