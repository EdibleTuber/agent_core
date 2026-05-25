"""Stdio conformance suite — runs against a live MCP worker over stdio."""
import pytest

from agent_core.workers.conformance import assert_stdio_conformance
from agent_core.workers.types import WorkerSpec


@pytest.mark.asyncio
async def test_stdio_fixture_passes_conformance(stdio_fixture_spec):
    """The stdio fixture spec satisfies the live-transport conformance checks."""
    await assert_stdio_conformance(stdio_fixture_spec)


@pytest.mark.asyncio
async def test_nonexistent_command_fails_conformance():
    """A bogus command (file doesn't exist) fails the conformance check (raises)."""
    bad_spec = WorkerSpec(
        name="bogus",
        transport="stdio",
        risk_default="low",
        command="/nonexistent/binary",
        args=[],
    )
    with pytest.raises(AssertionError):
        await assert_stdio_conformance(bad_spec)
