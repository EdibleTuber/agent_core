"""Minimal FastMCP server used as a live Streamable HTTP fixture for
agent_core's MCP client + discovery tests.

Exposes two toy tools:
    noop_low   — returns "ok" with the input echoed.
    risky_high — returns "did the thing" with the target echoed.

Both echo input args back for assertion convenience.
"""
from __future__ import annotations

from fastmcp import FastMCP


def build_stub() -> FastMCP:
    """Construct a fresh FastMCP instance with two toy tools registered."""
    mcp = FastMCP("agent-core-stub-worker")

    @mcp.tool()
    def noop_low(message: str = "hi") -> dict:
        """A toy tool that returns ok with the input echoed."""
        return {"status": "ok", "echo": message}

    @mcp.tool()
    def risky_high(target: str) -> dict:
        """A toy tool that pretends to do something risky."""
        return {"status": "did the thing", "target": target}

    return mcp
