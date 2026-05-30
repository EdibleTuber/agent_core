"""Minimal FastMCP server over stdio for agent_core's stdio MCP client tests.

Exposes the same two toy tools as the Streamable HTTP fixture for
parity:
    noop_low   — returns ok with the input echoed.
    risky_high — returns "did the thing" with the target echoed.

Launched by stdio_client as a subprocess; communicates over its
stdin/stdout per the MCP stdio transport spec.
"""
from __future__ import annotations

from fastmcp import FastMCP


def build_stub() -> FastMCP:
    mcp = FastMCP("agent-core-stdio-stub-worker")

    @mcp.tool(meta={"agent_core/risk_tier": "low"})
    def noop_low(message: str = "hi") -> dict:
        """A toy tool that returns ok with the input echoed."""
        return {"status": "ok", "echo": message}

    @mcp.tool(meta={"agent_core/risk_tier": "high"})
    def risky_high(target: str) -> dict:
        """A toy tool that pretends to do something risky."""
        return {"status": "did the thing", "target": target}

    return mcp


if __name__ == "__main__":
    build_stub().run(transport="stdio")
