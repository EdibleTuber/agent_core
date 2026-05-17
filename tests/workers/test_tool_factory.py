"""Tests for the dynamic Tool factory."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.workers.tool_factory import make_tool_class
from agent_core.workers.types import WorkerSpec


def _spec(name: str = "stub") -> WorkerSpec:
    return WorkerSpec(
        name=name,
        endpoint="http://x.invalid/mcp",
        transport="streamable_http",
        risk_default="low",
    )


def _tool_def(name: str = "noop_low") -> dict:
    """Minimal MCP tool definition (matches mcp.types.Tool shape)."""
    return {
        "name": name,
        "description": "A toy tool.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": [],
        },
    }


def test_factory_produces_tool_subclass():
    """The factory returns a class that is-a Tool with the right metadata."""
    from agent_core.tools.base import Tool

    pool = MagicMock()
    cls = make_tool_class(_spec(), _tool_def(), pool)
    assert issubclass(cls, Tool)
    assert cls.name == "stub_noop_low"  # worker prefix
    assert cls.description == "A toy tool."
    assert cls.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_factory_run_calls_pool_call_tool():
    """The synthesized run() forwards to pool.call_tool."""
    pool = MagicMock()
    # The CallToolResult body the pool returns: one text block.
    fake_result = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"status": "ok"}'
    fake_result.content = [text_block]
    fake_result.isError = False  # explicit; the factory checks this
    pool.call_tool = AsyncMock(return_value=fake_result)

    cls = make_tool_class(_spec(), _tool_def(), pool)
    tool = cls()
    ctx = MagicMock()
    out = await tool.run({"message": "hello"}, ctx)

    pool.call_tool.assert_awaited_once_with("stub", "noop_low", {"message": "hello"})
    assert "ok" in out  # text block content surfaces in the return string


@pytest.mark.asyncio
async def test_factory_run_handles_error_result():
    """When the MCP result has isError=True, the tool returns an error string
    rather than the content as a normal value."""
    pool = MagicMock()
    fake_result = MagicMock()
    fake_result.isError = True
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "boom"
    fake_result.content = [text_block]
    pool.call_tool = AsyncMock(return_value=fake_result)

    cls = make_tool_class(_spec(), _tool_def(), pool)
    tool = cls()
    out = await tool.run({}, MagicMock())
    assert "error" in out.lower()
    assert "boom" in out


@pytest.mark.asyncio
async def test_factory_run_handles_call_exception():
    """When pool.call_tool raises (e.g., transport error), the tool returns
    a descriptive failure string (doesn't propagate)."""
    pool = MagicMock()
    pool.call_tool = AsyncMock(side_effect=RuntimeError("transport closed"))
    cls = make_tool_class(_spec(), _tool_def(), pool)
    tool = cls()
    out = await tool.run({}, MagicMock())
    assert "fail" in out.lower() or "error" in out.lower()
    assert "transport closed" in out
