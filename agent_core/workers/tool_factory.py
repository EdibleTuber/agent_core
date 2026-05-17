"""Dynamic Tool subclass factory.

For each MCP tool definition discovered from a worker, produce an
agent_core.tools.base.Tool subclass whose run(args, ctx) calls back
into the pool. The class is name-prefixed by worker (`{worker}_{tool}`)
to avoid cross-worker collisions.
"""
from __future__ import annotations

from typing import Any

from agent_core.tools.base import Tool
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.types import WorkerSpec


def _stringify_result(result: Any) -> str:
    """Flatten a CallToolResult's content blocks into a single string for the LLM."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        else:
            parts.append(repr(block))
    return "\n".join(parts)


def make_tool_class(
    worker: WorkerSpec,
    tool_def: dict,
    pool: MCPClientPool,
) -> type[Tool]:
    """Produce a Tool subclass that calls the given worker's tool via the pool.

    Args:
        worker: WorkerSpec for the worker (provides name + endpoint).
        tool_def: One MCP tool definition (dict with name, description, inputSchema).
        pool: The shared MCPClientPool the synthesized Tool calls into.

    Returns:
        A new Tool subclass. Caller registers it via agent.register_tools().
    """
    tool_name = tool_def["name"]
    prefixed = f"{worker.name}_{tool_name}"
    description_str = tool_def.get("description", "") or ""
    parameters_dict = tool_def.get("inputSchema") or {"type": "object", "properties": {}}

    # Capture by closure — these go into the class body via attribute assignment
    # rather than ClassVar so the factory closure semantics stay clear.
    class _DynamicTool(Tool):
        pass

    _DynamicTool.name = prefixed
    _DynamicTool.description = description_str
    _DynamicTool.parameters = parameters_dict
    _DynamicTool.__name__ = f"DynamicTool_{prefixed}"
    _DynamicTool.__qualname__ = f"DynamicTool_{prefixed}"

    async def _run(self, args: dict[str, Any], ctx: Any) -> str:
        try:
            result = await pool.call_tool(worker.name, tool_name, args)
        except Exception as exc:
            return f"{prefixed} call failed: {exc}"
        if getattr(result, "isError", False):
            return f"{prefixed} returned an error: {_stringify_result(result)}"
        return _stringify_result(result) or f"{prefixed} returned no content"

    _DynamicTool.run = _run
    return _DynamicTool
