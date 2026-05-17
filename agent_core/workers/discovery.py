"""discover_and_register — top-level worker discovery driver.

Iterates a list of WorkerSpec entries, connects each via the pool,
fetches list_tools, and produces Tool subclasses ready for an agent's
register_tools() to return.

A worker that fails to connect or list tools is logged loudly and
skipped — the agent still starts with whichever workers DID respond.
This is the framework-side enforcement of the spec's "connection
failures non-fatal, surfaced in /health" guarantee.
"""
from __future__ import annotations

import asyncio
import logging

from agent_core.tools.base import Tool
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.tool_factory import make_tool_class
from agent_core.workers.types import WorkerSpec

logger = logging.getLogger(__name__)


async def discover_and_register(
    specs: list[WorkerSpec],
    pool: MCPClientPool,
) -> list[type[Tool]]:
    """Discover tools across all workers; return ready-to-register Tool classes.

    Args:
        specs: WorkerSpec entries from the agent's WorkerRegistry.
        pool: The MCPClientPool that will back the synthesized Tools at call time.

    Returns:
        List of Tool subclasses; empty if no workers responded. Caller passes
        this list to its agent.register_tools() return value (or extends it
        alongside any declarative tools).
    """
    tool_classes: list[type[Tool]] = []
    for spec in specs:
        try:
            list_result = await asyncio.wait_for(pool.list_tools(spec.name), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as exc:
            logger.warning(
                "worker %s discovery failed (%s); skipping registration",
                spec.name,
                exc,
            )
            continue

        for tool in getattr(list_result, "tools", []):
            tool_def = {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "inputSchema": getattr(tool, "inputSchema", None)
                or {"type": "object", "properties": {}},
            }
            cls = make_tool_class(spec, tool_def, pool)
            tool_classes.append(cls)
            logger.info("registered tool %s from worker %s", cls.name, spec.name)
    return tool_classes
