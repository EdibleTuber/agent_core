"""discover_and_register — DEPRECATED boot-time worker discovery driver.

DEPRECATED since 1.8.0; use `WorkerManager.load` / `WorkerManager.load_autoload`
instead. This function predates the runtime worker lifecycle and duplicates a
strict subset of it: it has its own loop and its own fixed 2s timeout, and it
has NO per-worker locking, NO tool-name collision refusal, no WorkerOpResult,
no `last_error` for `/worker list`, and no generation bookkeeping (so nothing
evicts a session approval when a worker is replaced). It is retained only for
consumers that have not migrated.

`pool` must be a `RiskAwareToolPool`, not the inner `MCPClientPool`. Both expose
the same `list_tools`/`call_tool` signatures, but only the wrapper records the
per-tool wire tiers that later dispatches gate on, and only the wrapper gates
and audits dispatch at all — pass the inner pool and every synthesized tool
runs ungated with no wire tier ever recorded.

Iterates a list of WorkerSpec entries, connects each via the pool,
fetches list_tools, and produces Tool subclasses ready for an agent's
register_tools() to return.

A worker that fails to connect or list tools is logged loudly and
skipped — the agent still starts with whichever workers DID respond
("connection failures non-fatal, surfaced in /health"). WorkerManager
owns that guarantee now.
"""
from __future__ import annotations

import asyncio
import logging

from typing import TYPE_CHECKING

from agent_core.tools.base import Tool
from agent_core.workers.tool_factory import make_tool_class
from agent_core.workers.types import WorkerSpec

if TYPE_CHECKING:
    from agent_core.workers.risk_pool import RiskAwareToolPool

logger = logging.getLogger(__name__)


async def discover_and_register(
    specs: list[WorkerSpec],
    pool: "RiskAwareToolPool",
) -> list[type[Tool]]:
    """DEPRECATED — use `WorkerManager.load_autoload`. See the module docstring.

    Discover tools across all workers; return ready-to-register Tool classes.

    Args:
        specs: WorkerSpec entries from the agent's WorkerRegistry.
        pool: The RiskAwareToolPool that will back the synthesized Tools at
            call time. Passing the inner MCPClientPool disables the risk gate,
            approval and audit entirely, and records no wire tiers.

    Returns:
        List of Tool subclasses; empty if no workers responded. Caller passes
        this list to its agent.register_tools() return value (or extends it
        alongside any declarative tools).
    """
    tool_classes: list[type[Tool]] = []
    for spec in specs:
        try:
            list_result = await asyncio.wait_for(pool.list_tools(spec.name), timeout=2.0)
        except asyncio.CancelledError:
            # Never absorb a cancellation: continuing the loop inside a task
            # that is already cancelling makes every later await re-raise, which
            # presents as one dead worker killing its siblings' discovery.
            raise
        except Exception as exc:
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
