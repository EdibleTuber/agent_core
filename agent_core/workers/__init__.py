"""Worker contract and runtime: types, registry, risk gate, audit log,
conformance fixtures, and the live MCP client stack for MCP-based workers.

agent_core owns the transport (`MCPClient`, `MCPClientPool`), the enforcement
wrapper (`RiskAwareToolPool`) and the runtime lifecycle (`WorkerManager`, which
loads and unloads workers while the daemon runs); consuming agents supply the
`workers.yaml` catalog and the operator surface. The conformance suite still
defines what a worker must satisfy.
"""

from agent_core.workers.client import MCPClient
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.discovery import discover_and_register
from agent_core.workers.manager import (
    WorkerManager, WorkerOpResult, WorkerStatus,
)
from agent_core.workers.registry import WorkerRegistry
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_factory import make_tool_class

__all__ = [
    "MCPClient",
    "MCPClientPool",
    "RiskAwareToolPool",
    "WorkerManager",
    "WorkerOpResult",
    "WorkerRegistry",
    "WorkerStatus",
    "discover_and_register",
    "make_tool_class",
]
