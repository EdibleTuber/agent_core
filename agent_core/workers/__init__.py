"""Worker contract: types, registry, risk gate, audit log, and
conformance fixtures for MCP-based workers in agent_core consumers.

The contract is intentionally framework-only — there is no live MCP
client in v1.2.0. Consuming agents (PARE in Phase 1+) provide the
transport; agent_core provides the data shapes and the verification
suite their workers must pass.
"""

from agent_core.workers.client import MCPClient
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.discovery import discover_and_register
from agent_core.workers.tool_factory import make_tool_class

__all__ = [
    "MCPClient",
    "MCPClientPool",
    "discover_and_register",
    "make_tool_class",
]
