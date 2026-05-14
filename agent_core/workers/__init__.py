"""Worker contract: types, registry, risk gate, audit log, and
conformance fixtures for MCP-based workers in agent_core consumers.

The contract is intentionally framework-only — there is no live MCP
client in v1.2.0. Consuming agents (PARE in Phase 1+) provide the
transport; agent_core provides the data shapes and the verification
suite their workers must pass.
"""
