"""Conformance suite for agent_core worker contract.

Workers import `assert_conformance` and `MockWorkerContract` into their
own test packages. The MockWorkerContract is a reference implementation
of the contract surface — workers can copy its shape or stub out their
real implementation to satisfy it.

assert_conformance(worker) runs every required check against a worker
instance and raises AssertionError on the first failure with a clear
message.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_core.workers.types import (
    WORKER_CONTRACT_VERSION,
)


@runtime_checkable
class WorkerContract(Protocol):
    """The interface every worker must expose for conformance testing.

    Real workers (over MCP) translate these to `tools/list` and
    `tools/call` exchanges. The Protocol is the shape, not the wire."""

    def contract_version(self) -> int: ...
    def list_tools(self) -> list[dict[str, Any]]: ...


class MockWorkerContract:
    """Reference implementation. Exposes one example tool."""

    def __init__(self) -> None:
        self._version: int | None = WORKER_CONTRACT_VERSION
        self._tools: dict[str, dict[str, Any]] = {
            "noop": {
                "name": "noop",
                "risk_tier": "low",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": {"type": "object", "properties": {}},
            },
        }

    def contract_version(self) -> int | None:
        return self._version

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools.values())


_VALID_TIERS = {"low", "medium", "high", "critical"}


def assert_conformance(worker: WorkerContract) -> None:
    """Verify a worker exposes the contract correctly. Raises
    AssertionError with a clear message on first failure."""
    # Version present and integer-compatible.
    version = worker.contract_version()
    assert version is not None, "worker did not expose a contract version"
    assert isinstance(version, int), (
        f"contract version must be int, got {type(version).__name__}"
    )

    # Tool list is enumerable.
    tools = worker.list_tools()
    assert isinstance(tools, list), "list_tools must return a list"

    for tool in tools:
        # Required fields.
        assert "name" in tool, f"tool missing 'name': {tool!r}"
        assert "risk_tier" in tool, f"tool {tool['name']!r} missing 'risk_tier'"
        assert "input_schema" in tool, (
            f"tool {tool['name']!r} missing 'input_schema'"
        )
        assert "output_schema" in tool, (
            f"tool {tool['name']!r} missing 'output_schema'"
        )

        # Risk tier valid.
        tier = tool["risk_tier"]
        assert tier in _VALID_TIERS, (
            f"tool {tool['name']!r} has invalid risk_tier {tier!r}; "
            f"must be one of {sorted(_VALID_TIERS)}"
        )

        # Schemas are dict-shaped (JSON Schema sanity check).
        for key in ("input_schema", "output_schema"):
            assert isinstance(tool[key], dict), (
                f"tool {tool['name']!r} {key} must be a dict"
            )
            assert "type" in tool[key], (
                f"tool {tool['name']!r} {key} missing 'type' field"
            )
