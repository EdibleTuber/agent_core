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

import contextlib
import os

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agent_core.workers.artifacts import PRODUCES_META_KEY, VALID_PRODUCES
from agent_core.workers.client_pool import describe_failure
from agent_core.workers.risk import RISK_TIER_META_KEY

CONFORMANCE_TIMEOUT = float(os.environ.get("AGENT_CORE_CONFORMANCE_TIMEOUT", "30"))
"""Bound on each conformance step, in seconds.

This exists so a wedged server FAILS instead of hanging a suite forever. It is
not a performance assertion, and it should not be read as one: the previous
value was 2.0s, which on a developer box has a ~400x margin (a local
initialize measures ~0.005s) but is a live failure risk on a shared CI runner
that is doing other work. A bound tight enough to double as a benchmark will
eventually fail for reasons that have nothing to do with conformance.

Overridable so a genuinely slow target (a worker on a Pi across a tailnet)
can be checked without editing the library.
"""
from agent_core.workers.types import (
    WORKER_CONTRACT_VERSION,
)

if TYPE_CHECKING:
    from agent_core.workers.types import WorkerSpec


_WIRE_VALID_TIERS = {"low", "medium", "high", "critical"}


def _assert_valid_risk_tier_meta(tool: Any) -> None:
    """Assert a live tool advertises a valid risk_tier in its MCP `_meta`.

    Shared by the stdio and streamable_http live-conformance suites so the
    wire contract is enforced identically on both transports. A worker that
    fails to advertise a valid per-tool tier is rejected at build/test time,
    which is the compensating control that lets dispatch fall back to the
    risk_default floor without a runtime fail-safe (see resolve_declared_tier)."""
    meta = getattr(tool, "meta", None) or {}
    tier = meta.get(RISK_TIER_META_KEY) if isinstance(meta, dict) else None
    assert tier in _WIRE_VALID_TIERS, (
        f"tool {getattr(tool, 'name', tool)!r} must advertise a valid "
        f"{RISK_TIER_META_KEY!r} in _meta over the wire, got {tier!r}"
    )


def _assert_valid_produces_meta(tool: Any) -> None:
    """Assert a live tool's `produces` declaration is one this daemon knows.

    Unlike risk_tier, absent is valid here and means "result" -- `produces`
    is optional, not mandatory, so a worker that advertises nothing must
    pass. An unrecognised value that IS present is rejected at build/test
    time for the same reason the risk tier is: dispatch falls back to the
    safe reading, so nothing at runtime would ever surface the typo -- a tool
    meaning to declare `artifact` and writing `ARTIFACT` would silently
    stream its file contents as a tool result.
    """
    meta = getattr(tool, "meta", None) or {}
    if not isinstance(meta, dict) or PRODUCES_META_KEY not in meta:
        return
    produces = meta[PRODUCES_META_KEY]
    assert produces in VALID_PRODUCES, (
        f"tool {getattr(tool, 'name', tool)!r} advertises "
        f"{PRODUCES_META_KEY!r}={produces!r} in _meta, which is not one of "
        f"{VALID_PRODUCES}"
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


async def assert_streamable_http_conformance(endpoint: str) -> None:
    """Verify a live Streamable HTTP MCP worker meets contract expectations.

    Connects, initializes, lists tools, and checks each tool's metadata
    is well-formed. Raises AssertionError with a clear message on first
    failure.

    Workers' own test suites import this and run it against their
    real running server.
    """
    import asyncio

    from agent_core.workers.client import MCPClient

    client = MCPClient(endpoint)
    state = {"stage": "connect", "tools": None}

    async def _drive() -> None:
        """The entire client lifecycle, in ONE task.

        Everything the transport opens is also closed here, so its anyio
        scopes are entered and exited in the same task -- the invariant anyio
        actually enforces.
        """
        try:
            await client.connect()
            state["stage"] = "initialize"
            await client.initialize()
            state["stage"] = "list_tools"
            result = await client.list_tools()
            state["tools"] = getattr(result, "tools", None)
        finally:
            try:
                await client.close()
            except (Exception, asyncio.CancelledError):
                # A connection that never established cannot close cleanly,
                # and that is not what this check is about.
                pass

    # The bound OBSERVES the work; it does not wrap it in a cancel scope.
    #
    # This matters, and cost four CI runs to establish. Nine other tests drive
    # this same fixture with no asyncio timeout and pass on CI. The only two
    # that failed were the only two that imposed one -- and restructuring the
    # scope so it nested correctly around connect/close did NOT help, which
    # rules out simple LIFO ordering and points at the cancel scope itself.
    # asyncio.timeout cancels from OUTSIDE an anyio scope it knows nothing
    # about; asyncio.wait just watches a task, and cancellation, if needed, is
    # delivered INTO that task -- the sanctioned way to abort a partially
    # entered anyio scope, and the same pattern MCPClientPool._cancel_owner
    # already uses for exactly this reason.
    task = asyncio.create_task(_drive(), name=f"conformance:{endpoint}")
    done, _ = await asyncio.wait({task}, timeout=CONFORMANCE_TIMEOUT)

    if not done:
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait({task}, timeout=5)
        raise AssertionError(
            f"streamable_http_conformance: {state['stage']} exceeded "
            f"{CONFORMANCE_TIMEOUT}s for {endpoint!r}. The server accepted the "
            f"connection but never completed the response; check the server's "
            f"own log for a handler that returned early."
        )

    if task.cancelled():
        # NOT a timeout, and conflating the two cost real debugging time: an
        # MCP transport whose background task dies cancels its caller, so a
        # server-side failure arrives as a bare CancelledError -- which ends
        # the task CANCELLED, not merely failed. Task.exception() re-raises
        # for a cancelled task rather than returning, so this must be checked
        # first; doing it the other way round turned an unreachable endpoint
        # from a clean AssertionError into an escaping CancelledError.
        raise AssertionError(
            f"streamable_http_conformance: {state['stage']} was cancelled for "
            f"{endpoint!r} (the transport's task group failed, which usually "
            f"means the server errored -- or the endpoint is unreachable)"
        )

    exc = task.exception()
    if exc is not None:
        raise AssertionError(
            f"streamable_http_conformance: {state['stage']} failed for "
            f"{endpoint!r}: {describe_failure(exc)}"
        )

    tools = state["tools"]
    assert tools is not None, "list_tools returned no .tools attribute"
    assert isinstance(tools, list), f"tools is not a list: {type(tools).__name__}"

    for tool in tools:
        assert tool.name, f"tool has empty name: {tool!r}"
        schema = getattr(tool, "inputSchema", None)
        assert schema is not None, f"tool {tool.name!r} has no inputSchema"
        assert isinstance(schema, dict), (
            f"tool {tool.name!r} inputSchema is not a dict"
        )
        assert "type" in schema, (
            f"tool {tool.name!r} inputSchema missing top-level 'type'"
        )
        _assert_valid_risk_tier_meta(tool)
        _assert_valid_produces_meta(tool)


async def assert_stdio_conformance(spec: "WorkerSpec") -> None:
    """Verify a live stdio MCP worker meets contract expectations.

    Spawns the worker subprocess, runs the MCP handshake, lists tools,
    and checks each tool's metadata is well-formed. Raises
    AssertionError with a clear message on first failure.

    Workers' own test suites import this and pass their WorkerSpec in.
    """
    import asyncio

    from agent_core.workers.client import MCPClient

    if spec.transport != "stdio":
        raise AssertionError(
            f"stdio_conformance: spec {spec.name!r} has transport "
            f"{spec.transport!r}, not 'stdio'"
        )

    client = MCPClient.from_spec(spec)
    exc_to_raise: AssertionError | None = None
    try:
        try:
            await asyncio.wait_for(client.connect(), timeout=CONFORMANCE_TIMEOUT)
        except asyncio.TimeoutError as exc:
            exc_to_raise = AssertionError(
                f"stdio_conformance: connect timed out for {spec.name!r} "
                f"(command={spec.command!r})"
            )
        except (asyncio.CancelledError, FileNotFoundError, OSError) as exc:
            exc_to_raise = AssertionError(
                f"stdio_conformance: connect failed for {spec.name!r} "
                f"(command={spec.command!r}): {exc}"
            )
        except Exception as exc:
            exc_to_raise = AssertionError(
                f"stdio_conformance: connect failed for {spec.name!r} "
                f"(command={spec.command!r}): {exc}"
            )

        if exc_to_raise is not None:
            return  # Will raise in finally after cleanup

        try:
            await asyncio.wait_for(client.initialize(), timeout=CONFORMANCE_TIMEOUT)
        except asyncio.TimeoutError as exc:
            exc_to_raise = AssertionError(
                f"stdio_conformance: initialize timed out for {spec.name!r}"
            )
        except (asyncio.CancelledError, Exception) as exc:
            exc_to_raise = AssertionError(
                f"stdio_conformance: initialize failed for {spec.name!r}: {exc}"
            )

        if exc_to_raise is not None:
            return  # Will raise in finally after cleanup

        list_result = await client.list_tools()
        tools = getattr(list_result, "tools", None)
        assert tools is not None, "list_tools returned no .tools attribute"
        assert isinstance(tools, list), f"tools is not a list: {type(tools).__name__}"

        for tool in tools:
            assert tool.name, f"tool has empty name: {tool!r}"
            schema = getattr(tool, "inputSchema", None)
            assert schema is not None, (
                f"tool {tool.name!r} has no inputSchema"
            )
            assert isinstance(schema, dict), (
                f"tool {tool.name!r} inputSchema is not a dict"
            )
            assert "type" in schema, (
                f"tool {tool.name!r} inputSchema missing top-level 'type'"
            )
            _assert_valid_risk_tier_meta(tool)
            _assert_valid_produces_meta(tool)
    finally:
        try:
            await client.close()
        except (Exception, asyncio.CancelledError):
            # Suppress cleanup errors (e.g., connection never fully established).
            pass
        if exc_to_raise is not None:
            raise exc_to_raise
