import pytest
from agent_core.workers.risk import RiskGate, RISK_TIER_META_KEY
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WorkerSpec


class _Tool:
    def __init__(self, name, tier=None):
        self.name = name
        self.meta = {RISK_TIER_META_KEY: tier} if tier is not None else None


class _ListResult:
    def __init__(self, tools):
        self.tools = tools


class _Ok:
    isError = False
    content = []


class _FakeInner:
    """Stand-in MCPClientPool: records call_tool, serves canned list_tools."""
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def list_tools(self, worker):
        return _ListResult(self._tools)

    async def call_tool(self, worker, tool, arguments):
        self.calls.append((worker, tool))
        return _Ok()

    async def close_all(self):
        pass


class _Audit:
    def __init__(self):
        self.entries = []

    def append(self, entry):
        self.entries.append(entry)


def _pool(inner, spec, overrides=None):
    return RiskAwareToolPool(
        inner=inner,
        specs={spec.name: spec},
        risk_gate=RiskGate(overrides=overrides or []),
        approval_registry=ToolApprovalRegistry(),
        audit_log=_Audit(),
    )


@pytest.mark.asyncio
async def test_low_advertised_with_low_floor_auto_executes_and_records_source():
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_Tool("list_devices", "low")])
    pool = _pool(inner, spec)
    await pool.list_tools("frida")            # populate cache
    res = await pool.call_tool("frida", "list_devices", {})
    assert inner.calls == [("frida", "list_devices")]   # auto-executed (no prompt)
    assert pool._audit.entries[-1].declared_tier == "low"
    assert pool._audit.entries[-1].tier_source == "floor"


@pytest.mark.asyncio
async def test_critical_advertised_blocks_without_approval_channel():
    # low floor, but execute_script advertises critical -> declared critical ->
    # requires approval; with no send channel it must NOT auto-execute.
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_Tool("execute_script", "critical")])
    pool = _pool(inner, spec)
    await pool.list_tools("frida")
    res = await pool.call_tool("frida", "execute_script", {"source": "x"})
    assert inner.calls == []                  # blocked, never dispatched
    assert getattr(res, "isError", False) is True
    assert pool._audit.entries[-1].declared_tier == "critical"
    assert pool._audit.entries[-1].tier_source == "wire"


@pytest.mark.asyncio
async def test_missing_tier_falls_back_to_floor():
    # Option C: a tool advertising no tier uses the worker's risk_default floor
    # (not a dispatch-time fail-safe). low floor -> auto-executes, source "floor".
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_Tool("untagged_tool", None)])   # advertises no tier
    pool = _pool(inner, spec)
    await pool.list_tools("frida")
    res = await pool.call_tool("frida", "untagged_tool", {})
    assert inner.calls == [("frida", "untagged_tool")]   # auto-executed at floor
    assert pool._audit.entries[-1].declared_tier == "low"
    assert pool._audit.entries[-1].tier_source == "floor"


@pytest.mark.asyncio
async def test_call_before_discovery_uses_floor():
    # cache never populated (no list_tools call) -> advertised is None -> floor.
    # In the real flow discovery always precedes dispatch; this just documents
    # that an undiscovered tool resolves to risk_default, not a fail-safe.
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_Tool("list_devices", "low")])
    pool = _pool(inner, spec)
    res = await pool.call_tool("frida", "list_devices", {})
    assert inner.calls == [("frida", "list_devices")]
    assert pool._audit.entries[-1].declared_tier == "low"
    assert pool._audit.entries[-1].tier_source == "floor"
