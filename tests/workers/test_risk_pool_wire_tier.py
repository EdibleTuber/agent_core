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
        self._specs = {}

    def add_spec(self, spec):
        self._specs[spec.name] = spec

    def remove_spec(self, name):
        self._specs.pop(name, None)

    def spec(self, name):
        return self._specs.get(name)

    def names(self):
        return list(self._specs)

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
async def test_malformed_and_absent_advertised_tiers_are_distinguishable():
    """A malformed advertised tier is a possible tampering/bug signal
    (risk.py:57-59) and must be flagged tier_source="invalid_advertised",
    distinct from a genuinely absent tier's "floor" -- both resolve to the
    same declared_tier (the floor), so only tier_source carries the signal.

    Regression: RiskAwareToolPool.call_tool used to pre-combine the wire tier
    with the high-water mark via `_max_tier` BEFORE calling
    resolve_declared_tier. `_max_tier` only ever returns a recognized tier
    string or None, so a malformed-but-hashable value (a typo'd string, a
    bare int, ...) silently became None -- indistinguishable from "never
    advertised" -- and resolve_declared_tier reported "floor" for both.
    """
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([
        _Tool("untagged", None),   # genuinely absent
        _Tool("garbled", "ULTRA"),  # malformed: not a valid RiskTier string
        _Tool("also_garbled", 42),  # malformed: hashable but not a string
    ])
    pool = _pool(inner, spec)
    await pool.list_tools("frida")

    await pool.call_tool("frida", "untagged", {})
    assert pool._audit.entries[-1].declared_tier == "low"
    assert pool._audit.entries[-1].tier_source == "floor"

    await pool.call_tool("frida", "garbled", {})
    assert pool._audit.entries[-1].declared_tier == "low"
    assert pool._audit.entries[-1].tier_source == "invalid_advertised", (
        "a malformed wire tier was indistinguishable from an absent one"
    )

    await pool.call_tool("frida", "also_garbled", {})
    assert pool._audit.entries[-1].tier_source == "invalid_advertised"


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


class _RawMetaTool:
    """A tool whose `_meta` container is whatever the wire delivered, including
    something that is not a dict at all."""
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta


@pytest.mark.asyncio
@pytest.mark.parametrize("falsy_meta", [[], "", 0])
async def test_a_falsy_non_dict_meta_is_not_mistaken_for_an_honest_worker(falsy_meta):
    """A non-dict `_meta` is a malformed/tampered container and must surface as
    "invalid_advertised", never as "floor".

    list_tools documents exactly this intent -- a non-dict meta is "recorded AS
    IS rather than normalized to None", because None reads as an honest
    non-advertiser in the audit log. `meta = getattr(...) or {}` silently broke
    that promise for every FALSY non-dict: `[]`, `""` and `0` all collapsed to
    `{}`, which is a dict, so `.get()` returned None and the tool was logged as
    though it had simply declined to advertise. Truthy non-dicts ("nope", [1])
    were unaffected, which is why the existing malformed-tier test -- whose bad
    values live inside a dict -- never caught it.
    """
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_RawMetaTool("garbled_container", falsy_meta)])
    pool = _pool(inner, spec)
    await pool.list_tools("frida")

    await pool.call_tool("frida", "garbled_container", {})
    entry = pool._audit.entries[-1]
    assert entry.declared_tier == "low"        # still the floor, as designed
    assert entry.tier_source == "invalid_advertised", (
        f"meta={falsy_meta!r} was recorded as an honest non-advertiser"
    )


@pytest.mark.asyncio
async def test_an_absent_meta_still_reads_as_an_honest_non_advertiser():
    """The other half of the same distinction, pinned so a future fix to the
    line above cannot achieve it by flagging everything as malformed."""
    spec = WorkerSpec(name="frida", transport="stdio", command="x", risk_default="low")
    inner = _FakeInner([_RawMetaTool("quiet", None), _RawMetaTool("empty", {})])
    pool = _pool(inner, spec)
    await pool.list_tools("frida")

    for name in ("quiet", "empty"):
        await pool.call_tool("frida", name, {})
        assert pool._audit.entries[-1].tier_source == "floor", (
            f"{name!r} advertises nothing and must not be flagged as malformed")
