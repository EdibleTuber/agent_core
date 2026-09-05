"""Security invariants that must hold ACROSS load/unload/reload.

Each test here corresponds to a way the boot-frozen daemon was safe by
construction and the runtime-mutable one is not (spec section 6.4).
"""
import asyncio
import json

import pytest

from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry, ToolDecision
from agent_core.workers.types import WorkerSpec


def _pool(tmp_path, spec):
    inner = MCPClientPool([spec])
    return inner, RiskAwareToolPool(
        inner=inner, specs={spec.name: spec}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path),
    )


def _pool_multi(tmp_path, specs):
    inner = MCPClientPool(specs)
    return inner, RiskAwareToolPool(
        inner=inner, specs={s.name: s for s in specs}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path),
    )


def _spec(name="frida", floor="low"):
    return WorkerSpec(name=name, transport="stdio", risk_default=floor,
                      command="/bin/true")


def _audit_rows(tmp_path):
    """AuditLog exposes no reader (only .append) — read the JSONL it writes
    directly, the same way tests/workers/test_risk_pool.py's _audit_lines
    helper does."""
    files = list(tmp_path.glob("audit-*.jsonl"))
    rows = []
    for f in files:
        rows.extend(json.loads(line) for line in f.read_text().splitlines())
    return rows


class _Tool:
    def __init__(self, name, tier):
        self.name = name
        self.meta = {"agent_core/risk_tier": tier} if tier else None


class _Listing:
    def __init__(self, tools):
        self.tools = tools


async def test_tier_never_ratchets_down_on_reload(tmp_path, monkeypatch):
    """A reload against a build that stops advertising must NOT fall to the floor.

    frida's risk_default is low and only execute_script/write_memory are pinned,
    so read_memory and java_hook are protected solely by the wire tier. Without
    a high-water mark, a reload silently makes them auto-execute.
    """
    spec = _spec()
    inner, pool = _pool(tmp_path, spec)

    async def listing_high(worker):
        return _Listing([_Tool("read_memory", "high")])
    monkeypatch.setattr(inner, "list_tools", listing_high)
    await pool.list_tools("frida")
    assert pool.resolve_effective("frida", "read_memory") == "high"

    # Reload: the new build advertises nothing.
    pool.remove_spec("frida")
    pool.add_spec(spec)

    async def listing_silent(worker):
        return _Listing([_Tool("read_memory", None)])
    monkeypatch.setattr(inner, "list_tools", listing_silent)
    await pool.list_tools("frida")

    assert pool.resolve_effective("frida", "read_memory") == "high", (
        "reload lowered the effective tier — a wire downgrade must fail closed"
    )


async def test_session_approval_does_not_survive_a_reload(tmp_path):
    """An approval resolving AFTER the reload must not apply to the new process.

    Evicting on unload and load is not enough: the operator answers the prompt
    on their own schedule, and that can land after the load completes.
    """
    spec = _spec()
    inner, pool = _pool(tmp_path, spec)

    gen_before = pool.generation("frida")
    # Simulate the approval being granted against the pre-reload generation.
    pool.record_session_approval("frida", "java_hook", gen_before)
    assert pool.is_session_approved("frida", "java_hook")

    pool.remove_spec("frida")
    pool.add_spec(spec)

    assert not pool.is_session_approved("frida", "java_hook")
    # And a late-resolving approval stamped with the OLD generation is dropped.
    pool.record_session_approval("frida", "java_hook", gen_before)
    assert not pool.is_session_approved("frida", "java_hook"), (
        "an approval granted before the reload was applied to the new worker"
    )


async def test_close_all_also_clears_approvals(tmp_path):
    """close_all lazily reconnects fresh subprocesses; approvals must not carry."""
    spec = _spec()
    inner, pool = _pool(tmp_path, spec)
    pool.record_session_approval("frida", "java_hook", pool.generation("frida"))
    await pool.close_all()
    assert not pool.is_session_approved("frida", "java_hook")


async def test_close_all_clears_approvals_for_a_never_reloaded_worker_too(tmp_path):
    """Regression: close_all used `list(self._generations) or self._inner.names()`.

    `or` short-circuits rather than unions: as soon as ANY worker has been
    through add_spec/remove_spec once, `_generations` is non-empty and
    `_inner.names()` is never consulted, so every worker that was never
    individually reloaded keeps generation 0 and its session approvals
    survive a full close_all teardown -- exactly the case the previous test
    can't see, because it never calls add_spec/remove_spec on anything.

    Here `frida` gets reloaded (bumping `_generations`); `adb` never does.
    close_all must still evict adb's approval.
    """
    frida = _spec("frida")
    adb = _spec("adb")
    inner, pool = _pool_multi(tmp_path, [frida, adb])

    pool.record_session_approval("adb", "shell", pool.generation("adb"))
    assert pool.is_session_approved("adb", "shell")

    # Reload an UNRELATED worker first. This must not make close_all skip adb.
    pool.remove_spec("frida")
    pool.add_spec(frida)

    await pool.close_all()
    assert not pool.is_session_approved("adb", "shell"), (
        "close_all skipped a never-reloaded worker and laundered its approval "
        "across a full teardown"
    )


async def test_cancelled_dispatch_is_audited(tmp_path, monkeypatch):
    """Unload mid-dispatch surfaces as CancelledError, a BaseException that every
    guard on this path misses — so the dispatch executed with no audit row."""
    spec = _spec(floor="low")
    inner, pool = _pool(tmp_path, spec)

    async def boom(worker, tool, arguments):
        raise asyncio.CancelledError()
    monkeypatch.setattr(inner, "call_tool", boom)

    with pytest.raises(asyncio.CancelledError):
        await pool.call_tool("frida", "list_devices", {})

    rows = _audit_rows(tmp_path)
    assert rows, "expected an audit row for the cancelled dispatch"
    assert rows[-1]["outcome"] == "cancelled"


async def test_hostile_tier_earlier_in_listing_does_not_abort_discovery(tmp_path, monkeypatch):
    """A hostile/unhashable advertised tier (risk.py:57-59 documents that the
    wire tier "may be hostile/arbitrary (dict, list, bool, ...)") for one tool
    must not raise and abort discovery of tools listed after it -- nor drop
    them to the floor by leaving them unrecorded.

    Regression for `_max_tier`'s `t in _TIER_ORDER` membership test, which
    raises TypeError on an unhashable `t` (a dict/list) since `_TIER_ORDER`
    is a dict; the raise happened inside the list_tools loop, after the
    hostile tool but before any tool listed after it was processed.
    """
    spec = _spec()
    inner, pool = _pool(tmp_path, spec)

    async def hostile_listing(worker):
        return _Listing([
            _Tool("write_memory", "critical"),
            _Tool("hostile_tool", {"x": 1}),   # unhashable -- must not raise
            _Tool("read_memory", "high"),      # listed AFTER the hostile entry
        ])
    monkeypatch.setattr(inner, "list_tools", hostile_listing)

    await pool.list_tools("frida")   # must not raise

    assert pool.resolve_effective("frida", "read_memory") == "high", (
        "a hostile entry earlier in the listing aborted discovery, leaving a "
        "later tool's wire tier unrecorded and its effective tier at the floor"
    )


async def test_specs_are_read_through_not_duplicated(tmp_path):
    """One source of truth: a spec removed from the inner pool is gone here too.

    If the two dicts drift with the inner cleared and the outer kept, dispatch
    resolves at the worker's floor and skips HITL entirely.
    """
    spec = _spec()
    inner, pool = _pool(tmp_path, spec)
    inner.remove_spec("frida")
    assert pool.spec_for("frida") is None
