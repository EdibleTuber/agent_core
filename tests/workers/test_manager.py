"""WorkerManager lifecycle against a real stdio worker subprocess."""
import asyncio
import contextlib
import json

import pytest

from agent_core.tools.executor import ToolExecutor
from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.manager import WorkerManager
from agent_core.workers.registry import WorkerRegistry
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WorkerSpec


@pytest.fixture(autouse=True)
def _empty_builtin_tools(monkeypatch):
    """Isolate manager tests from the live BUILTIN_TOOLS list.

    BUILTIN_TOOLS now carries entries requiring agent.config /
    agent.allowlist / etc. (tests/test_tools_executor.py's
    `_empty_builtin_tools` documents the same need for the same reason).
    The bare `_Agent` stub below has none of those attrs, and these tests
    only care about worker-synthesized tools, so empty the builtins the
    same way tests/test_tools_executor.py does.
    """
    monkeypatch.setattr("agent_core.tools.builtin.BUILTIN_TOOLS", [])
    monkeypatch.setattr("agent_core.tools.executor.BUILTIN_TOOLS", [])


class _Agent:
    pass


def _manager(tmp_path, specs):
    reg = WorkerRegistry()
    for s in specs:
        reg.add(s)
    inner = MCPClientPool([])
    pool = RiskAwareToolPool(
        inner=inner, specs={}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    ex = ToolExecutor.build(_Agent(), [])
    return WorkerManager(reg, pool, ex), pool, ex, inner


def _audited_outcomes(tmp_path):
    """AuditLog has no reader API (only `append`, per audit.py) — read the
    JSONL rows directly the same way tests/workers/test_audit.py does."""
    outcomes = []
    for path in sorted(tmp_path.glob("audit-*.jsonl")):
        for line in path.read_text().splitlines():
            outcomes.append(json.loads(line)["outcome"])
    return outcomes


async def test_load_registers_prefixed_tools(tmp_path, stdio_stub_spec):
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    res = await mgr.load("stub")
    assert res.ok, res.error
    assert res.tool_count == 2
    assert "stub_noop_low" in ex and "stub_risky_high" in ex


async def test_unload_removes_only_its_own_tools(tmp_path, stdio_stub_spec):
    """A non-worker tool must survive the worker's unload.

    With BUILTIN_TOOLS emptied by the autouse fixture and no agent tools
    passed, `builtins` was previously always `set()`, so this only ever
    asserted "the executor ends up empty" — it could not tell `remove_worker`
    apart from a blanket `self._tools.clear()`. A sentinel non-worker tool
    gives the baseline teeth.
    """
    from agent_core.tools.base import Tool

    class _Keeper(Tool):
        name = "keeper"
        description = "a non-worker tool that must not be touched by unload"
        parameters = {"type": "object", "properties": {}}

        async def run(self, args, ctx):
            return "kept"

    reg = WorkerRegistry()
    reg.add(stdio_stub_spec("stub", "low"))
    inner = MCPClientPool([])
    pool = RiskAwareToolPool(
        inner=inner, specs={}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    ex = ToolExecutor.build(_Agent(), [_Keeper])
    mgr = WorkerManager(reg, pool, ex)

    builtins = set(ex.names())
    assert builtins == {"keeper"}
    await mgr.load("stub")
    res = await mgr.unload("stub")
    assert res.ok and res.tool_count == 2
    assert set(ex.names()) == builtins
    assert not inner.is_connected("stub")


async def test_unload_of_a_slow_close_yields_disconnect_timeout(
        tmp_path, stdio_stub_spec, monkeypatch):
    """A worker whose close() outlives disconnect_timeout must not wedge
    unload() forever -- it must report disconnect_timeout within the bound.

    Regression test for the round-2 fix to client_pool.MCPClientPool._reap
    (asyncio.shield). Without it, the manager's own `wait_for(...,
    timeout=self._disconnect_timeout)` cancellation reached the owner task
    itself (through _reap's bare `await task`) rather than stopping at the
    caller's own frame -- so unload() ended up waiting out however long the
    (now also cancelled) owner's close() actually took, unbounded, instead
    of bounding the wait and reporting disconnect_timeout. Notably, close()
    here does NOT need to ignore cancellation to prove this: with the fix,
    the owner is never cancelled at all in this path (that's the point --
    only the caller's own wait unwinds), so a plain slow close() that
    hasn't returned yet is already sufficient.
    """
    from agent_core.workers.client import MCPClient

    async def _slow_close(self):
        await asyncio.sleep(3600)

    monkeypatch.setattr(MCPClient, "close", _slow_close)

    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    mgr._disconnect_timeout = 0.3
    assert (await mgr.load("stub")).ok
    owner = inner._owners["stub"]

    res = await mgr.unload("stub")
    assert not res.ok
    assert res.error_kind == "disconnect_timeout"
    assert not owner.done(), "the owner (and its slow close()) must survive unload()'s timeout"

    # The owner is still parked inside the monkeypatched close(), never
    # cancelled -- clean it up directly rather than leave it running for the
    # rest of the session.
    owner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await owner


async def test_unload_from_a_different_task(tmp_path, stdio_stub_spec):
    """The daemon loads in astartup's task and unloads in a handler task."""
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    await mgr.load("stub")
    res = await asyncio.create_task(mgr.unload("stub"))
    assert res.ok, res.error
    assert not inner.is_connected("stub")


async def test_dispatch_after_unload_does_not_resurrect_the_worker(
        tmp_path, stdio_stub_spec):
    """If the pool keeps declared (not loaded) specs, the next call lazily
    respawns the worker you just unloaded, silently undoing the unload."""
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    await mgr.load("stub")
    await mgr.unload("stub")
    with pytest.raises(KeyError):
        await inner.list_tools("stub")
    assert not inner.is_connected("stub")


async def test_reload_yields_a_fresh_process(tmp_path, stdio_stub_spec):
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    await mgr.load("stub")
    first = inner._owner_pid("stub")
    res = await mgr.reload("stub")
    assert res.ok, res.error
    assert inner._owner_pid("stub") != first


async def test_failed_load_leaves_no_residue(tmp_path):
    spec = WorkerSpec(name="broken", transport="stdio", risk_default="low",
                      command="/nonexistent/binary")
    mgr, pool, ex, inner = _manager(tmp_path, [spec])
    res = await mgr.load("broken")
    assert not res.ok
    assert res.error_kind == "spawn_failed"
    assert inner.spec("broken") is None
    assert not any(n.startswith("broken_") for n in ex.names())
    assert mgr.status()[0].last_error


async def test_load_of_unknown_worker(tmp_path, stdio_stub_spec):
    mgr, *_ = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    res = await mgr.load("nope")
    assert not res.ok and res.error_kind == "unknown_worker"
    assert "stub" in res.error


async def test_load_and_unload_are_idempotent(tmp_path, stdio_stub_spec):
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok
    assert (await mgr.load("stub")).ok
    assert (await mgr.unload("stub")).ok
    assert (await mgr.unload("stub")).ok


async def test_collision_fails_the_whole_load(tmp_path, stdio_stub_spec):
    """One shadowing tool among many must not half-register the worker."""
    from agent_core.tools.base import Tool

    class _Clash(Tool):
        name = "stub_noop_low"
        description = "d"
        parameters = {"type": "object", "properties": {}}

    reg = WorkerRegistry()
    reg.add(stdio_stub_spec("stub", "low"))
    inner = MCPClientPool([])
    pool = RiskAwareToolPool(
        inner=inner, specs={}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    ex = ToolExecutor.build(_Agent(), [_Clash])
    mgr = WorkerManager(reg, pool, ex)

    res = await mgr.load("stub")
    assert not res.ok and res.error_kind == "tool_collision"
    assert "stub_noop_low" in res.error
    assert "stub_risky_high" not in ex, "partial registration"
    # The collision is only discovered after connect()+list_tools() have
    # already succeeded, so this is the one test that drives _fail's
    # rollback against a live subprocess. The rollback must fully undo the
    # connect, not just the tool registration.
    assert not inner.is_connected("stub")
    assert inner.spec("stub") is None


async def test_load_autoload_skips_autoload_false(tmp_path, stdio_stub_spec):
    on = stdio_stub_spec("on", "low")
    off = stdio_stub_spec("off", "low")
    off = off.model_copy(update={"autoload": False})
    mgr, pool, ex, inner = _manager(tmp_path, [on, off])
    results = await mgr.load_autoload()
    assert {r.name for r in results} == {"on"}
    assert inner.is_connected("on") and not inner.is_connected("off")


async def test_unavailable_reason(tmp_path, stdio_stub_spec):
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert "not loaded" in mgr.unavailable_reason("stub")
    await mgr.load("stub")
    assert mgr.unavailable_reason("stub") is None
    assert "not declared" in mgr.unavailable_reason("ghost")


def test_worker_of_prefers_the_longest_matching_worker_name(tmp_path, stdio_stub_spec):
    """A worker named `x` must not claim a tool belonging to worker `x_y`."""
    mgr, pool, ex, inner = _manager(
        tmp_path, [stdio_stub_spec("x", "low"), stdio_stub_spec("x_y", "low")])
    assert mgr.worker_of("x_y_z") == "x_y"
    assert mgr.worker_of("x_z") == "x"
    assert mgr.worker_of("nope") is None


async def test_lifecycle_rows_are_audited(tmp_path, stdio_stub_spec):
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    await mgr.load("stub")
    await mgr.unload("stub")
    outcomes = _audited_outcomes(tmp_path)
    assert "worker_loaded" in outcomes and "worker_unloaded" in outcomes
