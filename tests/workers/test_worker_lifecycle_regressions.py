"""Regression tests for the v1.8.0 final-review findings.

Each test here pins one of the ways the runtime worker lifecycle could
downgrade the risk gate or orphan a worker subprocess. They are grouped in
their own module (rather than folded into test_manager.py) because they all
share the same "drive a real stdio child, then interfere with it" shape.
"""
import asyncio
import contextlib
import json
import os
from pathlib import Path

import pytest

from agent_core.tools.executor import ToolExecutor
from agent_core.workers import client_pool as client_pool_mod
from agent_core.workers.audit import AuditLog
from agent_core.workers.client import MCPClient
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.manager import WorkerManager
from agent_core.workers.registry import WorkerRegistry
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry, ToolDecision
from agent_core.workers.types import WorkerSpec


@pytest.fixture(autouse=True)
def _empty_builtin_tools(monkeypatch):
    """Same isolation test_manager.py uses: the bare _Agent stub below has
    none of the attrs BUILTIN_TOOLS entries now require."""
    monkeypatch.setattr("agent_core.tools.builtin.BUILTIN_TOOLS", [])
    monkeypatch.setattr("agent_core.tools.executor.BUILTIN_TOOLS", [])


class _Agent:
    pass


def _manager(tmp_path, specs, **kw):
    reg = WorkerRegistry()
    for s in specs:
        reg.add(s)
    inner = MCPClientPool([])
    pool = RiskAwareToolPool(
        inner=inner, specs={}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    ex = ToolExecutor.build(_Agent(), [])
    return WorkerManager(reg, pool, ex, **kw), pool, ex, inner


def _audit_rows(tmp_path):
    rows = []
    for path in sorted(tmp_path.glob("audit-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def _pid_alive(pid: int | None) -> bool:
    """True only while `pid` is a *running* process.

    `os.kill(pid, 0)` is not usable here: the worker child is our own
    subprocess, so after a SIGKILL it stays a reaped-pending zombie that
    signal 0 still reports as existing. Read the state field out of
    /proc/<pid>/stat instead and treat Z (and a vanished pid) as dead.
    """
    if pid is None:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    # comm can contain spaces/parens; the state field is the first token
    # after the final ')'.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


async def _slow_close(self):
    await asyncio.sleep(3600)


# --- CRITICAL 1 ---------------------------------------------------------


async def test_cancelled_load_leaves_no_live_ungated_worker(
        tmp_path, stdio_stub_spec, monkeypatch):
    """Cancelling a /worker load mid-connect must leave NO residue.

    Pre-fix composition: `_load_locked` registered the spec (clearing the
    worker's wire-tier table) before awaiting connect, and neither
    `_load_locked`'s rollback nor `MCPClientPool.connect`'s cleanup caught
    CancelledError. The daemon cancels a handler task the instant its client
    disconnects, so a Ctrl-C'd `/worker load` left the spec registered, the
    subprocess spawned and the client published — with `_tool_tiers` empty,
    `is_loaded()` False and `_errors` empty. A direct `tool_pool.call_tool`
    (which PARE does by hard-coded name) then resolved every tool to the
    worker's `risk_default` floor and auto-executed it with no prompt.
    """
    orig_connect = MCPClient.connect
    started = asyncio.Event()

    async def _slow_connect(self):
        started.set()
        await asyncio.sleep(0.25)
        await orig_connect(self)

    monkeypatch.setattr(MCPClient, "connect", _slow_connect)

    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    task = asyncio.create_task(mgr.load("stub"))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Give the (pre-fix, un-cancelled) owner task every chance to finish its
    # connect and publish the client, so this asserts the settled state.
    await asyncio.sleep(0.6)

    assert inner.spec("stub") is None, "the spec survived a cancelled load"
    assert not inner.is_connected("stub"), "a client survived a cancelled load"
    assert "stub" not in inner._owners, "an owner task survived a cancelled load"
    assert not mgr.is_loaded("stub")
    assert not any(n.startswith("stub_") for n in ex.names())
    assert mgr.status()[0].last_error, "a cancelled load must be visible in status()"

    # The load never completed, so nothing may dispatch. With the spec gone
    # the pool resolves an unknown worker to "high" and fails closed on the
    # missing approval channel; pre-fix the spec was still live and this
    # dispatched at the floor with no prompt at all.
    result = await pool.call_tool("stub", "risky_high", {"target": "x"})
    assert getattr(result, "isError", False), (
        "a cancelled load left a live, reachable, ungated worker: the call "
        "dispatched instead of being blocked")
    rows = [r for r in _audit_rows(tmp_path) if r["tool"] == "risky_high"]
    assert all(r["outcome"] != "ok" for r in rows)


# --- CRITICAL 2 ---------------------------------------------------------


async def test_disconnect_timeout_hard_kills_the_recorded_child(
        tmp_path, stdio_stub_spec, monkeypatch):
    """Spec section 7: "on timeout, HARD-KILL the recorded child pid".

    Pre-fix the bound shipped but the kill did not, so a worker holding e.g.
    a live Frida attachment survived its own unload — and `_reap` popped the
    owner out of `_owners` before awaiting, so after the manager's timeout
    cancelled it no API could reach the orphan again.
    """
    monkeypatch.setattr(MCPClient, "close", _slow_close)

    mgr, pool, ex, inner = _manager(
        tmp_path, [stdio_stub_spec("stub", "low")], disconnect_timeout=0.3)
    assert (await mgr.load("stub")).ok
    pid = inner._owner_pid("stub")
    assert pid is not None and _pid_alive(pid)

    res = await mgr.unload("stub")
    assert not res.ok and res.error_kind == "disconnect_timeout"

    for _ in range(50):
        if not _pid_alive(pid):
            break
        await asyncio.sleep(0.05)
    assert not _pid_alive(pid), (
        f"worker child {pid} survived a wedged unload — for a process-attaching "
        f"worker that is a live attachment outliving the daemon")


# --- CRITICAL 3 ---------------------------------------------------------


async def test_close_all_is_bounded_against_a_wedged_worker(
        tmp_path, stdio_stub_spec, monkeypatch):
    """`close_all` is the ONLY reaper for a worker that never entered
    `manager._loaded`, and pre-fix its reap had no timeout at all: a worker
    wedged in close hung `ashutdown` forever, so systemd SIGKILLed the daemon
    with worker subprocesses still up.
    """
    monkeypatch.setattr(MCPClient, "close", _slow_close)
    monkeypatch.setattr(client_pool_mod, "DEFAULT_OWNER_JOIN_TIMEOUT", 0.3)

    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok
    pid = inner._owner_pid("stub")
    # Simulate the Critical-1 orphan shape: connected and owned by the pool,
    # but absent from the manager's `_loaded`, so close_all's bounded unload
    # loop never sees it and only `pool.close_all()` can reap it.
    mgr._loaded.pop("stub")

    await asyncio.wait_for(mgr.close_all(), timeout=5)

    assert "stub" not in inner._owners
    for _ in range(50):
        if not _pid_alive(pid):
            break
        await asyncio.sleep(0.05)
    assert not _pid_alive(pid), f"worker child {pid} survived close_all()"


# --- IMPORTANT 4 --------------------------------------------------------


def test_worker_manager_rejects_a_bare_client_pool(tmp_path):
    """`RiskAwareToolPool` is the entire security boundary, and every method
    WorkerManager calls except `emit_lifecycle` exists identically on the
    inner pool — so a mis-wire produced a working worker fleet with no risk
    gate, no approval and no audit, signalled only by a logger.warning."""
    with pytest.raises(TypeError, match="RiskAwareToolPool"):
        WorkerManager(WorkerRegistry(), MCPClientPool([]),
                      ToolExecutor.build(_Agent(), []))


def test_worker_manager_accepts_the_risk_pool(tmp_path):
    inner = MCPClientPool([])
    pool = RiskAwareToolPool(
        inner=inner, specs={}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    WorkerManager(WorkerRegistry(), pool, ToolExecutor.build(_Agent(), []))


# --- IMPORTANT 5 --------------------------------------------------------


async def test_reload_preserves_the_wire_tier_ratchet_and_evicts_approvals(
        tmp_path, stdio_stub_spec):
    """The composite ordering (`_unload_locked` bumps, `_load_locked` bumps,
    `list_tools` refills) against a real reload.

    NOTE: this one passes pre-fix — it pins behaviour that was already
    correct but untested, per required-test 5.
    """
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok
    assert pool.resolve_effective("stub", "risky_high") == "high"

    gen = pool.generation("stub")
    pool.record_session_approval("stub", "risky_high", gen)
    assert pool.is_session_approved("stub", "risky_high")

    assert (await mgr.reload("stub")).ok
    assert pool.generation("stub") > gen
    assert not pool.is_session_approved("stub", "risky_high"), (
        "a session approval survived a reload and pre-approved a new process")
    assert pool._tier_highwater[("stub", "risky_high")] == "high"
    assert pool.resolve_effective("stub", "risky_high") == "high"


async def test_reload_cannot_lower_the_risk_default_floor(
        tmp_path, stdio_stub_spec):
    """`_tier_highwater` was fed only from the wire meta, so the ratchet said
    nothing about the floor. `reload()` re-reads the registry and
    `WorkerRegistry.add()` is public: lower `risk_default`, re-add, reload,
    and every tool that does not advertise drops to the new floor."""
    spec = stdio_stub_spec("stub", "high")
    mgr, pool, ex, inner = _manager(tmp_path, [spec])
    assert (await mgr.load("stub")).ok
    # noop_low advertises "low"; the worker-wide floor is what holds it up.
    assert pool.resolve_effective("stub", "noop_low") == "high"

    mgr._registry.add(spec.model_copy(update={"risk_default": "low"}))
    assert (await mgr.reload("stub")).ok

    assert pool.resolve_effective("stub", "noop_low") == "high", (
        "a reload against a lowered risk_default downgraded a tool below the "
        "highest floor seen this session")


# --- IMPORTANT 6 --------------------------------------------------------


async def test_lifecycle_rows_carry_artifact_forensics_and_an_action(
        tmp_path, stdio_stub_spec):
    """Section 7's artifact-swap forensics: the resolved command path plus
    the binary's mtime/size, so a swap at the same path is visible; and an
    `action` field so a reload is not indistinguishable from an unrelated
    unload followed by a load."""
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok
    assert (await mgr.reload("stub")).ok

    loaded = [r for r in _audit_rows(tmp_path) if r["outcome"] == "worker_loaded"]
    assert len(loaded) == 2
    first = loaded[0]["args"]
    assert first["action"] == "load"
    assert first["resolved_command"] and os.path.isabs(first["resolved_command"])
    assert first["command_mtime"] is not None
    assert first["command_size"] is not None
    assert loaded[1]["args"]["action"] == "reload"

    unloaded = [r for r in _audit_rows(tmp_path) if r["outcome"] == "worker_unloaded"]
    assert unloaded and unloaded[-1]["args"]["action"] == "reload"


# --- IMPORTANT 8 --------------------------------------------------------


async def test_parked_approval_blocked_by_a_concurrent_reload(
        tmp_path, stdio_stub_spec):
    """A reload landing while an approval prompt is parked must not let the
    answer pre-approve the new process.

    NOTE: this one passes pre-fix — it is the already-shipped guard at the
    end of `_await_operator`, previously untested against a real reload.
    """
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok

    sent: list = []

    async def _send(msg):
        sent.append(msg)

    pool._send = _send
    call = asyncio.create_task(
        pool.call_tool("stub", "risky_high", {"target": "x"}))
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert sent, "no approval request was emitted"

    assert (await mgr.reload("stub")).ok
    pool._registry.resolve(sent[0].proposal_id,
                           ToolDecision(approved=True, justification="ok"))
    result = await asyncio.wait_for(call, timeout=10)
    assert getattr(result, "isError", False)
    assert "reloaded" in result.content[0].text


async def test_generation_is_rechecked_immediately_before_dispatch(
        tmp_path, stdio_stub_spec, monkeypatch):
    """The pre-fix guard was correct only by the ACCIDENTAL absence of a
    suspension point between the approval check and `_inner.call_tool`.

    Simulate the capture hook / dispatch semaphore / rate limiter the finding
    warns about by inserting exactly one `await` on that path, and land a
    reload in the window it opens. An operator-approved high-tier call must
    not dispatch against a different subprocess.
    """
    mgr, pool, ex, inner = _manager(tmp_path, [stdio_stub_spec("stub", "low")])
    assert (await mgr.load("stub")).ok

    original = pool._execute_and_audit
    landed = asyncio.Event()

    async def _yielding_execute(*args, **kwargs):
        if not landed.is_set():
            landed.set()
            await asyncio.sleep(0)          # the newly-introduced suspension point
            assert (await mgr.reload("stub")).ok
        return await original(*args, **kwargs)

    monkeypatch.setattr(pool, "_execute_and_audit", _yielding_execute)

    sent: list = []

    async def _send(msg):
        sent.append(msg)

    pool._send = _send
    call = asyncio.create_task(
        pool.call_tool("stub", "risky_high", {"target": "x"}))
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert sent, "no approval request was emitted"
    pool._registry.resolve(sent[0].proposal_id,
                           ToolDecision(approved=True, justification="ok"))

    result = await asyncio.wait_for(call, timeout=10)
    assert landed.is_set()
    assert getattr(result, "isError", False), (
        "an operator-approved call dispatched against a worker that was "
        "reloaded after the approval check")
    assert "reload" in result.content[0].text.lower()


# --- MINOR 3 ------------------------------------------------------------


async def test_malformed_meta_container_is_recorded_as_invalid_advertised(tmp_path):
    """A non-dict `meta` normalized to None recorded as `"floor"` —
    indistinguishable in the audit log from an honest non-advertiser."""
    spec = WorkerSpec(name="w", transport="stdio", risk_default="low",
                      command="/bin/true")
    inner = MCPClientPool([spec])
    pool = RiskAwareToolPool(
        inner=inner, specs={spec.name: spec}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))

    class _Tool:
        name = "t"
        meta = ["not", "a", "dict"]

    class _Listing:
        tools = [_Tool()]

    async def _listing(worker):
        return _Listing()

    async def _call(worker, tool, arguments):
        class _R:
            isError = False
            content = []
        return _R()

    inner.list_tools = _listing
    inner.call_tool = _call
    await pool.list_tools("w")
    await pool.call_tool("w", "t", {})
    assert _audit_rows(tmp_path)[-1]["tier_source"] == "invalid_advertised"


async def test_highwater_escalation_preserves_invalid_advertised(tmp_path):
    """When the high-water mark escalates over a malformed advertised value,
    overwriting `tier_source` to "wire" erased the tampering signal."""
    spec = WorkerSpec(name="w", transport="stdio", risk_default="low",
                      command="/bin/true")
    inner = MCPClientPool([spec])
    pool = RiskAwareToolPool(
        inner=inner, specs={spec.name: spec}, risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))
    pool._tier_highwater[("w", "t")] = "high"
    pool._tool_tiers[("w", "t")] = "not-a-tier"
    declared, source = pool._resolve_declared("w", "t")
    assert declared == "high"
    assert source == "invalid_advertised"
