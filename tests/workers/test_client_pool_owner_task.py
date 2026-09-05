"""Connection teardown must not be bound to the task that connected.

anyio binds a cancel scope to the entering task. The daemon connects workers
in its startup task (Agent.astartup) but unloads them from a per-message
handler task (daemon.py:103), so a pool that closes in the caller's task
raises RuntimeError -- and, measured, leaves the worker subprocess running
until the event loop exits. Every pre-existing test in test_client_pool.py
connects and closes inside one coroutine, so none of them can catch this.
"""
import asyncio
import contextlib
import os
import sys

import pytest


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


async def test_reap_propagates_caller_cancellation_without_cancelling_the_owner(
        stdio_stub_spec):
    """Mirrors the reviewer's probe for the round-2 finding: cancelling
    whoever calls `_reap` (e.g. the daemon shutting down mid-`unload()`, or
    `MCPClientPool.disconnect` being cancelled by `WorkerManager`'s own
    `wait_for(disconnect_timeout)`) must propagate as CancelledError to that
    caller, not be silently absorbed as though it were the owner task's own
    outcome.

    Without `asyncio.shield`, `Task.cancel()` cancels whatever future is
    currently the caller's `_fut_waiter`; when the caller is blocked in a
    bare `await task`, `task` IS that future, so cancelling the caller
    cancels the owner task too, as a direct side effect -- not merely raises
    CancelledError past it. That makes `task.cancelled()` become True as a
    result of the CALLER's own cancellation, defeating the disambiguation
    `_reap` relies on and silently eating the cancellation.

    Since the final review, the caller's cancellation ALSO means "stop
    waiting for this owner" -- it is how a manager-level disconnect_timeout
    reaches the pool -- so the owner is handed to `_abandon()`: its child is
    hard-killed and the task is cancelled and parked in `_orphans` rather
    than left running behind a dropped reference. What must NOT change is
    that the caller's own cancellation still propagates.
    """
    from agent_core.workers.client_pool import MCPClientPool

    pool = MCPClientPool([stdio_stub_spec("stub", "low")])
    await pool.list_tools("stub")             # connects; owner now parked
    owner = pool._owners["stub"]
    pid = pool._owner_pid("stub")
    assert pid is not None and _alive(pid)

    ran_after = []

    async def caller():
        await pool._reap("stub")
        ran_after.append("after")

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)                     # let it enter _reap's await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran_after == [], "the caller's cancellation must propagate, not be absorbed"
    # Abandoned, not silently dropped: still reachable (in _orphans until it
    # finishes unwinding) and its child already killed.
    assert owner in pool._orphans or owner.done()

    with contextlib.suppress(asyncio.CancelledError):
        await owner
    assert owner.done()
    assert not _alive(pid), "the abandoned owner's child survived _reap"


async def test_reap_returns_cleanly_when_the_owner_finishes_normally(stdio_stub_spec):
    """Control case for the test above: an uncancelled `_reap` must still
    return once the owner task finishes on its own, not hang or raise."""
    from agent_core.workers.client_pool import MCPClientPool

    pool = MCPClientPool([stdio_stub_spec("stub", "low")])
    await pool.list_tools("stub")
    pool._stop["stub"].set()                   # let the owner finish on its own
    await pool._reap("stub")                    # must return, not raise/hang
    assert pool._owners.get("stub") is None


async def test_disconnect_from_a_different_task(stdio_stub_spec):
    from agent_core.workers.client_pool import MCPClientPool

    pool = MCPClientPool([stdio_stub_spec("stub", "low")])

    # Connect in THIS task, the way astartup() would.
    result = await pool.list_tools("stub")
    assert {t.name for t in result.tools} == {"noop_low", "risky_high"}
    pid = pool._owner_pid("stub")
    assert pid is not None and _alive(pid)

    # Dispatch from another task must keep working.
    async def dispatch():
        return await pool.call_tool("stub", "noop_low", {"message": "hi"})

    res = await asyncio.create_task(dispatch())
    assert not getattr(res, "isError", False)

    # Disconnect from a DIFFERENT task, the way /worker unload would.
    async def teardown():
        await pool.disconnect("stub")

    await asyncio.create_task(teardown())

    assert not pool.is_connected("stub")
    for _ in range(50):                 # give the child a moment to reap
        if not _alive(pid):
            break
        await asyncio.sleep(0.1)
    assert not _alive(pid), "worker subprocess survived disconnect"


async def test_connect_timeout_leaves_no_residue(stdio_stub_spec):
    """A worker that never completes connect must not wedge the pool."""
    from agent_core.workers.client_pool import MCPClientPool
    from agent_core.workers.types import WorkerSpec

    spec = WorkerSpec(name="slow", transport="streamable_http",
                      risk_default="low", endpoint="http://127.0.0.1:9/mcp")
    pool = MCPClientPool([spec])
    # A refused streamable_http connection surfaces from the mcp SDK's
    # internal anyio task-group cancellation as a bare CancelledError (a
    # BaseException, not an Exception). connect() normalizes that into a
    # ConnectionError (a normal Exception, chained via __cause__) precisely
    # so a single unreachable worker can't cancel whichever task called
    # connect() -- e.g. the daemon's startup task. asyncio.TimeoutError
    # covers the case where the connect attempt genuinely never completes
    # (see test_connect_timeout_kills_stdio_child for that path).
    with pytest.raises((asyncio.TimeoutError, ConnectionError)):
        await pool.connect("slow", timeout=1.0)
    assert not pool.is_connected("slow")
    assert pool._owners.get("slow") is None


async def test_connect_timeout_kills_stdio_child(tmp_path):
    """A stdio worker that hangs mid-connect must not leak its subprocess.

    Unlike test_connect_timeout_leaves_no_residue (a fast refusal that never
    reaches asyncio.wait_for's deadline), this drives the actual timeout ->
    _cancel_owner -> task.cancel() path: the child spawns, writes its own
    pid, and then sleeps forever without ever completing the MCP handshake,
    so client.initialize() blocks until connect()'s timeout fires.

    The child's pid is captured via a file it writes itself, rather than
    through pool internals -- MCPClientPool never publishes the client into
    self._clients until *after* initialize() succeeds, so _owner_pid() would
    see nothing for a worker that's still hanging in connect().
    """
    from agent_core.workers.client_pool import MCPClientPool
    from agent_core.workers.types import WorkerSpec

    pidfile = tmp_path / "child.pid"
    script = (
        "import os, sys\n"
        "with open(sys.argv[1], 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    spec = WorkerSpec(
        name="hang", transport="stdio", risk_default="low",
        command=sys.executable, args=["-c", script, str(pidfile)],
    )
    pool = MCPClientPool([spec])

    with pytest.raises(asyncio.TimeoutError):
        await pool.connect("hang", timeout=1.0)

    for _ in range(50):                 # child writes its pid almost
        if pidfile.exists():            # instantly, well before the 1s
            break                       # timeout -- but poll defensively.
        await asyncio.sleep(0.05)
    assert pidfile.exists(), "stdio child never started"
    pid = int(pidfile.read_text())

    assert not pool.is_connected("hang")
    assert pool._owners.get("hang") is None

    for _ in range(50):                 # give the child a moment to reap
        if not _alive(pid):
            break
        await asyncio.sleep(0.1)
    assert not _alive(pid), "stdio child survived a cancelled connect"
