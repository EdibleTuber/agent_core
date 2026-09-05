"""Connection teardown must not be bound to the task that connected.

anyio binds a cancel scope to the entering task. The daemon connects workers
in its startup task (Agent.astartup) but unloads them from a per-message
handler task (daemon.py:103), so a pool that closes in the caller's task
raises RuntimeError -- and, measured, leaves the worker subprocess running
until the event loop exits. Every pre-existing test in test_client_pool.py
connects and closes inside one coroutine, so none of them can catch this.
"""
import asyncio
import os

import pytest


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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
    # BaseException, not an Exception) rather than a socket error -- so the
    # expected-exception tuple must name it explicitly.
    with pytest.raises((asyncio.TimeoutError, OSError, asyncio.CancelledError, Exception)):
        await pool.connect("slow", timeout=1.0)
    assert not pool.is_connected("slow")
    assert pool._owners.get("slow") is None
