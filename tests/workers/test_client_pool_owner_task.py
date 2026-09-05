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
import sys

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
