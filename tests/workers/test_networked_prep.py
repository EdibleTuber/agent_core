"""Three defects that only show up when a worker is across a network.

Every test here drives a real MCPClient against a real (unreachable) endpoint
rather than a mock: all three bugs live in the SDK's teardown behaviour, which
is exactly what a mock would paper over.
"""
import asyncio
import contextlib
import gc

import httpx
import pytest

from agent_core.tools.executor import ToolExecutor
from agent_core.workers.audit import AuditLog
from agent_core.workers.client import (DEFAULT_HTTP_CONNECT_TIMEOUT,
                                       DEFAULT_HTTP_READ_TIMEOUT, MCPClient)
from agent_core.workers.client_pool import MCPClientPool, describe_failure
from agent_core.workers.manager import WorkerManager
from agent_core.workers.registry import WorkerRegistry
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WorkerSpec

# Port 9 is discard: on this host nothing listens, and a connect to loopback
# is refused immediately rather than dropped, so these stay fast.
DEAD = "http://127.0.0.1:9/mcp"


@pytest.fixture(autouse=True)
def _empty_builtin_tools(monkeypatch):
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
    return WorkerManager(reg, pool, ToolExecutor.build(_Agent(), [])), pool


def _live_http_clients():
    gc.collect()
    return [o for o in gc.get_objects()
            if isinstance(o, httpx.AsyncClient) and not o.is_closed]


# --- (a) the close() leak -------------------------------------------------

async def test_failed_http_connect_leaks_no_http_client():
    """A failed connect must not leave an httpx.AsyncClient alive.

    streamable_http_client only manages the lifecycle of a client it created
    itself, so passing one makes MCPClient its owner. Before the fix, close()
    awaited the session's __aexit__ first and returned early when it raised --
    which is the NORMAL outcome here, because an HTTP client is lazy, so an
    unreachable endpoint fails inside initialize() rather than connect(). The
    transport context then never exited, and the asyncgen was finalised much
    later by the GC in an arbitrary task, producing a 'cancel scope in a
    different task' RuntimeError with a completely different cause from the
    identically-worded bug the owner-task design fixed.
    """
    before = len(_live_http_clients())
    for _ in range(3):
        client = MCPClient(endpoint=DEAD, connect_timeout=3, read_timeout=3)
        try:
            await client.connect()
            await client.initialize()
        except BaseException:
            pass
        finally:
            with pytest.raises(BaseException):
                await client.close()
    assert len(_live_http_clients()) == before


async def test_connect_failure_leaves_no_half_open_state():
    """connect() promises to leave no residue; assert it, don't assume it."""
    client = MCPClient(endpoint=DEAD, connect_timeout=3, read_timeout=3)
    with pytest.raises(BaseException):
        await client.connect()
        await client.initialize()
    with contextlib.suppress(BaseException):
        await client.close()
    assert client._session is None
    assert client._transport_ctx is None
    assert client._http_client is None


# --- (b) two timeouts, not one -------------------------------------------

def test_timeouts_are_two_separate_httpx_parameters():
    """One field cannot bound both halves.

    httpx.Timeout's positional argument covers connect/write/pool; the
    RESPONSE read is a separate parameter defaulting to 300s. A single
    `request_timeout` would have left a slow worker hanging for five minutes
    while appearing to be configured.
    """
    spec = WorkerSpec(name="w", transport="streamable_http", endpoint=DEAD,
                      risk_default="low", connect_timeout=7, read_timeout=11)
    client = MCPClient.from_spec(spec)
    assert client.connect_timeout == 7
    assert client.read_timeout == 11


def test_unset_timeouts_keep_the_sdk_defaults():
    """An existing stdio deployment must not change behaviour under this
    branch, so the fallbacks are the SDK's own numbers, not new ones."""
    spec = WorkerSpec(name="w", transport="streamable_http", endpoint=DEAD,
                      risk_default="low")
    client = MCPClient.from_spec(spec)
    assert client.connect_timeout is None and client.read_timeout is None
    assert (DEFAULT_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT) == (30.0, 300.0)


async def test_read_timeout_reaches_the_session():
    """read_timeout must bound tools/call, which is what client_pool awaits
    unbounded. ClientSession.read_timeout_seconds is the only thing that does
    it for every request, so check it actually arrives there."""
    from datetime import timedelta

    captured = {}
    real_init = __import__("mcp.client.session", fromlist=["ClientSession"]).ClientSession.__init__

    def spy(self, *a, **kw):
        captured.update(kw)
        return real_init(self, *a, **kw)

    import mcp.client.session as sess
    sess.ClientSession.__init__ = spy
    try:
        client = MCPClient(endpoint=DEAD, connect_timeout=2, read_timeout=5)
        try:
            await client.connect()
        except BaseException:
            pass
        finally:
            with contextlib.suppress(BaseException):
                await client.close()
    finally:
        sess.ClientSession.__init__ = real_init
    assert captured.get("read_timeout_seconds") == timedelta(seconds=5)


async def test_manager_prefers_the_spec_connect_timeout(tmp_path, monkeypatch):
    """A tailnet worker legitimately needs longer than one on this box.
    Before this, the pool's hard-coded 10s always won and the operator's
    number was silently ignored."""
    spec = WorkerSpec(name="w", transport="streamable_http", endpoint=DEAD,
                      risk_default="low", connect_timeout=17, autoload=False)
    mgr, pool = _manager(tmp_path, [spec])
    seen = {}

    async def fake_connect(worker, timeout=None):
        seen["timeout"] = timeout
        raise ConnectionError("nope")

    monkeypatch.setattr(pool, "connect", fake_connect)
    await mgr.load("w")
    assert seen["timeout"] == 17
    await pool.close_all()


# --- (c) saying which machine, and why ------------------------------------

def test_describe_failure_drops_the_cancellations_around_the_real_cause():
    """anyio task groups cancel siblings, so the real error arrives inside a
    group that is mostly CancelledError. repr() of that group tells the
    operator nothing about which host refused what."""
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [
        asyncio.CancelledError("Cancelled via cancel scope 0x7f00"),
        httpx.ConnectError("All connection attempts failed"),
        asyncio.CancelledError("Cancelled via cancel scope 0x7f01"),
    ])
    described = describe_failure(group)
    assert described == "ConnectError: All connection attempts failed"


def test_describe_failure_collapses_identical_repeats():
    group = BaseExceptionGroup("g", [httpx.ConnectError("refused")] * 4)
    assert describe_failure(group) == "ConnectError: refused"


def test_describe_failure_falls_back_when_only_cancellations_remain():
    """Dropping every leaf would leave the operator with an empty string."""
    group = BaseExceptionGroup("g", [asyncio.CancelledError("scope 0x1")])
    assert "CancelledError" in describe_failure(group)


async def test_unreachable_http_worker_is_not_spawn_failed(tmp_path):
    """We never spawned it, so it cannot have failed to spawn.

    The two need different actions from the operator -- check the binary
    here, versus check the host there -- so they get different names.
    """
    spec = WorkerSpec(name="ghost", transport="streamable_http", endpoint=DEAD,
                      risk_default="low", connect_timeout=5, read_timeout=5,
                      autoload=False)
    mgr, pool = _manager(tmp_path, [spec])
    res = await mgr.load("ghost")
    assert not res.ok
    assert res.error_kind == "unreachable"
    await pool.close_all()


async def test_unreachable_error_names_the_endpoint_and_the_cause(tmp_path):
    """The measured 'before' was:

        ConnectionError: connecting to worker 'frida' failed
          (cancelled internally: CancelledError('Cancelled via cancel scope
          0x7e671b0f3a70'))

    -- no endpoint, no cause, and a memory address. With three machines in
    play the endpoint IS the diagnosis.
    """
    spec = WorkerSpec(name="ghost", transport="streamable_http", endpoint=DEAD,
                      risk_default="low", connect_timeout=5, read_timeout=5,
                      autoload=False)
    mgr, pool = _manager(tmp_path, [spec])
    res = await mgr.load("ghost")
    status = {s.name: s for s in mgr.status()}["ghost"]
    for text in (res.error, status.last_error):
        assert "127.0.0.1:9" in text, text
        assert "cancel scope" not in text, text
        assert "ConnectError" in text or "ConnectionError" in text, text
    await pool.close_all()


async def test_stdio_worker_with_a_missing_binary_is_still_spawn_failed(tmp_path):
    """The new ErrorKind must not swallow the case it was split away from."""
    spec = WorkerSpec(name="gone", transport="stdio",
                      command="/nonexistent/worker-binary",
                      risk_default="low", autoload=False)
    mgr, pool = _manager(tmp_path, [spec])
    res = await mgr.load("gone")
    assert not res.ok
    assert res.error_kind == "spawn_failed"
    await pool.close_all()


def test_target_names_the_endpoint_for_http_and_the_command_for_stdio():
    pool = MCPClientPool([
        WorkerSpec(name="net", transport="streamable_http", endpoint=DEAD,
                   risk_default="low"),
        WorkerSpec(name="local", transport="stdio", command="/usr/bin/thing",
                   args=["--serve"], risk_default="low"),
    ])
    assert pool.target("net") == DEAD
    assert pool.target("local") == "/usr/bin/thing --serve"
    assert pool.target("unknown") == "unknown"


# --- diagnosis: an error message that says nothing costs real time ---------

def test_describe_failure_never_returns_an_empty_string():
    """`worker stub discovery failed ()` is a real line from a real CI run.

    asyncio.wait_for raises TimeoutError, whose str() is empty, and the log
    interpolated it with %s. An operator staring at a red run learned nothing
    about whether the worker was slow, unreachable, or erroring.
    """
    for exc in (asyncio.TimeoutError(), TimeoutError(), ValueError(),
                RuntimeError(""), asyncio.CancelledError()):
        described = describe_failure(exc)
        assert described.strip(), f"{type(exc).__name__} produced {described!r}"
        assert type(exc).__name__ in described


def test_conformance_distinguishes_a_cancellation_from_a_timeout():
    """They have different causes and different fixes.

    An MCP transport whose background task dies cancels its caller, so a
    SERVER-side failure arrives as a bare CancelledError. Reporting that as
    'initialize timed out' points the reader at latency when the real problem
    is a handler that returned without completing its response.
    """
    import inspect

    from agent_core.workers import conformance

    src = inspect.getsource(conformance.assert_streamable_http_conformance)
    assert "except (asyncio.TimeoutError, asyncio.CancelledError)" not in src, (
        "the two are collapsed again; a cancellation would be reported as a timeout")
    assert "was cancelled" in src and "exceeded" in src


def test_the_conformance_bound_is_not_a_benchmark():
    """A bound tight enough to double as a performance assertion fails for
    reasons unrelated to what it guards. Locally an initialize measures
    ~0.005s; the old 2.0s bound still failed on a shared CI runner."""
    from agent_core.workers.conformance import CONFORMANCE_TIMEOUT
    from agent_core.workers.discovery import DISCOVERY_TIMEOUT

    assert CONFORMANCE_TIMEOUT >= 10
    assert DISCOVERY_TIMEOUT >= 10
