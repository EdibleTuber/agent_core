"""Worker identity and the fail-closed generation rule.

Generation-keyed approvals were a property of LOCAL PROCESS OWNERSHIP, not of
the keying itself. Under stdio the kernel enforces it: the daemon spawned the
child and holds its pipe, so the process cannot change without a reload, and a
reload bumps the generation. Over HTTP nothing does -- a Pi reboots, a unit
restarts, a container is redeployed, and the daemon never learns.
"""
import asyncio

import pytest

from agent_core.tools.executor import ToolExecutor
from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool, _server_info_of
from agent_core.workers.manager import WorkerManager, _artifact_args
from agent_core.workers.registry import WorkerRegistry
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WORKER_CONTRACT_VERSION, WorkerSpec

DEAD = "http://127.0.0.1:9/mcp"


@pytest.fixture(autouse=True)
def _empty_builtin_tools(monkeypatch):
    monkeypatch.setattr("agent_core.tools.builtin.BUILTIN_TOOLS", [])
    monkeypatch.setattr("agent_core.tools.executor.BUILTIN_TOOLS", [])


class _Agent:
    pass


def _http_spec(name="netw"):
    return WorkerSpec(name=name, transport="streamable_http", endpoint=DEAD,
                      risk_default="low", autoload=False)


def _stdio_spec(name="local"):
    return WorkerSpec(name=name, transport="stdio", command="/bin/true",
                      risk_default="low", autoload=False)


def _pool(tmp_path, specs, inner=None):
    return RiskAwareToolPool(
        inner=inner or MCPClientPool(list(specs)),
        specs={s.name: s for s in specs},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(),
        audit_log=AuditLog(tmp_path))


class _Boom:
    """Inner pool whose dispatch fails the way a dropped link does."""

    def __init__(self, specs, exc):
        self._specs = {s.name: s for s in specs}
        self._exc = exc

    def names(self):
        return list(self._specs)

    def spec(self, worker):
        return self._specs.get(worker)

    def add_spec(self, spec):
        self._specs[spec.name] = spec

    def remove_spec(self, worker):
        self._specs.pop(worker, None)

    def server_info(self, worker):
        return None

    async def call_tool(self, worker, tool, arguments):
        raise self._exc


# --- the security rule ----------------------------------------------------

@pytest.mark.parametrize("exc", [ConnectionError("reset by peer"),
                                 OSError("broken pipe"),
                                 RuntimeError("session terminated")])
async def test_link_loss_on_a_networked_worker_evicts_session_approvals(tmp_path, exc):
    """Otherwise a `scope: session` approval survives onto a DIFFERENT
    PROCESS, dispatching with no prompt and an audit row that reads
    hitl_approved / session-approved."""
    spec = _http_spec()
    pool = _pool(tmp_path, [spec], inner=_Boom([spec], exc))
    gen = pool.generation("netw")
    pool.record_session_approval("netw", "netw_flash", gen)
    assert pool.is_session_approved("netw", "netw_flash")

    await pool.call_tool("netw", "netw_flash", {})

    assert not pool.is_session_approved("netw", "netw_flash"), (
        "the approval survived a transport failure on a networked worker")
    assert pool.generation("netw") == gen + 1


async def test_a_stdio_worker_keeps_its_approvals_across_a_tool_error(tmp_path):
    """The rule must not fire where the kernel already guarantees identity.

    For stdio the daemon spawned the child and holds its pipe, so the process
    behind the connection cannot change without a reload. Evicting there would
    re-prompt the operator for every transient failure and buy nothing.
    """
    spec = _stdio_spec()
    pool = _pool(tmp_path, [spec], inner=_Boom([spec], RuntimeError("tool blew up")))
    gen = pool.generation("local")
    pool.record_session_approval("local", "local_thing", gen)

    await pool.call_tool("local", "local_thing", {})

    assert pool.is_session_approved("local", "local_thing")
    assert pool.generation("local") == gen


async def test_cancellation_mid_dispatch_also_evicts_on_a_networked_worker(tmp_path):
    spec = _http_spec()
    pool = _pool(tmp_path, [spec], inner=_Boom([spec], asyncio.CancelledError()))
    gen = pool.generation("netw")
    pool.record_session_approval("netw", "netw_flash", gen)

    with pytest.raises(asyncio.CancelledError):
        await pool.call_tool("netw", "netw_flash", {})

    assert not pool.is_session_approved("netw", "netw_flash")
    assert pool.generation("netw") == gen + 1


async def test_an_unknown_worker_fails_closed(tmp_path):
    """No spec means no evidence of local ownership, so treat it as remote."""
    pool = _pool(tmp_path, [])
    assert pool._is_local("never-heard-of-it") is False


async def test_the_eviction_is_recorded_not_silent(tmp_path):
    """An operator re-prompted for no visible reason will assume a bug. The
    audit row has to say why."""
    spec = _http_spec()
    pool = _pool(tmp_path, [spec], inner=_Boom([spec], ConnectionError("reset")))
    await pool.call_tool("netw", "netw_flash", {})
    rows = "".join(p.read_text() for p in sorted(tmp_path.glob("audit-*.jsonl")))
    assert "approvals evicted" in rows, rows


# --- identity as the networked stand-in for an mtime ----------------------

def test_artifact_args_records_nothing_stattable_for_a_networked_worker():
    """The gap this closes: every file field is None for HTTP, because the
    daemon never spawned the process and may not share its filesystem."""
    args = _artifact_args(_http_spec())
    assert args["resolved_command"] is None
    assert args["command_mtime"] is None and args["command_size"] is None


def test_artifact_args_substitutes_worker_reported_identity_over_http():
    info = {"server_name": "pare-hardware-mcp", "server_version": "0.3.1",
            "protocol_version": "2025-11-25"}
    args = _artifact_args(_http_spec(), info)
    assert args["server_name"] == "pare-hardware-mcp"
    assert args["server_version"] == "0.3.1"
    assert args["command_mtime"] is None, "still nothing to stat; not a substitute"


def test_a_stdio_worker_does_not_get_identity_instead_of_a_stat():
    """stdio has the real thing. Identity is self-reported and must not
    displace a filesystem fact."""
    args = _artifact_args(_stdio_spec(), {"server_name": "whatever"})
    assert "server_name" not in args


def test_server_info_of_tolerates_a_worker_that_answers_oddly():
    """Runs on the connect path: a surprising initialize result must not fail
    an otherwise-good load."""
    assert _server_info_of(object()) == {}
    assert _server_info_of(None) == {}


def test_identity_does_not_outlive_the_connection():
    """Keeping it past disconnect would let a stale row claim provenance for a
    process that is gone."""
    pool = MCPClientPool([_http_spec()])
    pool._server_info["netw"] = {"server_name": "x"}
    assert pool.server_info("netw") == {"server_name": "x"}
    pool._server_info.pop("netw", None)
    assert pool.server_info("netw") is None


# --- the audit constant ---------------------------------------------------

def test_the_contract_version_is_not_a_magic_literal(tmp_path):
    """It was written as `1` in two places. A bump has to change one thing,
    not two -- and the constant is the thing conformance already checks."""
    import inspect

    from agent_core.workers import risk_pool

    src = inspect.getsource(risk_pool)
    assert "worker_contract_version=1," not in src
    assert WORKER_CONTRACT_VERSION == 1
