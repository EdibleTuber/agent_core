"""`loaded` must stop meaning "nothing has told us otherwise".

Under stdio that was free: the daemon owns the child, so a dead worker is not
silent. Over HTTP a sleeping laptop left /worker list reporting `loaded` with
a stale tool count indefinitely.
"""
import asyncio

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
    monkeypatch.setattr("agent_core.tools.builtin.BUILTIN_TOOLS", [])
    monkeypatch.setattr("agent_core.tools.executor.BUILTIN_TOOLS", [])


class _Agent:
    pass


class _Pool(RiskAwareToolPool):
    """Real risk pool, scripted transport underneath."""

    def __init__(self, tmp_path, specs, ping_exc=None):
        super().__init__(inner=MCPClientPool(list(specs)),
                         specs={s.name: s for s in specs},
                         risk_gate=RiskGate(overrides=[]),
                         approval_registry=ToolApprovalRegistry(),
                         audit_log=AuditLog(tmp_path))
        self.ping_exc = ping_exc
        self.pings = 0

    async def ping(self, worker):
        self.pings += 1
        if self.ping_exc is not None:
            raise self.ping_exc
        return object()


def _mgr(tmp_path, specs, **kw):
    reg = WorkerRegistry()
    for s in specs:
        reg.add(s)
    pool = _Pool(tmp_path, specs, kw.pop("ping_exc", None))
    return WorkerManager(reg, pool, ToolExecutor.build(_Agent(), []), **kw), pool


def _http(name="netw"):
    return WorkerSpec(name=name, transport="streamable_http",
                      endpoint="http://100.64.0.7:9101/mcp", risk_default="low",
                      autoload=False)


def _stdio(name="local"):
    return WorkerSpec(name=name, transport="stdio", command="/bin/true",
                      risk_default="low", autoload=False)


async def test_a_worker_whose_host_vanishes_stops_reporting_healthy(tmp_path):
    """The whole point. Before this, `loaded` stayed True forever."""
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec], ping_exc=ConnectionError("no route to host"))
    mgr._loaded["netw"] = ["netw_a", "netw_b"]

    assert await mgr.probe("netw") is False
    st = {s.name: s for s in mgr.status()}["netw"]
    assert st.reachable is False
    assert st.loaded is True, (
        "loaded and reachable answer different questions; conflating them "
        "would silently drop the tools from the model's schema on a blip")
    assert "unreachable at" in st.last_error
    assert "100.64.0.7:9101" in st.last_error, "must name WHICH machine"


async def test_stdio_workers_are_not_probed(tmp_path):
    """The kernel already guarantees it. A timer would spend a request per
    worker per interval to confirm what the process model tells us."""
    spec = _stdio()
    mgr, pool = _mgr(tmp_path, [spec])
    mgr._loaded["local"] = ["local_a"]
    assert await mgr.probe("local") is None
    assert pool.pings == 0
    assert {s.name: s for s in mgr.status()}["local"].reachable is None


async def test_an_unloaded_worker_is_not_probed(tmp_path):
    mgr, pool = _mgr(tmp_path, [_http()])
    assert await mgr.probe("netw") is None
    assert pool.pings == 0


async def test_a_probe_failure_evicts_session_approvals(tmp_path):
    """Same signal as a dropped dispatch, so the same consequence -- or the
    two paths disagree and a link that dropped BETWEEN calls leaves an
    approval standing for a process that may have restarted."""
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec], ping_exc=ConnectionError("reset"))
    mgr._loaded["netw"] = ["netw_flash"]
    gen = pool.generation("netw")
    pool.record_session_approval("netw", "netw_flash", gen)

    await mgr.probe("netw")

    assert not pool.is_session_approved("netw", "netw_flash")
    assert pool.generation("netw") == gen + 1


async def test_recovery_clears_only_a_probe_authored_error(tmp_path):
    """A load-time failure the operator has not acted on is not 'fixed' by
    one good ping."""
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec])
    mgr._loaded["netw"] = ["netw_a"]

    mgr._errors["netw"] = "unreachable at http://x/mcp: ConnectError: nope"
    assert await mgr.probe("netw") is True
    assert mgr._errors.get("netw") is None

    mgr._errors["netw"] = "tool_collision: netw_a already registered"
    assert await mgr.probe("netw") is True
    assert mgr._errors["netw"].startswith("tool_collision")


async def test_a_hanging_probe_cannot_wedge_the_sweep(tmp_path):
    """A probe that can hang is worse than no probe: it would wedge the loop
    that exists to notice hangs."""
    spec = _http()

    async def _hang(worker):
        await asyncio.sleep(30)

    mgr, pool = _mgr(tmp_path, [spec], probe_timeout=0.2)
    pool.ping = _hang
    mgr._loaded["netw"] = ["netw_a"]
    result = await asyncio.wait_for(mgr.probe("netw"), 5)
    assert result is False


async def test_probe_all_surveys_the_fleet_despite_one_bad_worker(tmp_path):
    good, bad = _http("good"), _http("bad")
    mgr, pool = _mgr(tmp_path, [good, bad, _stdio()])
    mgr._loaded.update({"good": ["a"], "bad": ["b"], "local": ["c"]})

    async def _ping(worker):
        if worker == "bad":
            raise ConnectionError("gone")
        return object()

    pool.ping = _ping
    assert await mgr.probe_all() == {"good": True, "bad": False}


async def test_reachability_does_not_outlive_the_connection(tmp_path):
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec], ping_exc=ConnectionError("gone"))
    mgr._loaded["netw"] = ["netw_a"]
    await mgr.probe("netw")
    assert {s.name: s for s in mgr.status()}["netw"].reachable is False

    await mgr.unload("netw")
    assert {s.name: s for s in mgr.status()}["netw"].reachable is None, (
        "a stale False would make /worker list contradict itself after a reload")


async def test_status_names_the_endpoint(tmp_path):
    """With three machines in play, a healthy networked worker was otherwise
    indistinguishable from any other without reading workers.yaml."""
    mgr, _ = _mgr(tmp_path, [_http(), _stdio()])
    by = {s.name: s for s in mgr.status()}
    assert by["netw"].endpoint == "http://100.64.0.7:9101/mcp"
    assert by["local"].endpoint is None


async def test_the_loop_starts_stops_and_survives_a_bad_sweep(tmp_path):
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec], liveness_interval=0.05)
    mgr._loaded["netw"] = ["netw_a"]
    sweeps = []

    async def _ping(worker):
        sweeps.append(worker)
        if len(sweeps) == 1:
            raise ConnectionError("first sweep fails")
        return object()

    pool.ping = _ping
    mgr.start_liveness()
    mgr.start_liveness()          # idempotent
    for _ in range(100):
        if len(sweeps) >= 3:
            break
        await asyncio.sleep(0.02)
    await mgr.stop_liveness()
    assert len(sweeps) >= 3, "a failed sweep killed the loop"
    assert mgr._liveness_task is None


async def test_liveness_can_be_switched_off(tmp_path):
    mgr, _ = _mgr(tmp_path, [_http()], liveness_interval=0)
    mgr.start_liveness()
    assert mgr._liveness_task is None
    await mgr.stop_liveness()


async def test_the_manager_can_actually_resolve_a_target_through_the_risk_pool(tmp_path):
    """Regression: WorkerManager asks the pool it was GIVEN -- the risk pool,
    not the inner client pool. Without a passthrough there, _target() fell
    back to the bare worker name and every message built from it dropped the
    endpoint."""
    spec = _http()
    mgr, pool = _mgr(tmp_path, [spec])
    assert pool.target("netw") == spec.endpoint
    assert mgr._target("netw") == spec.endpoint
