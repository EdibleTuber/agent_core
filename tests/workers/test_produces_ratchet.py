"""`produces` must ratchet the way the risk tier does.

_tier_highwater is deliberately never cleared by _bump(), because a reload
would otherwise be a downgrade channel. The same argument applies here: if a
reload could turn `artifact` back into `result`, a worker could disable
descriptor validation by reconnecting.
"""
import pytest

from agent_core.workers.artifacts import PRODUCES_META_KEY
from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WorkerSpec


class _Tool:
    def __init__(self, name, produces=None):
        self.name = name
        self.meta = {} if produces is None else {PRODUCES_META_KEY: produces}


class _Listing:
    def __init__(self, tools):
        self.tools = tools


class _Inner(MCPClientPool):
    def __init__(self, specs, listing):
        super().__init__(list(specs))
        self._listing = listing

    async def list_tools(self, worker):
        return self._listing


def _pool(tmp_path, listing):
    spec = WorkerSpec(name="hardware", transport="stdio", command="/bin/true",
                      risk_default="high", artifact_root="/mnt/bench-store")
    return RiskAwareToolPool(
        inner=_Inner([spec], listing), specs={"hardware": spec},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(), audit_log=AuditLog(tmp_path))


async def test_an_undeclared_tool_produces_a_result(tmp_path):
    pool = _pool(tmp_path, _Listing([_Tool("read_uart")]))
    await pool.list_tools("hardware")
    assert pool.produces("hardware", "read_uart") == "result"


async def test_a_declared_tool_produces_an_artifact(tmp_path):
    pool = _pool(tmp_path, _Listing([_Tool("dump_firmware", "artifact")]))
    await pool.list_tools("hardware")
    assert pool.produces("hardware", "dump_firmware") == "artifact"


async def test_artifact_never_downgrades_to_result(tmp_path):
    """The ratchet. A worker that reconnects advertising `result` for a tool
    previously seen as `artifact` must not thereby switch off descriptor
    validation."""
    pool = _pool(tmp_path, _Listing([_Tool("dump_firmware", "artifact")]))
    await pool.list_tools("hardware")
    assert pool.produces("hardware", "dump_firmware") == "artifact"

    pool._inner._listing = _Listing([_Tool("dump_firmware", "result")])
    pool._bump("hardware")
    await pool.list_tools("hardware")
    assert pool.produces("hardware", "dump_firmware") == "artifact", (
        "a reload downgraded produces, which would disable descriptor "
        "validation for that tool")


async def test_an_unrecognised_value_is_treated_as_result_at_dispatch(tmp_path):
    """Build-time conformance rejects it (Task 7). At dispatch the safe
    reading is `result`: an unrecognised value must not be taken as a licence
    to skip validation."""
    pool = _pool(tmp_path, _Listing([_Tool("weird", "ARTIFACT")]))
    await pool.list_tools("hardware")
    assert pool.produces("hardware", "weird") == "result"


async def test_an_unknown_tool_produces_a_result(tmp_path):
    pool = _pool(tmp_path, _Listing([]))
    assert pool.produces("hardware", "never-seen") == "result"
