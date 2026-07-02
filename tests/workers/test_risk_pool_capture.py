# tests/workers/test_risk_pool_capture.py
"""Test that RiskAwareToolPool optionally routes successful results through CaptureLayer."""
import json
import tempfile

import pytest

from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.risk import RiskGate
from agent_core.workers.audit import AuditLog
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.workers.types import WorkerSpec
from agent_core.capture.store import CaptureStore
from agent_core.capture.layer import CaptureLayer, stringify_result


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, text):
        self.isError = False
        self.content = [_Block(text)]


class _InnerPool:
    async def call_tool(self, worker, tool, arguments):
        # Returns a list payload large enough to exceed a 100-byte inline_budget.
        return _Result(json.dumps([{"hex": "ab" * 2000}]))


def _low_spec(name: str) -> WorkerSpec:
    return WorkerSpec(name=name, transport="stdio", command="x", risk_default="low")


def _pool(layer):
    """Build a low-risk RiskAwareToolPool (no approval gate needed)."""
    tmp = tempfile.mkdtemp()
    spec = _low_spec("frida")
    return RiskAwareToolPool(
        inner=_InnerPool(),
        specs={spec.name: spec},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(),
        audit_log=AuditLog(tmp),
        capture_layer=layer,
    )


@pytest.mark.asyncio
async def test_model_path_receives_stub():
    """Model path (capture=True, the default) gets a stub for an oversized result."""
    layer = CaptureLayer(CaptureStore.open_memory(), inline_budget=100, launch_ts=1.0)
    pool = _pool(layer)
    result = await pool.call_tool("frida", "read_memory", {}, ctx=None)
    doc = json.loads(stringify_result(result))
    assert "captured" in doc


@pytest.mark.asyncio
async def test_operator_path_receives_full_payload():
    """Operator path (capture=False) sees the real payload, but the record is still stored."""
    layer = CaptureLayer(CaptureStore.open_memory(), inline_budget=100, launch_ts=1.0)
    pool = _pool(layer)
    result = await pool.call_tool("frida", "enumerate_processes", {}, ctx=None, capture=False)
    assert "ab" * 2000 in stringify_result(result)  # not substituted
    assert len(layer.store.recent()) == 1            # but stored
