# tests/workers/test_risk_pool_capture.py
"""Test that RiskAwareToolPool optionally routes successful results through CaptureLayer."""
import json

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
    def __init__(self, text, *, is_error=False):
        self.isError = is_error
        self.content = [_Block(text)]


class _SpecBookkeeping:
    """Minimal MCPClientPool.spec()/add_spec()/remove_spec()/names() surface,
    since RiskAwareToolPool now reads specs through to the inner pool."""
    def __init__(self):
        self._specs = {}

    def add_spec(self, spec):
        self._specs[spec.name] = spec

    def remove_spec(self, name):
        self._specs.pop(name, None)

    def spec(self, name):
        return self._specs.get(name)

    def names(self):
        return list(self._specs)


class _InnerPool(_SpecBookkeeping):
    async def call_tool(self, worker, tool, arguments):
        # Returns a list payload large enough to exceed a 100-byte inline_budget.
        return _Result(json.dumps([{"hex": "ab" * 2000}]))


class _ErrorInnerPool(_SpecBookkeeping):
    async def call_tool(self, worker, tool, arguments):
        return _Result("tool execution failed", is_error=True)


def _low_spec(name: str) -> WorkerSpec:
    return WorkerSpec(name=name, transport="stdio", command="x", risk_default="low")


def _pool(layer, tmp_path, inner=None):
    """Build a low-risk RiskAwareToolPool (no approval gate needed)."""
    spec = _low_spec("frida")
    return RiskAwareToolPool(
        inner=inner or _InnerPool(),
        specs={spec.name: spec},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=ToolApprovalRegistry(),
        audit_log=AuditLog(tmp_path),
        capture_layer=layer,
    )


@pytest.mark.asyncio
async def test_model_path_receives_stub(tmp_path):
    """Model path (capture=True, the default) gets a stub for an oversized result."""
    layer = CaptureLayer(CaptureStore.open_memory(), inline_budget=100, launch_ts=1.0)
    pool = _pool(layer, tmp_path)
    result = await pool.call_tool("frida", "read_memory", {}, ctx=None)
    doc = json.loads(stringify_result(result))
    assert "captured" in doc


@pytest.mark.asyncio
async def test_operator_path_receives_full_payload(tmp_path):
    """Operator path (capture=False) sees the real payload, but the record is still stored."""
    layer = CaptureLayer(CaptureStore.open_memory(), inline_budget=100, launch_ts=1.0)
    pool = _pool(layer, tmp_path)
    result = await pool.call_tool("frida", "enumerate_processes", {}, ctx=None, capture=False)
    assert "ab" * 2000 in stringify_result(result)  # not substituted
    assert len(layer.store.recent()) == 1            # but stored


@pytest.mark.asyncio
async def test_capture_layer_none_is_passthrough(tmp_path):
    """When capture_layer=None the inner result is returned untouched (existing agents unchanged)."""
    pool = _pool(None, tmp_path)
    result = await pool.call_tool("frida", "read_memory", {}, ctx=None)
    assert result.isError is False
    assert "ab" * 2000 in stringify_result(result)


@pytest.mark.asyncio
async def test_error_result_captured_but_never_substituted(tmp_path):
    """Error results ARE stored (a failed run stays searchable/reviewable) but
    are passed through verbatim — never swapped for a stub — so the model sees
    the failure."""
    layer = CaptureLayer(CaptureStore.open_memory(), inline_budget=100, launch_ts=1.0)
    pool = _pool(layer, tmp_path, inner=_ErrorInnerPool())
    result = await pool.call_tool("frida", "read_memory", {}, ctx=None)
    assert result.isError is True
    assert len(layer.store.recent()) == 1
