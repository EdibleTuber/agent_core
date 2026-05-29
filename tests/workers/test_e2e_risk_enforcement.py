# tests/workers/test_e2e_risk_enforcement.py
import asyncio
import json
import pytest

from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry, ToolDecision
from agent_core.workers.risk import RiskGate
from agent_core.workers.audit import AuditLog


def _make(specs, reg, audit_dir, send):
    return RiskAwareToolPool(
        inner=MCPClientPool(specs), specs={s.name: s for s in specs},
        risk_gate=RiskGate(overrides=[]), approval_registry=reg,
        audit_log=AuditLog(audit_dir), send_message=send,
    )


def _audit_rows(audit_dir):
    files = list(audit_dir.glob("audit-*.jsonl"))
    assert len(files) == 1
    return [json.loads(l) for l in files[0].read_text().splitlines()]


@pytest.mark.asyncio
async def test_low_executes_no_prompt(stdio_stub_spec, tmp_path):
    spec = stdio_stub_spec("stub", "low")
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m): sent.append(m)
    pool = _make([spec], reg, tmp_path, send)
    try:
        out = await pool.call_tool("stub", "noop_low", {})
    finally:
        await pool.close_all()
    assert sent == []
    assert not getattr(out, "isError", False)
    assert _audit_rows(tmp_path)[0]["outcome"] == "ok"


@pytest.mark.asyncio
async def test_high_approved_executes(stdio_stub_spec, tmp_path):
    spec = stdio_stub_spec("stub", "high")
    reg = ToolApprovalRegistry()
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    pool = _make([spec], reg, tmp_path, send)
    try:
        out = await pool.call_tool("stub", "risky_high", {"target": "com.example"})
    finally:
        await pool.close_all()
    assert not getattr(out, "isError", False)
    assert _audit_rows(tmp_path)[0]["outcome"] == "hitl_approved"


@pytest.mark.asyncio
async def test_high_denied_blocks(stdio_stub_spec, tmp_path):
    spec = stdio_stub_spec("stub", "high")
    reg = ToolApprovalRegistry()
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=False, justification="no"))
    pool = _make([spec], reg, tmp_path, send)
    try:
        out = await pool.call_tool("stub", "risky_high", {"target": "com.example"})
    finally:
        await pool.close_all()
    assert getattr(out, "isError", False)
    assert _audit_rows(tmp_path)[0]["outcome"] == "hitl_denied"


@pytest.mark.asyncio
async def test_high_timeout_blocks(stdio_stub_spec, tmp_path):
    spec = stdio_stub_spec("stub", "high")
    reg = ToolApprovalRegistry(default_timeout_seconds=0.2)
    async def send(m): pass  # never resolves
    pool = _make([spec], reg, tmp_path, send)
    try:
        out = await pool.call_tool("stub", "risky_high", {"target": "com.example"})
    finally:
        await pool.close_all()
    assert getattr(out, "isError", False)
    assert _audit_rows(tmp_path)[0]["outcome"] == "hitl_denied"
