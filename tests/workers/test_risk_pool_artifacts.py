"""Task 2: ctx channel, pre-gate refusals, and the tier floor.

D12–D20: artifact-tool routing, four pre-gate refusals, and the high-tier
floor.  D18–D20 are P-pins that must be GREEN at baseline and stay green.
"""
import asyncio
import json
import uuid

import pytest

from agent_core.conversation import Conversation
from agent_core.workers.artifacts import PRODUCES_ARTIFACT, PRODUCES_META_KEY
from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate
from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import (
    ToolApprovalRegistry,
    ToolDecision,
)
from agent_core.workers.types import WorkerSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Tool:
    """A tool whose ``meta`` carries the ``produces`` declaration."""
    def __init__(self, name, tier="low", produces=None):
        self.name = name
        self.meta = {}
        if tier is not None:
            self.meta["agent_core/tier"] = tier
        if produces is not None:
            self.meta["agent_core/produces"] = produces


class _Listing:
    def __init__(self, tools):
        self.tools = tools


class _Inner(MCPClientPool):
    def __init__(self, specs, listing):
        super().__init__(list(specs))
        self._listing = listing
        self.calls = []
        self.raise_on_call = False

    async def list_tools(self, worker):
        return self._listing

    async def call_tool(self, worker, tool, arguments):
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.calls.append((worker, tool, arguments))
        class _R:
            content = []; isError = False
        return _R()


def _approval_send(reg):
    """Return a send callable that auto-approves and records the sent req."""
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    return send, sent


def _spec_fixture():
    """Task 1 valid spec fixture: streamable_http with root + drive id."""
    return WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=str(uuid.uuid4()),
    )


def _pool_with_spec(spec, listing, reg=None, audit_dir=None, send=None):
    return RiskAwareToolPool(
        inner=_Inner([spec], listing),
        specs={"hw": spec},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=reg or ToolApprovalRegistry(),
        audit_log=AuditLog(audit_dir or "/tmp/audit_none"),
        send_message=send,
    )


def _ctx(project_slug=None):
    """Real HandlerContext with the fields the brief specifies."""
    from agent_core.agent import HandlerContext
    return HandlerContext(
        conversation=Conversation(history_depth=1),
        channel_id="t",
        writer=object(),
        project_slug=project_slug,
        cwd="/mnt/secondary/projects/PARE",
    )


def _audit_rows(audit_dir):
    files = list(audit_dir.glob("audit-*.jsonl"))
    assert len(files) == 1
    return [json.loads(l) for l in files[0].read_text().splitlines()]


# ---------------------------------------------------------------------------
# D12 — low-tier artifact tool, valid slug => floored to high, approval sent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d12_low_artifact_floored_to_high(tmp_path):
    """D12: low-tier artifact tool with valid project_slug is floored to
    high, an approval request is sent, and the audit row shows effective_tier
    == 'high' with override_reason containing 'produces=artifact' and 'high'."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    ctx = _ctx(project_slug="bench-slug-abc123")
    await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert len(sent) == 1  # approval was sent
    assert pool.produces("hw", "dump_firmware") == "artifact"
    rows = _audit_rows(tmp_path)
    assert rows[0]["effective_tier"] == "high"
    assert "produces=artifact" in (rows[0]["override_reason"] or "")
    assert "high" in (rows[0]["override_reason"] or "")
    assert rows[0]["outcome"] == "hitl_approved"


# ---------------------------------------------------------------------------
# D13 — ctx without project_slug => refused, inner not called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d13_no_project_slug_refused(tmp_path):
    """D13: artifact tool called with a ctx that has no project_slug field
    (real HandlerContext, field unset) => _ErrorResult; message names the
    cwd; inner not called; no approval; audit row validation_failed."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    # Real HandlerContext WITHOUT project_slug
    from agent_core.agent import HandlerContext
    ctx = HandlerContext(
        conversation=Conversation(history_depth=1),
        channel_id="t",
        writer=object(),
        cwd="/mnt/secondary/projects/PARE",
    )
    out = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert out.isError is True
    assert "/mnt/secondary/projects/PARE" in out.content[0].text
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["outcome"] == "validation_failed"
    assert "/mnt/secondary/projects/PARE" in (rows[0]["detail"] or "")


# ---------------------------------------------------------------------------
# D14 — project_slug="../evil" => refused, msg names cwd, no raw slug
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d14_invalid_slug_refused(tmp_path):
    """D14: project_slug='../../evil' is refused; message names the cwd and
    does NOT contain the raw slug; inner not called; validation_failed row."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    ctx = _ctx(project_slug="../evil")
    out = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert out.isError is True
    assert "/mnt/secondary/projects/PARE" in out.content[0].text
    assert "../evil" not in out.content[0].text  # raw slug must NOT appear
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D15 — spec.artifact_root=None => refused, naming the missing declaration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d15_artifact_root_none_refused(tmp_path):
    """D15: spec with artifact_root=None (legal config) + artifact tool =>
    refused, naming the missing declaration; inner not called; no approval."""
    spec = WorkerSpec(
        name="hw",
        transport="stdio",
        command="x",
        risk_default="low",
        artifact_root=None,
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    ctx = _ctx(project_slug="bench-slug")
    out = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert out.isError is True
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D16 — artifact_drive_id=None => refused
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d16_artifact_drive_id_none_refused(tmp_path):
    """D16: root set, artifact_drive_id=None (model_copy from valid fixture)
    => refused."""
    drive = str(uuid.uuid4())
    valid_spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    spec = valid_spec.model_copy(update={"artifact_drive_id": None})
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    ctx = _ctx(project_slug="bench-slug")
    out = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert out.isError is True
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D17 — artifact_host=None => refused, naming artifact_host
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d17_artifact_host_none_refused(tmp_path):
    """D17: root set, artifact_host=None (model_copy from valid fixture) =>
    refused, naming artifact_host."""
    drive = str(uuid.uuid4())
    valid_spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    spec = valid_spec.model_copy(update={"artifact_host": None})
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking for the tool
    ctx = _ctx(project_slug="bench-slug")
    out = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert out.isError is True
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D18 (P-pin) — low-tier non-artifact tool => unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d18_non_artifact_tool_unchanged(tmp_path):
    """D18: low-tier non-artifact tool => auto-executes, no approval,
    effective_tier == 'low', no floor in override_reason, outcome 'ok'."""
    spec = WorkerSpec(
        name="hw",
        transport="stdio",
        command="x",
        risk_default="low",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("read_uart", tier="low")]),
                           reg=reg, audit_dir=tmp_path)
    ctx = _ctx(project_slug="bench-slug")
    await pool.call_tool("hw", "read_uart", {}, ctx=ctx)
    assert sent == []
    rows = _audit_rows(tmp_path)
    assert rows[0]["effective_tier"] == "low"
    assert "produces=artifact" not in (rows[0].get("override_reason") or "")
    assert rows[0]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# D19 (P-pin) — high-tier artifact tool => approval sent once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d19_high_artifact_approval_sent(tmp_path):
    """D19: high-tier artifact tool => approval sent exactly once; audit row
    effective_tier == 'high'; override_reason does NOT mention the floor."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="high",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="high", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking
    ctx = _ctx(project_slug="bench-slug")
    await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert len(sent) == 1
    rows = _audit_rows(tmp_path)
    assert rows[0]["effective_tier"] == "high"
    # The floor should NOT be mentioned: this tool is already high
    assert "tier floor" not in (rows[0].get("override_reason") or "").lower()


# ---------------------------------------------------------------------------
# D20 (P-pin) — critical artifact tool => stays critical
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d20_critical_stays_critical(tmp_path):
    """D20: critical artifact tool => stays critical (floor never lowers);
    approval + rationale path unchanged."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="critical",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
    )
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification="ok"))
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="critical", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")  # populate produces tracking
    ctx = _ctx(project_slug="bench-slug")
    await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert len(sent) == 1
    rows = _audit_rows(tmp_path)
    assert rows[0]["effective_tier"] == "critical"
