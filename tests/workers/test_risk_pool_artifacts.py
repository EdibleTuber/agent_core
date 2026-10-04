"""Artifact routing in the tool pool, tasks 2 and 3.

D12–D20: ctx channel, pre-gate refusals, and the tier floor.
D21–D35: injection, extraction, validation, reconciliation, handoff.

D18–D20, D22, and D31–D35 are P-pins that must be GREEN at baseline and
stay green.
"""
import asyncio
import json
import uuid

import pytest

from agent_core.conversation import Conversation
from agent_core.workers.artifacts import (
    PRODUCES_ARTIFACT, PRODUCES_META_KEY,
    ARTIFACT_DESCRIPTOR_FIELDS,
)
from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate, RISK_TIER_META_KEY
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
            self.meta[RISK_TIER_META_KEY] = tier
        if produces is not None:
            self.meta[PRODUCES_META_KEY] = produces


class _Listing:
    def __init__(self, tools):
        self.tools = tools


class _Inner(MCPClientPool):
    def __init__(self, specs, listing):
        super().__init__(list(specs))
        self._listing = listing
        self.calls = []
        self.raise_on_call = False
        self.return_error = False

    async def list_tools(self, worker):
        return self._listing

    async def call_tool(self, worker, tool, arguments):
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.calls.append((worker, tool, arguments))
        class _R:
            content = []; isError = False
        if self.return_error:
            _R.isError = True
            _R.content = [{"text": "boom"}]
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


def _pool_with_spec(spec, listing, reg=None, audit_dir=None, send=None, capture=None):
    return RiskAwareToolPool(
        inner=_Inner([spec], listing),
        specs={"hw": spec},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=reg or ToolApprovalRegistry(),
        audit_log=AuditLog(audit_dir or "/tmp/audit_none"),
        send_message=send,
        capture_layer=capture,
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


def _valid_descriptor(drive_id):
    """Build a fully valid descriptor for the test spec."""
    return {
        "path": "/mnt/bench-store/bench-slug-abc123/fw.bin",
        "size": 1048576,
        "sha256": "a" * 64,
        "hashed_at": "2026-01-01T00:00:00Z",
        "media_type": "application/octet-stream",
        "drive_id": drive_id,
        "host": "100.97.133.126",
    }


def _fake_result(text):
    """Build a result object with exactly one text content block."""
    class _R:
        content = [type("_B", (), {"type": "text", "text": text})()]
        isError = False
    return _R()


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
    assert "unavailable" in out.content[0].text
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
    assert "artifact_root is not declared" in out.content[0].text
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
    assert "artifact_drive_id" in out.content[0].text
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
    assert "artifact_host" in out.content[0].text
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
    assert rows[0]["outcome"] == "hitl_approved"
    # The floor should NOT be mentioned: this tool is already high
    assert "tier floor" not in (rows[0].get("override_reason") or "").lower()


# ---------------------------------------------------------------------------
# D20 (P-pin) — critical artifact tool => stays critical
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d20_critical_stays_critical(tmp_path):
    """D20: wire tier (critical) escalates above the spec default (high);
    the floor neither adds nor changes it — critical stays critical.
    Approval + rationale path unchanged."""
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
    assert rows[0]["declared_tier"] == "critical"
    assert rows[0]["effective_tier"] == "critical"
    assert rows[0]["outcome"] == "hitl_approved"
    assert "tier floor" not in (rows[0].get("override_reason") or "").lower()
# ---------------------------------------------------------------------------
# D21 — injection: model's forged values overwritten by real ones
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d21_injection_overwrites_forged(tmp_path):
    """D21: model sends forged project_slug and expected_drive_id;
    the injection overwrites them with the real values from spec/ctx.
    The inner receives the real values, not the forged ones.
    The approval snapshot contains the injected values."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    forged = {
        "project_slug": "model-forged",
        "expected_drive_id": "model-forged",
        "size": "2g",
    }
    await pool.call_tool("hw", "dump_firmware", forged, ctx=ctx)
    # The inner receives the live dict — injection overwrote the forged values
    _, _, inner_args = pool._inner.calls[-1]
    assert inner_args["project_slug"] == "bench-slug-abc123"
    assert inner_args["expected_drive_id"] == drive
    # The approval snapshot (first sent message) carries the injected values
    approval_msg = sent[0]
    assert approval_msg.arguments["project_slug"] == "bench-slug-abc123"
    assert approval_msg.arguments["expected_drive_id"] == drive


# ---------------------------------------------------------------------------
# D22 (P-pin, identity) — inner receives the same dict object
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d22_injection_in_place_identity(tmp_path):
    """D22: the inner receives the exact same dict object (id(...))
    that was passed to call_tool — in-place mutation, no rebuild."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    args = {"size": "2g"}
    args_id = id(args)
    await pool.call_tool("hw", "dump_firmware", args, ctx=ctx)
    _, _, inner_args = pool._inner.calls[-1]
    assert id(inner_args) == args_id  # same object, in-place mutation


# ---------------------------------------------------------------------------
# D23 — non-dict arguments on artifact tool => TypeError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d23_non_dict_arguments_raises_typeerror(tmp_path):
    """D23: non-dict arguments (a list) on an artifact tool =>
    pytest.raises(TypeError) from call_tool."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    with pytest.raises(TypeError):
        await pool.call_tool("hw", "dump_firmware", ["not", "a", "dict"], ctx=ctx)


# ---------------------------------------------------------------------------
# D24 — happy path: valid descriptor => handoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d24_happy_path_handoff(tmp_path):
    """D24: inner returns exactly one text block with a valid descriptor
    => result passes through verbatim; ctx.artifact_descriptor is set
    with 8 keys in ARTIFACT_DESCRIPTOR_FIELDS order + produced_by."""
    drive = str(uuid.uuid4())
    desc = _valid_descriptor(drive)
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    async def _stub(*a, **kw): return _fake_result(json.dumps(desc))
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    # Result passes through verbatim (not an error)
    assert not getattr(result, "isError", False)
    # ctx.artifact_descriptor is set
    assert ctx.artifact_descriptor is not None
    # 8 keys: 7 from ARTIFACT_DESCRIPTOR_FIELDS + produced_by
    assert set(ctx.artifact_descriptor.keys()) == set(ARTIFACT_DESCRIPTOR_FIELDS) | {"produced_by"}
    assert len(ctx.artifact_descriptor) == 8
    # host reconciled to spec's artifact_host
    assert ctx.artifact_descriptor["host"] == "100.97.133.126"
    # produced_by
    assert ctx.artifact_descriptor["produced_by"] == "hw.dump_firmware"
    # Other fields equal the descriptor's
    for key in ARTIFACT_DESCRIPTOR_FIELDS:
        if key != "host":  # host is reconciled
            assert ctx.artifact_descriptor[key] == desc[key]
    # Audit row is ok/hitl_approved (no validation_failed)
    rows = _audit_rows(tmp_path)
    assert rows[-1]["outcome"] == "hitl_approved"


# ---------------------------------------------------------------------------
# D25 — two text blocks => refused
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d25_two_text_blocks_refused(tmp_path):
    """D25: two text blocks (descriptor + prose) => refused;
    model-facing error names 'exactly one text content block' and '2';
    capture received verbatim; audit row validation_failed."""
    drive = str(uuid.uuid4())
    desc = _valid_descriptor(drive)
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    captured = []
    class _CaptureStub:
        async def maybe_substitute(self, worker, tool, result, substitute=True, session_id=None):
            captured.append((worker, tool, result))
            return result
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send, capture=_CaptureStub())
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    # Two text blocks
    class _R:
        content = [
            type("_B", (), {"type": "text", "text": json.dumps(desc)})(),
            type("_B", (), {"type": "text", "text": "extra prose"})(),
        ]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    # Model gets the refusal
    assert result.isError is True
    assert "exactly one text content block" in result.content[0].text
    assert "2" in result.content[0].text
    # Capture received the verbatim worker result (not the refusal)
    assert len(captured) == 1
    _, _, captured_result = captured[0]
    assert captured_result is not result  # model gets refusal, not captured
    # The captured result is the original worker result (two text blocks)
    assert len(captured_result.content) == 2
    # ctx.artifact_descriptor stays None
    assert ctx.artifact_descriptor is None
    # Audit row is validation_failed
    rows = _audit_rows(tmp_path)
    assert rows[-1]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D26 — one text block that is not JSON => refused
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d26_non_json_text_refused(tmp_path):
    """D26: one text block that is not JSON (prose) => refused,
    detail says it is not a JSON object; capture verbatim."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    captured = []
    class _CaptureStub:
        async def maybe_substitute(self, worker, tool, result, substitute=True, session_id=None):
            captured.append((worker, tool, result))
            return result
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send, capture=_CaptureStub())
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    class _R:
        content = [type("_B", (), {"type": "text", "text": "just prose"})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert result.isError is True
    assert "not a JSON object" in result.content[0].text.lower()
    # Capture received the verbatim worker result
    assert len(captured) == 1
    _, _, captured_result = captured[0]
    assert len(captured_result.content) == 1
    assert captured_result.content[0].text == "just prose"


# ---------------------------------------------------------------------------
# D27 — JSON array => refused (not a dict)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d27_json_array_refused(tmp_path):
    """D27: one text block that is a JSON array => refused,
    'not a JSON object'."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    captured = []
    class _CaptureStub:
        async def maybe_substitute(self, worker, tool, result, substitute=True, session_id=None):
            captured.append((worker, tool, result))
            return result
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send, capture=_CaptureStub())
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    class _R:
        content = [type("_B", (), {"type": "text", "text": json.dumps([1, 2, 3])})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert result.isError is True
    assert "not a JSON object" in result.content[0].text.lower()
    # Capture received the verbatim worker result
    assert len(captured) == 1
    _, _, captured_result = captured[0]
    assert len(captured_result.content) == 1


# ---------------------------------------------------------------------------
# D28 — invalid descriptor: path outside slug directory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d28_path_outside_slug_refused(tmp_path):
    """D28: path outside the slug directory (/mnt/bench-store/other-slug/x.iso)
    => DescriptorError -> refused with validator's message."""
    drive = str(uuid.uuid4())
    desc = _valid_descriptor(drive)
    desc["path"] = "/mnt/bench-store/other-slug/x.iso"
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    captured = []
    class _CaptureStub:
        async def maybe_substitute(self, worker, tool, result, substitute=True, session_id=None):
            captured.append((worker, tool, result))
            return result
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send, capture=_CaptureStub())
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    class _R:
        content = [type("_B", (), {"type": "text", "text": json.dumps(desc)})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert result.isError is True
    # Validator's message names the worker+tool
    assert "hw.dump_firmware" in result.content[0].text
    # Capture received the verbatim worker result
    assert len(captured) == 1
    _, _, captured_result = captured[0]
    assert len(captured_result.content) == 1
    # Audit row is validation_failed
    rows = _audit_rows(tmp_path)
    assert rows[-1]["outcome"] == "validation_failed"


# ---------------------------------------------------------------------------
# D29 — drive_id mismatch => refused
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d29_drive_id_mismatch_refused(tmp_path):
    """D29: descriptor drive_id != spec.artifact_drive_id => refused."""
    drive = str(uuid.uuid4())
    desc = _valid_descriptor(drive)
    desc["drive_id"] = str(uuid.uuid4())  # mismatch
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    class _R:
        content = [type("_B", (), {"type": "text", "text": json.dumps(desc)})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert result.isError is True
    # Validator's message names the worker+tool
    assert "hw.dump_firmware" in result.content[0].text
    assert "drive_id" in result.content[0].text.lower()


# ---------------------------------------------------------------------------
# D30 — host mismatch: not refused, reconciled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d30_host_mismatch_reconciled(tmp_path):
    """D30: descriptor host 'pare-bench', artifact_host '100.97.133.126'
    => NOT refused; result passes through; ctx.artifact_descriptor['host']
    == '100.97.133.126' (reconciled); audit detail contains both hosts."""
    drive = str(uuid.uuid4())
    desc = _valid_descriptor(drive)
    desc["host"] = "pare-bench"  # mismatch
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    class _R:
        content = [type("_B", (), {"type": "text", "text": json.dumps(desc)})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    # NOT refused — result passes through
    assert not getattr(result, "isError", False)
    # ctx.artifact_descriptor host reconciled to spec's
    assert ctx.artifact_descriptor is not None
    assert ctx.artifact_descriptor["host"] == "100.97.133.126"
    # Audit detail contains both hosts
    rows = _audit_rows(tmp_path)
    detail = rows[-1].get("detail") or ""
    assert "pare-bench" in detail
    assert "100.97.133.126" in detail


# ---------------------------------------------------------------------------
# D31 (P-pin) — in-band worker error => no validation attempted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d31_in_band_error_skips_validation(tmp_path):
    """D31: in-band worker error (inner returns isError=True) =>
    no validation attempted; result passes through; capture verbatim."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    captured = []
    class _CaptureStub:
        async def maybe_substitute(self, worker, tool, result, substitute=True, session_id=None):
            captured.append((worker, tool, result))
            return result
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send, capture=_CaptureStub())
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    pool._inner.return_error = True
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    # Result passes through (in-band error, not a refusal)
    assert result.isError is True
    # ctx.artifact_descriptor stays None (no validation attempted)
    assert ctx.artifact_descriptor is None
    # Capture received the verbatim worker result
    assert len(captured) == 1
    _, _, captured_result = captured[0]
    assert captured_result is result
    # Audit row is 'error' (existing outcome)
    rows = _audit_rows(tmp_path)
    assert rows[-1]["outcome"] == "error"


# ---------------------------------------------------------------------------
# D32 (P-pin) — dispatch exception => _ErrorResult
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d32_dispatch_exception_error_result(tmp_path):
    """D32: dispatch exception (inner raises) => _ErrorResult;
    ctx.artifact_descriptor stays None."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="low",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="low", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    pool._inner.raise_on_call = True
    result = await pool.call_tool("hw", "dump_firmware", {}, ctx=ctx)
    assert result.isError is True
    assert ctx.artifact_descriptor is None


# ---------------------------------------------------------------------------
# D33 (P-pin) — generation recheck => refused before dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d33_generation_recheck_refused(tmp_path):
    """D33: bump the worker's generation between call_tool entry and
    dispatch => refused before dispatch; no validation; no handoff."""
    drive = str(uuid.uuid4())
    spec = WorkerSpec(
        name="hw",
        transport="streamable_http",
        endpoint="http://100.97.133.126:9101/mcp",
        risk_default="high",  # high tier so approval is needed
        artifact_root="/mnt/bench-store",
        artifact_drive_id=drive,
        artifact_host="100.97.133.126",
    )
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
    pool = _pool_with_spec(spec, _Listing([_Tool("dump_firmware", tier="high", produces="artifact")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    await pool.list_tools("hw")
    ctx = _ctx(project_slug="bench-slug-abc123")
    call_task = asyncio.create_task(
        pool.call_tool("hw", "dump_firmware", {}, ctx=ctx))
    # Wait for approval request to be sent
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert sent, "no approval request was emitted"
    # Bump generation while approval is pending
    pool._bump("hw")
    # Resolve approval
    reg.resolve(sent[0].proposal_id, ToolDecision(approved=True, justification="ok"))
    result = await asyncio.wait_for(call_task, timeout=10)
    # Refused before dispatch — result is _ErrorResult
    assert result.isError is True
    assert "reload" in result.content[0].text.lower()
    # ctx.artifact_descriptor stays None (no validation happened)
    assert ctx.artifact_descriptor is None


# ---------------------------------------------------------------------------
# D34 (P-pin) — non-artifact tool with JSON result => passes through
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d34_non_artifact_tool_json_passes_through(tmp_path):
    """D34: non-artifact tool whose result is a JSON-object text block
    => passes through; ctx.artifact_descriptor stays None."""
    spec = WorkerSpec(
        name="hw",
        transport="stdio",
        command="x",
        risk_default="low",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("read_uart", tier="low")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    ctx = _ctx(project_slug="bench-slug")
    json_text = json.dumps({"foo": "bar"})
    class _R:
        content = [type("_B", (), {"type": "text", "text": json_text})()]
        isError = False
    async def _stub(*a, **kw): return _R()
    pool._inner.call_tool = _stub
    result = await pool.call_tool("hw", "read_uart", {}, ctx=ctx)
    assert not getattr(result, "isError", False)
    assert ctx.artifact_descriptor is None


# ---------------------------------------------------------------------------
# D35 (P-pin) — non-artifact tool with list arguments => no TypeError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_d35_non_artifact_tool_list_args_no_error(tmp_path):
    """D35: non-artifact tool with list arguments => no TypeError
    (the injection is guarded on produces)."""
    spec = WorkerSpec(
        name="hw",
        transport="stdio",
        command="x",
        risk_default="low",
    )
    reg = ToolApprovalRegistry()
    send, sent = _approval_send(reg)
    pool = _pool_with_spec(spec, _Listing([_Tool("read_uart", tier="low")]),
                           reg=reg, audit_dir=tmp_path, send=send)
    ctx = _ctx(project_slug="bench-slug")
    result = await pool.call_tool("hw", "read_uart", ["not", "a", "dict"], ctx=ctx)
    assert not getattr(result, "isError", False)
