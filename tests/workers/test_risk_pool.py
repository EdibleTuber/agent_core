# tests/workers/test_risk_pool.py
import asyncio
import json
import pytest

from agent_core.workers.risk_pool import RiskAwareToolPool
from agent_core.workers.tool_approval import ToolApprovalRegistry, ToolDecision
from agent_core.workers.risk import RiskGate
from agent_core.workers.audit import AuditLog
from agent_core.workers.types import WorkerSpec
from agent_core.protocol.messages import ToolApprovalRequestMessage


class _InnerPool:
    def __init__(self):
        self.calls = []
        self.raise_on_call = False
    async def call_tool(self, worker, tool, arguments):
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.calls.append((worker, tool, arguments))
        class _R: content = []; isError = False
        return _R()
    async def list_tools(self, worker):
        class _T: tools = []
        return _T()
    async def close_all(self):
        pass


def _pool(inner, specs, gate=None, reg=None, audit_dir=None, send=None):
    return RiskAwareToolPool(
        inner=inner,
        specs={s.name: s for s in specs},
        risk_gate=gate or RiskGate(overrides=[]),
        approval_registry=reg or ToolApprovalRegistry(),
        audit_log=AuditLog(audit_dir),
        send_message=send or (lambda m: asyncio.sleep(0)),
    )


def _spec(name, tier):
    return WorkerSpec(name=name, transport="stdio", command="x", risk_default=tier)


def _audit_lines(audit_dir):
    files = list(audit_dir.glob("audit-*.jsonl"))
    assert len(files) == 1
    return [json.loads(l) for l in files[0].read_text().splitlines()]


@pytest.mark.asyncio
async def test_low_tier_auto_executes_and_audits(tmp_path):
    inner = _InnerPool()
    sent = []
    async def send(m): sent.append(m)
    pool = _pool(inner, [_spec("echo", "low")], audit_dir=tmp_path, send=send)
    await pool.call_tool("echo", "ping", {})
    assert inner.calls == [("echo", "ping", {})]
    assert sent == []
    rows = _audit_lines(tmp_path)
    assert len(rows) == 1 and rows[0]["effective_tier"] == "low" and rows[0]["outcome"] == "ok"


@pytest.mark.asyncio
async def test_high_tier_blocks_until_approved(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("frida", "exec", {"x": 1})
    assert inner.calls == [("frida", "exec", {"x": 1})]
    assert len(sent) == 1 and isinstance(sent[0], ToolApprovalRequestMessage)
    assert _audit_lines(tmp_path)[0]["outcome"] == "hitl_approved"


@pytest.mark.asyncio
async def test_high_tier_denied_does_not_call_inner(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=False, justification="no"))
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    out = await pool.call_tool("frida", "exec", {"x": 1})
    assert inner.calls == []
    assert out.isError is True
    assert _audit_lines(tmp_path)[0]["outcome"] == "hitl_denied"


@pytest.mark.asyncio
async def test_critical_approved_without_justification_is_denied(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    pool = _pool(inner, [_spec("frida", "critical")], reg=reg, audit_dir=tmp_path, send=send)
    out = await pool.call_tool("frida", "wipe", {})
    assert inner.calls == []
    assert out.isError is True


@pytest.mark.asyncio
async def test_send_failure_denies_immediately_and_audits_undeliverable(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry(default_timeout_seconds=5.0)
    async def send(m):
        raise ConnectionError("socket closed")
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    out = await pool.call_tool("frida", "exec", {})
    assert inner.calls == []
    assert out.isError is True
    rows = _audit_lines(tmp_path)
    assert rows[0]["outcome"] == "approval_undeliverable"
    # the registry must not leak the pending entry after a send failure
    assert reg._pending == {}


@pytest.mark.asyncio
async def test_timeout_denies(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry(default_timeout_seconds=0.2)
    async def send(m): pass  # never resolves
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    out = await pool.call_tool("frida", "exec", {})
    assert inner.calls == []
    assert out.isError is True
    assert _audit_lines(tmp_path)[0]["outcome"] == "hitl_denied"


@pytest.mark.asyncio
async def test_session_scope_skips_subsequent_prompt(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None, scope="session"))
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("frida", "exec", {"n": 1})
    await pool.call_tool("frida", "exec", {"n": 2})
    assert len(sent) == 1                      # second call did not prompt
    assert len(inner.calls) == 2               # both executed
    rows = _audit_lines(tmp_path)
    assert rows[1]["detail"] == "session-approved"


@pytest.mark.asyncio
async def test_critical_cannot_be_session_approved(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification="ok", scope="session"))
    pool = _pool(inner, [_spec("frida", "critical")], reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("frida", "wipe", {"n": 1})
    await pool.call_tool("frida", "wipe", {"n": 2})
    assert len(sent) == 2                       # prompted both times despite session scope


@pytest.mark.asyncio
async def test_args_snapshot_is_deepcopied_into_audit(tmp_path):
    inner = _InnerPool()
    pool = _pool(inner, [_spec("echo", "low")], audit_dir=tmp_path)
    args = {"nested": {"k": "v"}}
    await pool.call_tool("echo", "ping", args)
    args["nested"]["k"] = "mutated"            # mutate after the call
    rows = _audit_lines(tmp_path)
    assert rows[0]["args"]["nested"]["k"] == "v"   # audit kept the snapshot


@pytest.mark.asyncio
async def test_audit_justification_with_json_payload_round_trips(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    evil = 'denied"}\n{"injected":"row'
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=False, justification=evil))
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("frida", "exec", {})
    rows = _audit_lines(tmp_path)              # parses cleanly -> no injection
    assert len(rows) == 1 and rows[0]["detail"] == evil


@pytest.mark.asyncio
async def test_parallel_calls_resolved_out_of_order(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    captured = []
    async def send(m): captured.append(m)
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    t1 = asyncio.create_task(pool.call_tool("frida", "exec", {"id": 1}))
    t2 = asyncio.create_task(pool.call_tool("frida", "exec", {"id": 2}))
    while len(captured) < 2:
        await asyncio.sleep(0.01)
    reg.resolve(captured[1].proposal_id, ToolDecision(approved=True, justification=None))
    reg.resolve(captured[0].proposal_id, ToolDecision(approved=True, justification=None))
    await asyncio.gather(t1, t2)
    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_list_tools_and_close_proxy_ungated(tmp_path):
    inner = _InnerPool()
    pool = _pool(inner, [_spec("frida", "critical")], audit_dir=tmp_path)
    await pool.list_tools("frida")             # no prompt, no audit
    await pool.close_all()
    assert list(tmp_path.glob("audit-*.jsonl")) == []


@pytest.mark.asyncio
async def test_approved_high_inner_error_audits_error(tmp_path):
    inner = _InnerPool()
    inner.raise_on_call = False
    reg = ToolApprovalRegistry()
    async def send(m):
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    # make the inner result report isError
    async def erroring_call(worker, tool, arguments):
        inner.calls.append((worker, tool, arguments))
        class _R: content = []; isError = True
        return _R()
    inner.call_tool = erroring_call
    out = await pool.call_tool("frida", "exec", {})
    assert inner.calls == [("frida", "exec", {})]      # it WAS approved + executed
    rows = _audit_lines(tmp_path)
    assert rows[0]["outcome"] == "error"               # execution truth wins, not hitl_approved


@pytest.mark.asyncio
async def test_gate_override_reason_is_audited(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    # worker declares low, but a name-pattern override escalates frida_exec to high
    gate = RiskGate(overrides=[("frida_exec", "high")])
    pool = _pool(inner, [_spec("frida", "low")], gate=gate, reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("frida", "exec", {})
    assert len(sent) == 1                               # escalation forced an approval prompt
    rows = _audit_lines(tmp_path)
    assert rows[0]["declared_tier"] == "low"
    assert rows[0]["effective_tier"] == "high"
    assert rows[0]["override_reason"] == "name pattern 'frida_exec' forces high"


@pytest.mark.asyncio
async def test_cancellation_cleans_up_pending(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry(default_timeout_seconds=5.0)
    captured = []
    async def send(m): captured.append(m)   # never resolves
    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=send)
    task = asyncio.create_task(pool.call_tool("frida", "exec", {}))
    while not captured:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reg._pending == {}                # finally: discard cleaned up
    assert inner.calls == []                 # inner never reached


@pytest.mark.asyncio
async def test_unknown_worker_defaults_to_high_and_prompts(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()
    sent = []
    async def send(m):
        sent.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))
    # specs does NOT include "ghost"
    pool = _pool(inner, [_spec("frida", "low")], reg=reg, audit_dir=tmp_path, send=send)
    await pool.call_tool("ghost", "doit", {})
    assert len(sent) == 1                     # unknown worker -> high -> prompted
    rows = _audit_lines(tmp_path)
    assert rows[0]["declared_tier"] == "high"


class _Ctx:
    """Minimal stand-in for HandlerContext carrying an awaitable emit."""
    def __init__(self, emit):
        self.emit = emit


@pytest.mark.asyncio
async def test_ctx_emit_is_preferred_over_constructor_send(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry()

    async def constructor_send(m):
        raise AssertionError("constructor send_message used instead of ctx.emit")

    emit_calls = []
    async def ctx_emit(m):
        emit_calls.append(m)
        reg.resolve(m.proposal_id, ToolDecision(approved=True, justification=None))

    pool = _pool(inner, [_spec("frida", "high")], reg=reg, audit_dir=tmp_path, send=constructor_send)
    await pool.call_tool("frida", "exec", {"x": 1}, ctx=_Ctx(ctx_emit))
    assert len(emit_calls) == 1          # request went out via ctx.emit
    assert inner.calls == [("frida", "exec", {"x": 1})]   # approved + executed
    assert _audit_lines(tmp_path)[0]["outcome"] == "hitl_approved"


@pytest.mark.asyncio
async def test_no_send_channel_fails_closed(tmp_path):
    inner = _InnerPool()
    reg = ToolApprovalRegistry(default_timeout_seconds=5.0)
    # neither ctx (with emit) nor a constructor send_message
    pool = RiskAwareToolPool(
        inner=inner,
        specs={s.name: s for s in [_spec("frida", "high")]},
        risk_gate=RiskGate(overrides=[]),
        approval_registry=reg,
        audit_log=AuditLog(tmp_path),
        send_message=None,
    )
    out = await pool.call_tool("frida", "exec", {})   # no ctx
    assert getattr(out, "isError", False)
    assert inner.calls == []
    rows = _audit_lines(tmp_path)
    assert rows[0]["outcome"] == "approval_undeliverable"
    assert reg._pending == {}     # no entry created when there's no channel
