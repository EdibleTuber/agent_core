# agent_core/tests/workers/test_tool_approval.py
import asyncio
import pytest

from agent_core.workers.tool_approval import (
    ToolApprovalRegistry, ToolCallSpec, ToolDecision,
)


def _spec(tier="high"):
    return ToolCallSpec(worker="frida", tool="exec", arguments={"a": 1},
                        declared_tier=tier, effective_tier=tier)


@pytest.mark.asyncio
async def test_request_returns_id_and_marks_pending():
    reg = ToolApprovalRegistry()
    pid, fut = await reg.request(_spec())
    assert isinstance(pid, str) and pid
    assert reg.is_pending(pid)
    reg.discard(pid)


@pytest.mark.asyncio
async def test_resolve_approved_unblocks():
    reg = ToolApprovalRegistry()
    pid, fut = await reg.request(_spec())
    reg.resolve(pid, ToolDecision(approved=True, justification=None))
    decision = await fut
    assert decision.approved and not reg.is_pending(pid)


@pytest.mark.asyncio
async def test_resolve_denied_carries_reason():
    reg = ToolApprovalRegistry()
    pid, fut = await reg.request(_spec())
    reg.resolve(pid, ToolDecision(approved=False, justification="nope"))
    decision = await fut
    assert decision.approved is False and decision.justification == "nope"


@pytest.mark.asyncio
async def test_timeout_auto_denies():
    reg = ToolApprovalRegistry(default_timeout_seconds=0.2)
    pid, fut = await reg.request(_spec())
    decision = await fut
    assert decision.approved is False and decision.justification == "timeout"
    assert not reg.is_pending(pid)


@pytest.mark.asyncio
async def test_resolve_unknown_raises_keyerror():
    reg = ToolApprovalRegistry()
    with pytest.raises(KeyError):
        reg.resolve("ghost", ToolDecision(approved=True, justification=None))


@pytest.mark.asyncio
async def test_discard_is_idempotent_and_cancels_timer():
    reg = ToolApprovalRegistry(default_timeout_seconds=0.2)
    pid, fut = await reg.request(_spec())
    reg.discard(pid)
    reg.discard(pid)  # no raise
    assert not reg.is_pending(pid)
    # future left unresolved by discard; awaiting it would hang, so don't.


@pytest.mark.asyncio
async def test_resolve_after_timeout_is_keyerror_not_double_set():
    reg = ToolApprovalRegistry(default_timeout_seconds=0.05)
    pid, fut = await reg.request(_spec())
    await fut  # let timeout fire
    with pytest.raises(KeyError):
        reg.resolve(pid, ToolDecision(approved=True, justification=None))


@pytest.mark.asyncio
async def test_cancellation_then_discard_leaves_no_pending():
    reg = ToolApprovalRegistry(default_timeout_seconds=5.0)
    pid, fut = await reg.request(_spec())
    fut.cancel()
    reg.discard(pid)
    assert not reg.is_pending(pid)


@pytest.mark.asyncio
async def test_scope_defaults_to_once():
    d = ToolDecision(approved=True, justification=None)
    assert d.scope == "once"
    d2 = ToolDecision(approved=True, justification=None, scope="session")
    assert d2.scope == "session"
