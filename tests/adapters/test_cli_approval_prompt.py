# tests/adapters/test_cli_approval_prompt.py
import pytest
from agent_core.adapters.cli import handle_approval_request, _sanitize_args
from agent_core.protocol.messages import (
    ToolApprovalRequestMessage, ToolApprovalResponseMessage,
)


def _req(tier="high"):
    return ToolApprovalRequestMessage(
        proposal_id="p1", worker="frida", tool="exec",
        arguments={"javascript_code": "A" * 5000}, declared_tier=tier, effective_tier=tier,
    )


def test_sanitize_truncates_and_strips_control_chars():
    out = _sanitize_args({"k": "line1\nline2\x1b[31mred", "big": "B" * 9999})
    assert "\n" not in out and "\x1b" not in out
    assert len(out) < 1200  # truncated


@pytest.mark.asyncio
async def test_approve_once_sends_approved_scope_once():
    sent = []
    async def send(m): sent.append(m)
    async def prompt(_): return "y"
    await handle_approval_request(_req("high"), prompt, send)
    assert sent[0].approved is True and sent[0].scope == "once"


@pytest.mark.asyncio
async def test_deny_sends_not_approved():
    sent = []
    async def send(m): sent.append(m)
    async def prompt(_): return "n"
    await handle_approval_request(_req("high"), prompt, send)
    assert sent[0].approved is False


@pytest.mark.asyncio
async def test_approve_session_sets_scope_session():
    sent = []
    async def send(m): sent.append(m)
    async def prompt(_): return "a"
    await handle_approval_request(_req("high"), prompt, send)
    assert sent[0].approved is True and sent[0].scope == "session"


@pytest.mark.asyncio
async def test_justification_path_collects_text():
    sent = []
    answers = iter(["j", "needed for crash repro"])
    async def send(m): sent.append(m)
    async def prompt(_): return next(answers)
    await handle_approval_request(_req("high"), prompt, send)
    assert sent[0].approved is True and sent[0].justification == "needed for crash repro"


@pytest.mark.asyncio
async def test_critical_forces_justification_and_rejects_bare_y():
    sent = []
    answers = iter(["y", "really wipe it"])  # bare 'y' must be re-prompted for justification
    async def send(m): sent.append(m)
    async def prompt(_): return next(answers)
    await handle_approval_request(_req("critical"), prompt, send)
    assert sent[0].approved is True and sent[0].justification == "really wipe it"


@pytest.mark.asyncio
async def test_critical_empty_justification_denies():
    sent = []
    answers = iter(["j", ""])
    async def send(m): sent.append(m)
    async def prompt(_): return next(answers)
    await handle_approval_request(_req("critical"), prompt, send)
    assert sent[0].approved is False


@pytest.mark.asyncio
async def test_keyboard_interrupt_sends_denied():
    sent = []
    async def send(m): sent.append(m)
    async def prompt(_): raise KeyboardInterrupt
    await handle_approval_request(_req("high"), prompt, send)
    assert sent[0].approved is False and sent[0].justification == "cancelled"
