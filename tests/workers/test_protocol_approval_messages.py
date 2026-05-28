# tests/workers/test_protocol_approval_messages.py
from agent_core.protocol.transport import encode_message, decode_message
from agent_core.protocol.messages import (
    ToolApprovalRequestMessage, ToolApprovalResponseMessage,
)


def test_request_round_trips():
    msg = ToolApprovalRequestMessage(
        proposal_id="p1", worker="frida", tool="exec",
        arguments={"session_id": "s1"}, declared_tier="medium", effective_tier="high",
    )
    out = decode_message(encode_message(msg))
    assert isinstance(out, ToolApprovalRequestMessage)
    assert out.proposal_id == "p1" and out.effective_tier == "high"
    assert out.arguments == {"session_id": "s1"}


def test_response_round_trips_with_scope_and_justification():
    msg = ToolApprovalResponseMessage(
        proposal_id="p1", approved=True, justification="crash repro", scope="session",
    )
    out = decode_message(encode_message(msg))
    assert out.approved is True and out.justification == "crash repro" and out.scope == "session"


def test_response_defaults_scope_once():
    msg = ToolApprovalResponseMessage(proposal_id="p1", approved=False)
    out = decode_message(encode_message(msg))
    assert out.scope == "once" and out.justification is None
