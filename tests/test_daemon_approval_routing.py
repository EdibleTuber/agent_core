# tests/test_daemon_approval_routing.py
import asyncio
import pytest

from agent_core.daemon import Daemon
from agent_core.workers.tool_approval import (
    ToolApprovalRegistry, ToolCallSpec, ToolDecision,
)
from agent_core.protocol.messages import ToolApprovalResponseMessage


class _FakeAgent:
    """Minimal stand-in: Daemon only touches .tool_approval_registry in the
    routing path under test."""
    def __init__(self, registry=None):
        self.tool_approval_registry = registry


def _daemon(agent):
    # Daemon.__init__ may require only the agent; if it needs more, inspect it.
    return Daemon(agent)


def _spec():
    return ToolCallSpec(worker="frida", tool="exec", arguments={},
                        declared_tier="high", effective_tier="high")


@pytest.mark.asyncio
async def test_route_resolves_pending_approval():
    reg = ToolApprovalRegistry()
    pid, fut = await reg.request(_spec())
    daemon = _daemon(_FakeAgent(reg))
    daemon._route_approval_response(
        ToolApprovalResponseMessage(proposal_id=pid, approved=True, justification=None)
    )
    decision = await fut
    assert decision.approved is True
    assert not reg.is_pending(pid)


@pytest.mark.asyncio
async def test_route_unknown_proposal_id_does_not_raise():
    reg = ToolApprovalRegistry()
    daemon = _daemon(_FakeAgent(reg))
    # must not raise despite no such pending entry
    daemon._route_approval_response(
        ToolApprovalResponseMessage(proposal_id="ghost", approved=False, justification="x")
    )


@pytest.mark.asyncio
async def test_route_without_registry_is_noop():
    daemon = _daemon(_FakeAgent(registry=None))
    daemon._route_approval_response(
        ToolApprovalResponseMessage(proposal_id="x", approved=True, justification=None)
    )  # no raise
