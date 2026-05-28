# agent_core/workers/tool_approval.py
"""ToolApprovalRegistry — per-call HITL gates for MCP tool dispatch.

Separate from ApprovalRegistry (vault-proposal kinds). Holds a Future per
in-flight tool call keyed by proposal_id; resolve() is the operator path,
discard() is the idempotent cleanup path (cancellation / send failure).
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from agent_core.workers.types import RiskTier

DEFAULT_APPROVAL_TIMEOUT_SECONDS = 120.0
ApprovalScope = Literal["once", "session"]


@dataclass(frozen=True)
class ToolCallSpec:
    worker: str
    tool: str
    arguments: dict[str, Any]  # already a deepcopy snapshot when constructed by the pool
    declared_tier: RiskTier
    effective_tier: RiskTier


@dataclass(frozen=True)
class ToolDecision:
    approved: bool
    justification: str | None
    scope: ApprovalScope = "once"


@dataclass
class _Pending:
    spec: ToolCallSpec
    future: "asyncio.Future[ToolDecision]"
    timer: asyncio.TimerHandle | None = None


class ToolApprovalRegistry:
    def __init__(self, default_timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS) -> None:
        self._pending: dict[str, _Pending] = {}
        self._default_timeout = default_timeout_seconds

    async def request(
        self, spec: ToolCallSpec, timeout_seconds: float | None = None,
    ) -> tuple[str, "asyncio.Future[ToolDecision]"]:
        pid = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ToolDecision] = loop.create_future()
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        def _on_timeout() -> None:
            entry = self._pending.pop(pid, None)
            if entry is not None and not entry.future.done():
                entry.future.set_result(ToolDecision(approved=False, justification="timeout"))

        timer = loop.call_later(timeout, _on_timeout)
        self._pending[pid] = _Pending(spec=spec, future=fut, timer=timer)
        return pid, fut

    def resolve(self, proposal_id: str, decision: ToolDecision) -> None:
        """Operator response. Raises KeyError if the id is unknown — which is
        EXPECTED when a response arrives after timeout/cancellation; callers
        (the daemon dispatcher) must swallow KeyError."""
        entry = self._pending.pop(proposal_id)  # KeyError if absent — intentional
        if entry.timer is not None:
            entry.timer.cancel()
        if not entry.future.done():
            entry.future.set_result(decision)

    def discard(self, proposal_id: str) -> None:
        """Idempotent cleanup. Cancels the timer and drops the entry WITHOUT
        resolving the future. Safe to call multiple times."""
        entry = self._pending.pop(proposal_id, None)
        if entry is not None and entry.timer is not None:
            entry.timer.cancel()

    def is_pending(self, proposal_id: str) -> bool:
        return proposal_id in self._pending
