# agent_core/workers/risk_pool.py
"""RiskAwareToolPool — enforcement wrapper around MCPClientPool.

call_tool is the single chokepoint: risk-evaluate, gate high/critical on
operator approval, audit every dispatch. list_tools/close_all proxy
straight through (discovery is read-only and ungated).
"""
from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Awaitable, Callable

from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate
from agent_core.workers.tool_approval import (
    ToolApprovalRegistry, ToolCallSpec, ToolDecision,
)
from agent_core.workers.types import AuditEntry, WorkerSpec

SendMessage = Callable[[Any], Awaitable[None]]


class _ErrorResult:
    """Minimal CallToolResult-shaped object for denied/failed dispatch, so
    callers (tool_factory._stringify_result / isError checks) behave uniformly."""
    def __init__(self, message: str) -> None:
        self.isError = True

        class _Block:
            type = "text"
            text = message

        self.content = [_Block()]


class RiskAwareToolPool:
    def __init__(
        self,
        *,
        inner: MCPClientPool,
        specs: dict[str, WorkerSpec],
        risk_gate: RiskGate,
        approval_registry: ToolApprovalRegistry,
        audit_log: AuditLog,
        send_message: SendMessage,
    ) -> None:
        self._inner = inner
        self._specs = specs
        self._gate = risk_gate
        self._registry = approval_registry
        self._audit = audit_log
        self._send = send_message
        self._session_approved: set[tuple[str, str]] = set()

    # --- ungated proxies -------------------------------------------------
    async def list_tools(self, worker: str):
        return await self._inner.list_tools(worker)

    async def close_all(self) -> None:
        await self._inner.close_all()

    # --- gated dispatch --------------------------------------------------
    async def call_tool(self, worker: str, tool: str, arguments: dict[str, Any]):
        snapshot = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        spec = self._specs.get(worker)
        declared = spec.risk_default if spec else "high"  # unknown worker -> fail safe-ish
        decision = self._gate.evaluate(worker=worker, tool=tool, declared_tier=declared)
        effective = decision.effective_tier

        gate_reason: str | None = None
        if effective in ("high", "critical"):
            # Session-approved low-risk shortcut (never for critical).
            if effective != "critical" and (worker, tool) in self._session_approved:
                gate_reason = "session-approved"
            else:
                blocked = await self._await_operator(
                    worker, tool, snapshot, declared, effective,
                )
                if blocked is not None:  # denied / undeliverable / timeout
                    return blocked

        return await self._execute_and_audit(
            worker, tool, arguments, snapshot, declared, effective, gate_reason,
        )

    async def _await_operator(self, worker, tool, snapshot, declared, effective):
        """Returns an _ErrorResult if the call should NOT proceed, else None."""
        from agent_core.protocol.messages import ToolApprovalRequestMessage

        spec = ToolCallSpec(
            worker=worker, tool=tool, arguments=snapshot,
            declared_tier=declared, effective_tier=effective,
        )
        proposal_id, future = await self._registry.request(spec)
        req = ToolApprovalRequestMessage(
            proposal_id=proposal_id, worker=worker, tool=tool,
            arguments=snapshot, declared_tier=declared, effective_tier=effective,
        )
        try:
            await self._send(req)
        except Exception as exc:
            self._registry.discard(proposal_id)
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "approval_undeliverable", exc.__class__.__name__)
            return _ErrorResult(f"{worker}.{tool} blocked: approval channel unavailable")
        try:
            decision = await future
        finally:
            self._registry.discard(proposal_id)  # idempotent: covers cancel/normal/timeout

        # Critical requires a non-empty justification even on approval.
        if effective == "critical" and decision.approved and not (decision.justification or "").strip():
            decision = ToolDecision(approved=False, justification="justification required for critical tier")

        if not decision.approved:
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "hitl_denied", decision.justification)
            return _ErrorResult(f"{worker}.{tool} denied by operator: {decision.justification or 'no reason given'}")

        if decision.scope == "session" and effective != "critical":
            self._session_approved.add((worker, tool))
        return None  # approved -> proceed

    async def _execute_and_audit(self, worker, tool, arguments, snapshot, declared, effective, gate_reason):
        start = time.monotonic()
        try:
            result = await self._inner.call_tool(worker, tool, arguments)
        except Exception as exc:
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "error", exc.__class__.__name__)
            return _ErrorResult(f"{worker}.{tool} call failed: {exc}")
        latency = int((time.monotonic() - start) * 1000)
        is_error = bool(getattr(result, "isError", False))
        if gate_reason == "session-approved":
            outcome = "hitl_approved"
        elif effective in ("high", "critical"):
            outcome = "hitl_approved"
        else:
            outcome = "error" if is_error else "ok"
        self._emit(worker, tool, snapshot, declared, effective, latency, outcome, gate_reason)
        return result

    def _emit(self, worker, tool, snapshot, declared, effective, latency_ms, outcome, reason):
        self._audit.append(AuditEntry(
            request_id=uuid.uuid4().hex,
            worker=worker, tool=tool, args=snapshot,
            declared_tier=declared, effective_tier=effective,
            override_reason=reason, outcome=outcome,
            latency_ms=latency_ms, session_guid="pending",
            worker_contract_version=1,
        ))
