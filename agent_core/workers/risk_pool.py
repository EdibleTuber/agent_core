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
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from agent_core.capture.layer import CaptureLayer

from agent_core.workers.audit import AuditLog
from agent_core.workers.client_pool import MCPClientPool
from agent_core.workers.risk import RiskGate, RISK_TIER_META_KEY, resolve_declared_tier
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
        send_message: SendMessage | None = None,
        capture_layer: "CaptureLayer | None" = None,
    ) -> None:
        self._inner = inner
        self._specs = specs
        self._gate = risk_gate
        self._registry = approval_registry
        self._audit = audit_log
        self._send = send_message
        self._capture = capture_layer
        self._session_approved: set[tuple[str, str]] = set()
        self._tool_tiers: dict[tuple[str, str], str | None] = {}

    # --- ungated proxies -------------------------------------------------
    async def list_tools(self, worker: str):
        result = await self._inner.list_tools(worker)
        for tool in getattr(result, "tools", []) or []:
            name = getattr(tool, "name", None)
            if name is None:
                # A nameless tool from a buggy/hostile worker must not abort
                # discovery of the worker's remaining tools.
                continue
            meta = getattr(tool, "meta", None) or {}
            tier = meta.get(RISK_TIER_META_KEY) if isinstance(meta, dict) else None
            self._tool_tiers[(worker, name)] = tier
        return result

    async def close_all(self) -> None:
        await self._inner.close_all()

    # --- gated dispatch --------------------------------------------------
    async def call_tool(self, worker: str, tool: str, arguments: dict[str, Any], ctx: Any = None,
                        capture: bool = True):
        snapshot = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        spec = self._specs.get(worker)
        # A tool we never saw in discovery is treated as "no advertised tier"
        # (None) -> resolve_declared_tier fails safe to "high" for internal workers.
        advertised = self._tool_tiers.get((worker, tool))
        declared, tier_source = resolve_declared_tier(spec, advertised)
        decision = self._gate.evaluate(worker=worker, tool=tool, declared_tier=declared)
        effective = decision.effective_tier
        gate_override = decision.override_reason  # why escalated (None if declared==effective)

        session_note: str | None = None
        if effective in ("high", "critical"):
            if effective != "critical" and (worker, tool) in self._session_approved:
                session_note = "session-approved"
            else:
                send = self._resolve_send(ctx)
                blocked = await self._await_operator(
                    worker, tool, snapshot, declared, effective, gate_override, send,
                    tier_source,
                )
                if blocked is not None:  # denied / undeliverable / timeout
                    return blocked

        result = await self._execute_and_audit(
            worker, tool, arguments, snapshot, declared, effective, gate_override,
            session_note, tier_source,
        )
        if self._capture is not None and not getattr(result, "isError", False):
            session_id = arguments.get("session_id") if isinstance(arguments, dict) else None
            return self._capture.maybe_substitute(worker, tool, result, substitute=capture,
                                                  session_id=session_id)
        return result

    def _resolve_send(self, ctx):
        """Prefer the per-request connection channel (ctx.emit); fall back to a
        constructor-supplied send_message (used by tests); None means no channel."""
        emit = getattr(ctx, "emit", None) if ctx is not None else None
        if callable(emit):
            return emit
        return self._send

    async def _await_operator(self, worker, tool, snapshot, declared, effective, gate_override, send,
                              tier_source=None):
        """Returns an _ErrorResult if the call should NOT proceed, else None."""
        from agent_core.protocol.messages import ToolApprovalRequestMessage

        if send is None:
            # No approval channel available -> fail closed (no registry entry created).
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "approval_undeliverable", gate_override, "no approval channel",
                       tier_source)
            return _ErrorResult(f"{worker}.{tool} blocked: no approval channel available")

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
            await send(req)
        except Exception as exc:
            self._registry.discard(proposal_id)
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "approval_undeliverable", gate_override, exc.__class__.__name__,
                       tier_source)
            return _ErrorResult(f"{worker}.{tool} blocked: approval channel unavailable")
        try:
            decision = await future
        finally:
            self._registry.discard(proposal_id)  # idempotent: covers cancel/normal/timeout

        if effective == "critical" and decision.approved and not (decision.justification or "").strip():
            decision = ToolDecision(approved=False, justification="justification required for critical tier")

        if not decision.approved:
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "hitl_denied", gate_override, decision.justification,
                       tier_source)
            return _ErrorResult(f"{worker}.{tool} denied by operator: {decision.justification or 'no reason given'}")

        if decision.scope == "session" and effective != "critical":
            self._session_approved.add((worker, tool))
        return None  # approved -> proceed

    async def _execute_and_audit(self, worker, tool, arguments, snapshot, declared, effective, gate_override, session_note,
                                 tier_source=None):
        start = time.monotonic()
        try:
            result = await self._inner.call_tool(worker, tool, arguments)
        except Exception as exc:
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "error", gate_override, exc.__class__.__name__,
                       tier_source)
            return _ErrorResult(f"{worker}.{tool} call failed: {exc}")
        latency = int((time.monotonic() - start) * 1000)
        is_error = bool(getattr(result, "isError", False))
        if is_error:
            outcome = "error"
        elif session_note == "session-approved" or effective in ("high", "critical"):
            outcome = "hitl_approved"
        else:
            outcome = "ok"
        self._emit(worker, tool, snapshot, declared, effective, latency, outcome, gate_override, session_note,
                   tier_source)
        return result

    def _emit(self, worker, tool, snapshot, declared, effective, latency_ms, outcome, override_reason, detail,
              tier_source=None):
        self._audit.append(AuditEntry(
            request_id=uuid.uuid4().hex,
            worker=worker, tool=tool, args=snapshot,
            declared_tier=declared, effective_tier=effective,
            override_reason=override_reason, detail=detail, outcome=outcome,
            latency_ms=latency_ms, session_guid="pending",
            worker_contract_version=1, tier_source=tier_source,
        ))
