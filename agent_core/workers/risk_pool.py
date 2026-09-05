# agent_core/workers/risk_pool.py
"""RiskAwareToolPool — enforcement wrapper around MCPClientPool.

call_tool is the single chokepoint: risk-evaluate, gate high/critical on
operator approval, audit every dispatch. list_tools/close_all proxy
straight through (discovery is read-only and ungated).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
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

logger = logging.getLogger(__name__)

SendMessage = Callable[[Any], Awaitable[None]]

_TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _max_tier(a: str | None, b: str | None) -> str | None:
    """The higher of two tiers; None if neither is a recognized tier string.

    Inputs may be hostile/arbitrary (dict, list, bool, ...) — risk.py:59-61
    documents this for the wire tier specifically, and it applies equally to
    anything read from `_tool_tiers`/`_tier_highwater`. The isinstance guard
    must run before the `in` membership test, since `_TIER_ORDER` is a dict
    and membership-testing an unhashable value (a dict/list) raises TypeError.
    """
    known = [t for t in (a, b) if isinstance(t, str) and t in _TIER_ORDER]
    return max(known, key=lambda t: _TIER_ORDER[t]) if known else None


class _ErrorResult:
    """Minimal CallToolResult-shaped object for denied/failed dispatch, so
    callers (tool_factory._stringify_result / isError checks) behave uniformly."""
    def __init__(self, message: str) -> None:
        self.isError = True

        class _Block:
            type = "text"
            text = message

        self.content = [_Block()]


def _worker_error_message(result) -> str | None:
    """Return the worker's error message when the result carries an in-band
    ``{"error": true}`` envelope, else None.

    Internal workers report failures by returning normally with an error
    envelope rather than raising, so ``isError`` stays False. This surfaces that
    envelope for honest auditing. Defensive against non-JSON / non-envelope
    content: anything unparseable yields None (treated as success).
    """
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("error") is True:
            msg = str(payload.get("summary") or "worker error")
            detail = payload.get("detail")
            if detail:
                msg = f"{msg}: {detail}"
            return msg[:500]
    return None


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
        for spec in (specs or {}).values():
            inner.add_spec(spec)
        self._gate = risk_gate
        self._registry = approval_registry
        self._audit = audit_log
        self._send = send_message
        self._capture = capture_layer
        # (worker, tool, generation) — an approval is scoped to the exact worker
        # instance it was granted against.
        self._session_approved: set[tuple[str, str, int]] = set()
        self._tool_tiers: dict[tuple[str, str], str | None] = {}
        # Highest tier ever observed for a tool this session. NEVER evicted:
        # escalate-only must be monotonic across time, not just within one
        # resolution, or a reload becomes a downgrade channel (spec 6.4.1).
        self._tier_highwater: dict[tuple[str, str], str] = {}
        self._generations: dict[str, int] = {}

    # --- lifecycle ---------------------------------------------------------
    def spec_for(self, worker: str):
        """Read through to the inner pool — one source of truth for specs."""
        return self._inner.spec(worker)

    def generation(self, worker: str) -> int:
        return self._generations.get(worker, 0)

    def _bump(self, worker: str) -> None:
        self._generations[worker] = self.generation(worker) + 1
        self._session_approved = {
            e for e in self._session_approved if e[0] != worker
        }
        self._tool_tiers = {
            k: v for k, v in self._tool_tiers.items() if k[0] != worker
        }
        # _tier_highwater is deliberately NOT cleared.

    def add_spec(self, spec) -> None:
        self._inner.add_spec(spec)
        self._bump(spec.name)

    def remove_spec(self, worker: str) -> None:
        self._inner.remove_spec(worker)
        self._bump(worker)

    def record_session_approval(self, worker: str, tool: str, generation: int) -> None:
        """Record only if the worker has not been reloaded since the approval
        was requested. An operator answering a prompt after a reload must not
        pre-approve the new process."""
        if generation == self.generation(worker):
            self._session_approved.add((worker, tool, generation))

    def is_session_approved(self, worker: str, tool: str) -> bool:
        return (worker, tool, self.generation(worker)) in self._session_approved

    def _resolve_declared(self, worker: str, tool: str) -> tuple[str, str | None]:
        """Single source of truth for (declared_tier, tier_source): the ONE
        place the wire tier, the high-water mark, and the external_mcp
        floor-only contract are reconciled. `call_tool` and `resolve_effective`
        both call this rather than each recomputing it -- they drifted once
        already (spec 6.4.3) by being two implementations of one rule.
        """
        spec = self.spec_for(worker)
        wire_tier = self._tool_tiers.get((worker, tool))
        declared, tier_source = resolve_declared_tier(spec, wire_tier)
        if spec is not None and spec.kind == "external_mcp":
            # external_mcp: floor only. Per-tool wire tiers are not honored
            # (risk.py:46, risk.py:53, risk.py:67-68) -- and neither is the
            # high-water mark derived from them, or the floor-only contract
            # would leak back in through the escalation below.
            return declared, tier_source
        high = self._tier_highwater.get((worker, tool))
        if (tier_source != "unknown_worker" and high is not None
                and _TIER_ORDER.get(high, -1) > _TIER_ORDER.get(declared, -1)):
            declared, tier_source = high, "wire"
        return declared, tier_source

    def resolve_effective(self, worker: str, tool: str) -> str:
        """The tier a dispatch would resolve to right now. Extracted so tests
        and the tier-ratchet check can ask without dispatching."""
        declared, _ = self._resolve_declared(worker, tool)
        return self._gate.evaluate(worker=worker, tool=tool,
                                   declared_tier=declared).effective_tier

    def emit_lifecycle(self, worker: str, action: str, detail: str | None = None,
                       args: dict | None = None) -> None:
        """Control-plane audit row. Load/unload changes the enforcement config
        itself, which is the first thing an audit log exists for."""
        self._audit.append(AuditEntry(
            request_id=uuid.uuid4().hex, worker=worker, tool=None,
            args=args or {}, declared_tier="low", effective_tier="low",
            override_reason=None, detail=detail, outcome=action,
            latency_ms=0, session_guid="pending", worker_contract_version=1,
            tier_source=None,
        ))

    # --- ungated proxies -------------------------------------------------
    async def list_tools(self, worker: str):
        result = await self._inner.list_tools(worker)
        for tool in getattr(result, "tools", []) or []:
            name = getattr(tool, "name", None)
            if name is None:
                # A nameless tool from a buggy/hostile worker must not abort
                # discovery of the worker's remaining tools.
                continue
            try:
                meta = getattr(tool, "meta", None) or {}
                tier = meta.get(RISK_TIER_META_KEY) if isinstance(meta, dict) else None
                self._tool_tiers[(worker, name)] = tier
                hw = _max_tier(tier, self._tier_highwater.get((worker, name)))
                if hw is not None:
                    self._tier_highwater[(worker, name)] = hw
            except Exception:
                # Same guarantee as the nameless-tool case above, extended to a
                # malformed/hostile per-tool entry (bad meta shape, unhashable
                # tier value, ...): one bad entry must not abort discovery of
                # this worker's remaining tools, nor of tools already recorded.
                # Logged (not silent) so a vanishing tool leaves a trace.
                logger.warning(
                    "worker %r: malformed tool entry during discovery (tool=%r), skipped",
                    worker, name,
                )
        return result

    async def close_all(self) -> None:
        # Union, not `or`: `or` short-circuits on the first truthy operand, so
        # once ANY worker has ever been reloaded (`_generations` non-empty),
        # every never-reloaded worker in `_inner.names()` would be skipped and
        # keep its session approvals across a full teardown.
        for worker in set(self._generations) | set(self._inner.names()):
            self._bump(worker)
        await self._inner.close_all()

    # --- gated dispatch --------------------------------------------------
    async def call_tool(self, worker: str, tool: str, arguments: dict[str, Any], ctx: Any = None,
                        capture: bool = True):
        snapshot = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        # A tool with no advertised tier resolves to the worker's risk_default
        # FLOOR (risk.py:75-78) -- not a fail-safe to high. Safety for
        # dangerous tools that fail to advertise comes from operator pins plus
        # the session high-water mark below, never from a dispatch-time
        # fallback. declared/tier_source come from _resolve_declared, the same
        # single resolution path resolve_effective uses -- see its docstring.
        declared, tier_source = self._resolve_declared(worker, tool)
        gen = self.generation(worker)
        decision = self._gate.evaluate(worker=worker, tool=tool, declared_tier=declared)
        effective = decision.effective_tier
        gate_override = decision.override_reason  # why escalated (None if declared==effective)

        session_note: str | None = None
        if effective in ("high", "critical"):
            if effective != "critical" and self.is_session_approved(worker, tool):
                session_note = "session-approved"
            else:
                send = self._resolve_send(ctx)
                blocked = await self._await_operator(
                    worker, tool, snapshot, declared, effective, gate_override, send,
                    tier_source, gen,
                )
                if blocked is not None:  # denied / undeliverable / timeout
                    return blocked

        result = await self._execute_and_audit(
            worker, tool, arguments, snapshot, declared, effective, gate_override,
            session_note, tier_source,
        )
        if self._capture is not None:
            # Route ALL executed results through the capture layer — including
            # errors — so a failed run stays searchable. The layer stores
            # unconditionally and never stubs an error. (Approval blocks/denials
            # returned earlier and are intentionally not captured: no tool ran.)
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
                              tier_source, generation):
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
            self.record_session_approval(worker, tool, generation)
        if generation != self.generation(worker):
            return _ErrorResult(
                f"{worker} was reloaded while approval was pending; re-issue the call")
        return None  # approved -> proceed

    async def _execute_and_audit(self, worker, tool, arguments, snapshot, declared, effective, gate_override, session_note,
                                 tier_source=None):
        start = time.monotonic()
        try:
            result = await self._inner.call_tool(worker, tool, arguments)
        except asyncio.CancelledError:
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "cancelled", gate_override, "worker disconnected mid-dispatch",
                       tier_source)
            raise
        except Exception as exc:
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "error", gate_override, exc.__class__.__name__,
                       tier_source)
            return _ErrorResult(f"{worker}.{tool} call failed: {exc}")
        latency = int((time.monotonic() - start) * 1000)
        # Internal workers signal failure with an in-band {"error": true} envelope
        # while returning normally (FastMCP only sets isError on a raise), so the
        # protocol flag alone would record these as "ok". Honour both, and carry
        # the worker's own message into the audit detail.
        worker_err = _worker_error_message(result)
        is_error = bool(getattr(result, "isError", False)) or worker_err is not None
        if is_error:
            outcome = "error"
            detail = worker_err or session_note
        elif session_note == "session-approved" or effective in ("high", "critical"):
            outcome = "hitl_approved"
            detail = session_note
        else:
            outcome = "ok"
            detail = session_note
        self._emit(worker, tool, snapshot, declared, effective, latency, outcome, gate_override, detail,
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
