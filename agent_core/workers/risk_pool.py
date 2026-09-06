# agent_core/workers/risk_pool.py
"""RiskAwareToolPool — enforcement wrapper around MCPClientPool.

call_tool is the single chokepoint: risk-evaluate, gate high/critical on
operator approval, audit every dispatch.

`list_tools` and `close_all` are ungated for the CALLER, but neither is a
pass-through: `list_tools` writes the per-tool wire tiers and the session
high-water mark that every later resolution reads, and `close_all` bumps
every worker's generation and evicts its session approvals. Both mutate
security state, so both belong to this class rather than the inner pool.
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
from agent_core.workers.types import WORKER_CONTRACT_VERSION
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
        # Highest risk_default FLOOR ever seen for a worker this session, and
        # never evicted, for the same reason. _tier_highwater is fed only from
        # the wire meta, so it said nothing about the floor -- while reload()
        # re-reads the registry on every reload and WorkerRegistry.add() is
        # public. Lower a worker's risk_default, re-add it, /worker reload, and
        # every non-advertising tool dropped to the new floor: exactly the
        # downgrade channel the high-water comment above claims to close.
        self._floor_highwater: dict[str, str] = {}
        self._generations: dict[str, int] = {}
        # Registered LAST, and through _record_floor rather than a bare
        # inner.add_spec: seeding the floor ratchet eagerly is what keeps a
        # constructor-supplied worker that is never dispatched to from having
        # its floor lowered by a later add_spec/reload. Deliberately not via
        # self.add_spec() -- these are the session's initial specs, so bumping
        # every generation off zero here would be noise.
        for spec in (specs or {}).values():
            inner.add_spec(spec)
            self._record_floor(spec.name, spec)

    # --- lifecycle ---------------------------------------------------------
    def spec_for(self, worker: str):
        """Read through to the inner pool — one source of truth for specs."""
        return self._inner.spec(worker)

    def target(self, worker: str) -> str:
        """Read through to the inner pool -- what this worker points at.

        WorkerManager asks the pool it was GIVEN, which is this wrapper, not
        the inner client pool. Without this passthrough the lookup silently
        fell back to the bare worker name, so every message built from it --
        connect_timeout, unreachable, the liveness probe error -- named the
        one thing the operator already knew and omitted the endpoint, which
        with three machines in play is the entire diagnosis.
        """
        getter = getattr(self._inner, "target", None)
        return getter(worker) if callable(getter) else worker

    def server_info(self, worker: str) -> dict | None:
        """Read through to the inner pool -- one source of truth for identity."""
        getter = getattr(self._inner, "server_info", None)
        return getter(worker) if callable(getter) else None

    def _is_local(self, worker: str) -> bool:
        """True when the KERNEL guarantees this worker cannot be replaced
        without us noticing.

        For stdio it does: the daemon spawned the child and holds its pipe, so
        the process behind a connection cannot change without a reload, and a
        reload bumps the generation. For anything networked it does not -- a
        Pi can reboot, a systemd unit can restart, a container can be
        redeployed, and none of it reaches the daemon.

        Unknown workers count as NOT local: fail closed.
        """
        spec = self.spec_for(worker)
        return getattr(spec, "transport", None) == "stdio"

    def _bump_on_link_loss(self, worker: str) -> bool:
        """Treat a transport failure on a networked worker as a possible
        restart, and evict its session approvals.

        Generation-keyed approvals were a property of LOCAL PROCESS OWNERSHIP,
        not of the keying itself: every _bump call site is daemon-driven
        (add_spec, remove_spec, close_all), which is sound only while the
        daemon is the only thing that can change the process. Over HTTP it is
        not. Without this, a `scope: session` approval survives onto a
        DIFFERENT PROCESS -- dispatching with no prompt and an audit row
        reading hitl_approved / session-approved.

        The cost is deliberate and one-directional: after a network blip an
        operator re-approves. That is the right side to err on, because the
        failure it replaces is silent.
        """
        if self._is_local(worker):
            return False
        self._bump(worker)
        return True

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
        self._record_floor(spec.name, spec)
        self._bump(spec.name)

    def _record_floor(self, worker: str, spec) -> None:
        """Fold a spec's risk_default into the per-worker floor ratchet.

        Called from add_spec (so a reload's new spec is seen even if the worker
        is never dispatched to) and from _resolve_declared (so a spec that
        reached the inner pool by another route -- the constructor, a direct
        inner.add_spec -- is seen too).
        """
        if spec is None:
            return
        hw = _max_tier(getattr(spec, "risk_default", None),
                       self._floor_highwater.get(worker))
        if hw is not None:
            self._floor_highwater[worker] = hw

    def remove_spec(self, worker: str) -> None:
        self._inner.remove_spec(worker)
        self._bump(worker)

    async def connect(self, worker: str, timeout: float | None = None) -> None:
        """Delegate to the inner pool. WorkerManager calls this rather than
        reaching through `self._pool._inner` — the pool, not the manager,
        owns knowledge of the inner client pool's shape."""
        await self._inner.connect(worker, timeout=timeout)

    async def disconnect(self, worker: str, timeout: float | None = None) -> None:
        await self._inner.disconnect(worker, timeout=timeout)

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
        if tier_source == "unknown_worker":
            return declared, tier_source
        self._record_floor(worker, spec)
        # Floor ratchet, applied to EVERY kind including external_mcp: it is
        # derived from the operator's own workers.yaml, not from anything the
        # worker advertised, so honoring it does not breach the external_mcp
        # "floor only" contract -- and skipping it would leave a kind flipped
        # internal -> external_mcp across a reload as a downgrade channel of
        # its own (deferred item D6).
        floor_high = self._floor_highwater.get(worker)
        if (floor_high is not None
                and _TIER_ORDER.get(floor_high, -1) > _TIER_ORDER.get(declared, -1)):
            declared = floor_high
            if tier_source != "invalid_advertised":
                tier_source = "floor"
        if spec is not None and spec.kind == "external_mcp":
            # external_mcp: floor only. Per-tool wire tiers are not honored
            # (risk.py:46, risk.py:53, risk.py:67-68) -- and neither is the
            # high-water mark derived from them, or the floor-only contract
            # would leak back in through the escalation below.
            return declared, tier_source
        high = self._tier_highwater.get((worker, tool))
        if (high is not None
                and _TIER_ORDER.get(high, -1) > _TIER_ORDER.get(declared, -1)):
            declared = high
            # "invalid_advertised" survives the escalation: overwriting it with
            # "wire" erased the only record that the worker sent a malformed
            # tier (a contract violation / tampering signal). Escalation was
            # always correct here; the forensics were what degraded.
            if tier_source != "invalid_advertised":
                tier_source = "wire"
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
            latency_ms=0, session_guid="pending", worker_contract_version=WORKER_CONTRACT_VERSION,
            tier_source=None,
        ))

    # --- ungated proxies -------------------------------------------------
    async def ping(self, worker: str):
        """Ungated: a ping carries no arguments and invokes no tool, so there
        is nothing for the risk gate to evaluate. It is control-plane."""
        return await self._inner.ping(worker)

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
                # A non-dict meta container is recorded AS IS rather than
                # normalized to None: resolve_declared_tier maps any non-str,
                # non-None advertised value to "invalid_advertised", whereas
                # None records as "floor" -- indistinguishable in the audit log
                # from an honest non-advertiser.
                tier = meta.get(RISK_TIER_META_KEY) if isinstance(meta, dict) else meta
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
        """SHUTDOWN ONLY. Bumps every generation (evicting session approvals)
        and tears the inner pool down, but knows nothing about WorkerManager:
        calling it directly leaves `WorkerManager._loaded` stale, so `status()`
        goes on reporting loaded workers whose connections are closed. Runtime
        teardown belongs at `WorkerManager.unload`/`close_all`.
        """
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
            session_note, tier_source, generation=gen,
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
                                 tier_source=None, generation=None):
        start = time.monotonic()
        if generation is not None and generation != self.generation(worker):
            # The generation check at the end of _await_operator is not enough
            # on its own: it holds only because the path from there to
            # self._inner.call_tool happens to contain no await that yields.
            # Insert one anywhere on that path -- a capture hook, a dispatch
            # semaphore, a rate limiter -- and a reload landing in the new
            # window dispatches an operator-approved high/critical call against
            # a DIFFERENT subprocess, which is the exact hole the generation
            # counter exists to close. Re-checking here removes the reliance on
            # that accident rather than documenting it.
            self._emit(worker, tool, snapshot, declared, effective, 0,
                       "error", gate_override,
                       "worker reloaded between approval and dispatch", tier_source)
            return _ErrorResult(
                f"{worker} was reloaded before {tool} could be dispatched; "
                f"re-issue the call")
        try:
            result = await self._inner.call_tool(worker, tool, arguments)
        except asyncio.CancelledError:
            evicted = self._bump_on_link_loss(worker)
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "cancelled", gate_override,
                       "worker disconnected mid-dispatch"
                       + (" (approvals evicted: link loss on a networked "
                          "worker may mean a restarted process)" if evicted else ""),
                       tier_source)
            raise
        except Exception as exc:
            # A tool that merely FAILS comes back as a result with isError set,
            # not as an exception -- FastMCP only raises on protocol/transport
            # trouble. So reaching here on a networked worker means the link
            # itself misbehaved, which is indistinguishable from the worker
            # having restarted underneath us.
            evicted = self._bump_on_link_loss(worker)
            self._emit(worker, tool, snapshot, declared, effective,
                       int((time.monotonic() - start) * 1000),
                       "error", gate_override,
                       exc.__class__.__name__
                       + (" (approvals evicted: link loss on a networked "
                          "worker may mean a restarted process)" if evicted else ""),
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
            worker_contract_version=WORKER_CONTRACT_VERSION, tier_source=tier_source,
        ))
