"""MCPClientPool — one owner task per worker, lazy connect, reused across calls.

Why an owner task per worker: anyio binds a cancel scope to the task that
entered it. `MCPClient.connect()` enters two nested scopes (the stdio_client
task group and ClientSession.__aenter__), so `close()` MUST run in the task
that ran `connect()`. The daemon connects during `Agent.astartup()` (the serve
task) and disconnects from a per-message handler task, so a pool that closed in
the caller's task would raise

    RuntimeError: Attempted to exit cancel scope in a different task than it
                  was entered in

and leave the worker subprocess running for the life of the daemon. The owner
task connects, parks on a stop event, and closes in its own `finally` — the
same task throughout.

Dispatch is unaffected: calling a client from a foreign task is safe, and only
teardown carries the affinity requirement.

Every join on an owner task is bounded, and an owner we stop waiting for is
handed to `_abandon()`: its recorded child pid is SIGKILLed and the task is
kept in `_orphans` so `close_all()` can still reach it. Abandoning an owner
without those two steps leaves a worker subprocess alive and unreachable.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from collections import defaultdict
from typing import Any

from agent_core.workers.client import MCPClient
from agent_core.workers.types import WorkerSpec

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_OWNER_JOIN_TIMEOUT = 5.0
"""Bound on how long _cancel_owner waits for a cancelled owner task to unwind.

Without this, connect()'s own `timeout` is not actually a wall-clock bound:
the _ready wait is bounded, but a wedged owner task (stuck in a subprocess
call that ignores cancellation) can make the cleanup that follows a timeout
hang forever. Best-effort — see the module docstring."""


def _server_info_of(init_result) -> dict:
    """Pull name/version/protocol out of an MCP InitializeResult.

    Defensive throughout: this runs on the connect path, and a worker that
    returns something unexpected must not fail an otherwise-good load.
    """
    info = getattr(init_result, "serverInfo", None)
    out = {
        "server_name": getattr(info, "name", None),
        "server_version": getattr(info, "version", None),
        "protocol_version": getattr(init_result, "protocolVersion", None),
    }
    return {k: v for k, v in out.items() if v is not None}


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten an ExceptionGroup to the exceptions that actually happened."""
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_leaves(sub))
        return out
    return [exc]


def describe_failure(exc: BaseException) -> str:
    """A one-line cause an operator can act on.

    The SDK's transports run inside anyio task groups, so a refused connection
    reaches us as a BaseExceptionGroup whose members are mostly CancelledError
    -- the siblings the group cancelled -- with the real ConnectError among
    them. `repr()` of that group tells the operator nothing about which host
    refused what, which is the only thing they need to know. So: flatten,
    drop the cancellations UNLESS that is all there is, and if a leaf carries
    no message of its own, borrow its cause's.
    """
    leaves = _leaves(exc)
    real = [e for e in leaves if not isinstance(e, asyncio.CancelledError)]
    parts: list[str] = []
    for e in (real or leaves):
        text = str(e).strip()
        cause = e.__cause__ or e.__context__
        if not text and cause is not None:
            text = str(cause).strip()
        parts.append(f"{type(e).__name__}: {text}" if text else type(e).__name__)
    # dict.fromkeys: an ExceptionGroup routinely repeats one identical cause
    # once per cancelled sibling, and five copies is not more informative.
    return "; ".join(dict.fromkeys(parts))


def teardown_cause(worker: str, target: str,
                   close_exc: BaseException) -> ConnectionError | None:
    """The real reason a connect failed, recovered from the TEARDOWN.

    An unreachable Streamable-HTTP endpoint does not fail inside initialize().
    The SDK runs its transport in an anyio task group; when the POST dies the
    group cancels the caller, so initialize() raises a bare CancelledError
    carrying a cancel-scope address and nothing else. The httpx.ConnectError
    that actually happened only surfaces when the transport scope unwinds --
    which happens inside close(), on the failure path, where it was being
    suppressed. Both halves were discarded, and the operator got
    "cancelled internally: CancelledError('Cancelled via cancel scope 0x...')"
    with the endpoint nowhere in it.

    Returns None when close() raised nothing better than cancellations, in
    which case the original error stands.
    """
    if not any(not isinstance(e, asyncio.CancelledError)
               for e in _leaves(close_exc)):
        return None
    err = ConnectionError(
        f"connecting to worker {worker!r} at {target} failed: "
        f"{describe_failure(close_exc)}")
    err.__cause__ = close_exc
    return err


class MCPClientPool:
    """Holds one MCPClient per worker name, each owned by its own task."""

    def __init__(self, specs: list[WorkerSpec]) -> None:
        self._specs: dict[str, WorkerSpec] = {s.name: s for s in specs}
        self._clients: dict[str, MCPClient] = {}
        # What the worker said it was at the last successful initialize.
        # For a networked worker this is the ONLY provenance available: the
        # daemon did not spawn the process and cannot stat its binary, so
        # _artifact_args records None for every file field. See server_info().
        self._server_info: dict[str, dict] = {}
        self._owners: dict[str, asyncio.Task] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._stop: dict[str, asyncio.Event] = {}
        self._errors: dict[str, BaseException] = {}
        # The stdio child's pid, captured the moment the client is usable.
        # _own() pops self._clients BEFORE awaiting close(), so a worker wedged
        # in close() has no client to read a pid off any more -- and that is
        # exactly the case the hard-kill exists for.
        self._pids: dict[str, int] = {}
        # Owner tasks we gave up waiting for. Kept REACHABLE (the pre-fix _reap
        # popped _owners before awaiting, so a caller-cancelled reap dropped the
        # last reference to a still-running owner and no API could reach it
        # again) so close_all() can make a final pass at them.
        self._orphans: dict[asyncio.Task, tuple[str, int | None]] = {}
        # Per-worker, not global: one worker hanging in connect() must not block
        # every other worker's first use.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # --- spec bookkeeping -------------------------------------------------
    def add_spec(self, spec: WorkerSpec) -> None:
        """Register a spec. Idempotent; does not connect."""
        self._specs[spec.name] = spec

    def remove_spec(self, name: str) -> None:
        """Forget a spec. The caller disconnects first — this is deliberately
        sync so spec bookkeeping never awaits, and so a wedged teardown cannot
        leave the spec behind (see WorkerManager.unload's ordering)."""
        self._specs.pop(name, None)

    def spec(self, name: str) -> WorkerSpec | None:
        """The single source of truth for worker specs. RiskAwareToolPool reads
        through to this rather than keeping its own copy."""
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def is_connected(self, worker: str) -> bool:
        return worker in self._clients

    @staticmethod
    def _pid_of(client) -> int | None:
        """Dig the stdio child's pid out of the live transport context."""
        ctx = getattr(client, "_transport_ctx", None)
        gen = getattr(ctx, "gen", None)
        frame = getattr(gen, "ag_frame", None)
        proc = frame.f_locals.get("process") if frame is not None else None
        return getattr(proc, "pid", None)

    def server_info(self, worker: str) -> dict | None:
        """What the worker reported at initialize: name, version, protocol.

        This is the closest thing a NETWORKED worker has to the mtime/size an
        stdio worker's binary provides. It is not equivalent -- a worker
        controls what it reports, so it proves nothing against a hostile
        server -- but it turns "no provenance at all" into a value that
        changes when the remote build changes, which is what makes an
        unannounced restart or redeploy visible in the audit log.
        """
        return self._server_info.get(worker)

    def target(self, worker: str) -> str:
        """What this worker's connection actually points at, for error text.

        Without it, "connecting to worker 'frida' failed" names the only thing
        the operator already knew. With three machines in play the endpoint is
        the whole diagnosis.
        """
        spec = self._specs.get(worker)
        if spec is None:
            return worker
        if spec.transport == "stdio":
            return " ".join([spec.command or "", *spec.args]).strip() or worker
        return spec.endpoint or worker

    def _owner_pid(self, worker: str) -> int | None:
        """The stdio child's pid, for tests and for hard-kill on a wedged close.
        None for non-stdio transports or before connect completes.

        Falls back to the pid recorded at connect time: _own() drops the client
        from self._clients before it awaits close(), so during a wedged close --
        the one moment the hard-kill needs a pid -- the live lookup returns None.
        """
        client = self._clients.get(worker)
        if client is not None:
            pid = self._pid_of(client)
            if pid is not None:
                return pid
        return self._pids.get(worker)

    def _kill_pid(self, worker: str, pid: int | None) -> bool:
        """SIGKILL a worker child we have stopped waiting for.

        Spec section 7: "on timeout, HARD-KILL the recorded child pid". An
        owner task wedged in close() keeps its subprocess alive for the life of
        the daemon and beyond -- for a process-attaching worker that is a live
        attachment outliving the process that made it. SIGKILL rather than
        SIGTERM: we are here precisely because the graceful path already
        outlived its bound.
        """
        # The pid is dug out of an SDK generator's frame locals, so treat it as
        # untrusted before handing it to a destructive syscall: a non-int would
        # raise TypeError straight past the handlers below and out of
        # _abandon(), and os.kill(0, SIGKILL) signals the daemon's ENTIRE
        # process group -- the daemon itself included. pid 1 is never ours.
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            if pid is not None:
                logger.warning("worker %r: refusing to hard-kill implausible "
                               "pid %r", worker, pid)
            return False
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return False        # already gone; nothing to report
        except OSError:
            logger.warning("worker %r: could not hard-kill pid %s", worker, pid,
                           exc_info=True)
            return False
        logger.warning("worker %r: hard-killed child pid %s after its close() "
                       "outlived the bound", worker, pid)
        return True

    def _abandon(self, worker: str, task: asyncio.Task) -> None:
        """Stop waiting for an owner task: hard-kill its child, cancel it, and
        keep it reachable.

        Cancelling the owner from outside is the sanctioned way to abort a
        partially-entered anyio scope (see the module docstring); the kill goes
        first so a close() blocked on a dead child can actually unwind.
        """
        pid = self._pids.get(worker)
        self._orphans[task] = (worker, pid)
        task.add_done_callback(lambda t: self._orphans.pop(t, None))
        self._kill_pid(worker, pid)
        task.cancel()

    # --- connection lifecycle --------------------------------------------
    async def _own(self, worker: str) -> None:
        """Own one worker's connection for its entire lifetime.

        Connect, publish, park, close — all in this one task, which is what
        makes teardown legal.
        """
        client = None
        try:
            # INSIDE the try: a KeyError from a racing remove_spec used to
            # escape without ever setting _errors/_ready, so connect() sat out
            # its full timeout instead of failing fast and asyncio logged an
            # "exception was never retrieved" warning.
            client = MCPClient.from_spec(self._specs[worker])
            await client.connect()
            # Recorded as soon as the child EXISTS, not once the worker is
            # usable: a connect that succeeds and an initialize() that fails or
            # hangs still leaves a spawned subprocess, and that is one of the
            # cases the hard-kill has to be able to reach.
            pid = self._pid_of(client)
            if pid is not None:
                self._pids[worker] = pid
            init = await client.initialize()
            self._server_info[worker] = _server_info_of(init)
        except BaseException as exc:      # includes CancelledError on timeout
            self._errors[worker] = exc
            self._ready[worker].set()
            # connect()/initialize() may have partially entered the transport
            # and/or session scopes (e.g. the stdio subprocess is spawned but
            # initialize() never got a reply). MCPClient.close() null-checks
            # both independently, so it's safe to call here even on a
            # completely failed connect — and it MUST be called here, in this
            # same task, or a cancelled/failed connect leaks the subprocess
            # exactly like the bug this pool exists to fix.
            if client is not None:
                try:
                    await client.close()
                except BaseException as close_exc:
                    # Best-effort enrichment, never a new failure mode: if
                    # close() carried the real cause, upgrade the recorded
                    # error; otherwise leave it exactly as it was. connect()
                    # re-reads _errors after reaping this task, so the upgrade
                    # is visible to it.
                    better = teardown_cause(worker, self.target(worker), close_exc)
                    if better is not None:
                        self._errors[worker] = better
            raise
        self._clients[worker] = client
        self._ready[worker].set()
        try:
            await self._stop[worker].wait()
        finally:
            self._clients.pop(worker, None)
            # Identity belongs to the CONNECTION, not the worker name. Keeping
            # it past disconnect would let a stale row claim provenance for a
            # process that is gone.
            self._server_info.pop(worker, None)
            # Best-effort: a wedged worker must not keep the daemon from
            # completing the unload. WorkerManager bounds and hard-kills.
            with contextlib.suppress(BaseException):
                await client.close()

    async def connect(self, worker: str, timeout: float | None = None) -> None:
        """Spawn the owner task and wait until the worker is usable.

        Raises whatever connect/initialize raised, or asyncio.TimeoutError.
        Leaves no residue on failure.
        """
        if worker not in self._specs:
            raise KeyError(f"no worker named {worker!r} in this pool")
        async with self._locks[worker]:
            if worker in self._clients:
                return
            self._ready[worker] = asyncio.Event()
            self._stop[worker] = asyncio.Event()
            self._errors.pop(worker, None)
            self._owners[worker] = asyncio.create_task(
                self._own(worker), name=f"mcp-owner:{worker}")
            try:
                await asyncio.wait_for(
                    self._ready[worker].wait(),
                    timeout if timeout is not None else DEFAULT_CONNECT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Cancel INSIDE the owner task — the only safe way to abort a
                # partially-entered anyio scope.
                await self._cancel_owner(worker)
                raise
            except BaseException:
                # A cancellation delivered to the CALLER of connect() -- the
                # daemon cancels every owned handler task the instant its
                # client disconnects, so a Ctrl-C'd `/worker load` lands here
                # -- left the owner task running, the subprocess spawned and
                # _ready/_stop/_owners still keyed, in flat contradiction of
                # this method's "Leaves no residue on failure". Worse, the
                # owner then went on to publish self._clients[worker] and park
                # forever: a live, reachable worker behind a load that never
                # completed. Same cleanup as the timeout path, then re-raise so
                # the cancellation still propagates to the caller's task.
                with contextlib.suppress(Exception):
                    await self._cancel_owner(worker)
                raise
            if self._errors.get(worker) is not None:
                # BOUNDED, and it must be: this runs while still holding
                # self._locks[worker]. The owner reached here by failing
                # connect()/initialize(), and _own's failure path then awaits
                # client.close() -- which can wedge exactly like any other
                # close. An unbounded join would park connect() on the lock
                # forever, and close_all()'s own 5s bound would never be
                # reached because its disconnect() blocks acquiring that same
                # lock. That trades a leaked orphan for a hung shutdown, which
                # is strictly worse and is the outcome Critical 3 exists to
                # prevent.
                await self._reap(worker, timeout=DEFAULT_OWNER_JOIN_TIMEOUT)
                # Re-read AFTER the reap: _own's teardown is where an
                # unreachable HTTP endpoint's real cause appears, and it
                # rewrites _errors as it unwinds. If the reap timed out, this
                # is simply the original error, unchanged.
                exc = self._errors.get(worker)
                if exc is None:                       # pragma: no cover
                    return
                if isinstance(exc, asyncio.CancelledError):
                    # Reaching here means _ready was observed set *before*
                    # wait_for's timeout fired, so this is not our own
                    # _cancel_owner cancellation (that path returns via the
                    # `except asyncio.TimeoutError` branch above and never
                    # gets here) — it's a bare CancelledError the mcp SDK
                    # raised internally (e.g. its own task-group cancelling
                    # a sibling after a refused connection). Propagating a
                    # BaseException here would let a single unreachable
                    # worker cancel whichever task called connect() — e.g.
                    # the daemon's startup task. Normalize it to a plain
                    # Exception; the caller's own cancellation (if any) is
                    # a completely separate CancelledError raised directly
                    # at the `await asyncio.wait_for(...)` above, and is
                    # untouched by this branch.
                    raise ConnectionError(
                        f"connecting to worker {worker!r} at "
                        f"{self.target(worker)} failed: "
                        f"{describe_failure(exc)}"
                    ) from exc
                if isinstance(exc, BaseExceptionGroup):
                    # Same reasoning as the CancelledError branch above, for
                    # the same reason: an ExceptionGroup's repr buries the one
                    # cause that matters, and a BaseExceptionGroup carrying a
                    # CancelledError is a BaseException that would escape the
                    # caller's `except Exception`.
                    raise ConnectionError(
                        f"connecting to worker {worker!r} at "
                        f"{self.target(worker)} failed: "
                        f"{describe_failure(exc)}"
                    ) from exc
                raise exc

    async def _cancel_owner(self, worker: str) -> None:
        """Cancel the owner task and wait (bounded) for it to unwind.

        `suppress(BaseException)` around a bare `await task` would swallow a
        CancelledError delivered to the CALLER of connect() (e.g. the daemon
        shutting down) just as readily as the CancelledError that is `task`'s
        own, expected outcome of the `task.cancel()` above — the two are
        indistinguishable by type alone. `task.cancelled()` disambiguates:
        it's only True once `task` itself has actually finished cancelling,
        so a CancelledError raised here while `task.cancelled()` is still
        False can only be this coroutine's own cancellation, which must
        propagate. `asyncio.shield` keeps `wait_for`'s timeout-driven cleanup
        (`_cancel_and_wait`, which unconditionally cancels whatever it was
        awaiting) from cancelling `task` itself on the caller-cancellation
        path, which would otherwise flip `task.cancelled()` to True and
        defeat this exact check.
        """
        task = self._owners.get(worker)
        try:
            if task is not None:
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=DEFAULT_OWNER_JOIN_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        "owner task for worker %r did not unwind within %ss after "
                        "cancel(); hard-killing its child and abandoning it",
                        worker, DEFAULT_OWNER_JOIN_TIMEOUT)
                    self._abandon(worker, task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        self._abandon(worker, task)
                        raise
                except Exception:
                    pass  # the owner task's own connect/initialize failure
        finally:
            # Popped only here, once the join has either completed or handed
            # the task to _abandon() -- never before the await (see _reap).
            self._owners.pop(worker, None)
            # MUST run even on the re-raise above: the pre-fix `suppress
            # (BaseException)` always reached this; a bare `raise` inside the
            # try does not, and would otherwise leave _ready/_stop (and
            # _clients, if the owner published between the timeout and the
            # cancel) keyed by `worker` with no owner task behind them.
            self._cleanup(worker)

    async def _reap(self, worker: str, timeout: float | None = None) -> None:
        """Wait for the owner task to finish.

        `timeout` bounds the join; None means "wait as long as it takes"
        because the caller (WorkerManager) supplies its own bound. close_all()
        has no such caller and passes DEFAULT_OWNER_JOIN_TIMEOUT. On expiry --
        or on a cancellation of the caller, which is how a manager-level
        disconnect_timeout arrives -- the owner is handed to _abandon(), which
        hard-kills its child and keeps the task reachable.

        `asyncio.shield(task)` is not optional here, and a bare `await task`
        is not equivalent to `_cancel_owner`'s bare-`await`-free-of-shield
        predecessor being "good enough" -- it is the SAME bug. `Task.cancel()`
        cancels whatever future is currently in that task's `_fut_waiter`;
        when the caller is blocked in a bare `await task`, `task` itself IS
        that future, so cancelling the caller (e.g. daemon shutdown cancelling
        whichever task is running WorkerManager.unload) cancels the owner task
        too, as a direct side effect -- not merely raises CancelledError past
        it. That makes `task.cancelled()` become True as a result of the
        caller's OWN cancellation, not just the owner's, defeating the
        disambiguation below and silently absorbing the caller's cancellation
        (the same failure mode as the original `suppress(BaseException)`, and
        the reason a wedged worker's `disconnect_timeout` could fail to fire
        at all -- the manager's own `wait_for` timeout cancels this task,
        which without the shield would cancel the owner instead of raising
        here). `asyncio.shield` makes the caller await a separate wrapper
        future instead, so cancelling the caller cannot reach `task` and
        `task.cancelled()` still means what it says.
        """
        task = self._owners.get(worker)
        try:
            if task is not None:
                try:
                    if timeout is None:
                        await asyncio.shield(task)
                    else:
                        await asyncio.wait_for(asyncio.shield(task), timeout)
                except asyncio.TimeoutError:
                    self._abandon(worker, task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        # The caller gave up on us. WorkerManager's own
                        # `wait_for(disconnect, disconnect_timeout)` arrives
                        # exactly here, and this IS the mandated
                        # disconnect_timeout hard-kill path.
                        self._abandon(worker, task)
                        raise
                except Exception:
                    pass
        finally:
            # Popped only AFTER the join has completed or the task has been
            # handed to _abandon(). The pre-fix pop happened before the await,
            # so a caller-cancelled reap re-raised past it with the owner task
            # still running and no longer in _owners -- close_all() iterates
            # _owners, so the orphan (and its subprocess) became unreachable
            # for the life of the daemon.
            self._owners.pop(worker, None)
            self._cleanup(worker)

    def _cleanup(self, worker: str) -> None:
        self._clients.pop(worker, None)
        self._ready.pop(worker, None)
        self._stop.pop(worker, None)
        # Only after _abandon() has had its chance to read it: an abandoned
        # owner carries its pid forward in self._orphans.
        self._pids.pop(worker, None)

    async def disconnect(self, worker: str, timeout: float | None = None) -> None:
        """Stop the worker's owner task and wait for its close to finish.

        `timeout` bounds that wait (and hard-kills the child on expiry); None
        leaves it to the caller's own bound -- WorkerManager wraps this in a
        `wait_for(..., disconnect_timeout)` so it can report disconnect_timeout.

        Takes the same per-worker lock as connect() so the two are mutually
        exclusive: without it, a disconnect() could set the stop event out
        from under a connect() that is still waiting on _ready (handing the
        caller a "successful" connect to a client that's already torn down),
        and two concurrent disconnect() calls could race each other's
        _reap(), letting the second return before the first's close() has
        actually finished. _reap()/_cancel_owner() do not themselves take
        the lock, so this stays a single level of acquisition — no nesting,
        no deadlock against connect()'s own lock usage.
        """
        async with self._locks[worker]:
            stop = self._stop.get(worker)
            if stop is not None:
                stop.set()
            await self._reap(worker, timeout=timeout)
            self._errors.pop(worker, None)

    async def _ensure_connected(self, worker: str) -> MCPClient:
        if worker not in self._specs:
            raise KeyError(f"no worker named {worker!r} in this pool")
        if worker not in self._clients:
            await self.connect(worker)
        return self._clients[worker]

    # --- dispatch ---------------------------------------------------------
    async def list_tools(self, worker: str):
        client = await self._ensure_connected(worker)
        return await client.list_tools()

    async def call_tool(self, worker: str, tool: str, arguments: dict[str, Any],
                        ctx: Any = None):
        client = await self._ensure_connected(worker)
        return await client.call_tool(tool, arguments)

    async def close_all(self) -> None:
        """Shutdown reaper. EVERY wait here is bounded.

        This is the only reaper for a worker the manager never recorded as
        loaded (a load that failed or was cancelled after connect), so an
        unbounded join here meant `ashutdown` could hang forever against a
        worker wedged in close -- and under systemd the daemon then gets
        SIGKILLed with its worker subprocesses still up, which is the exact
        outcome the owner-task design exists to prevent.

        "Bounded" includes the per-worker lock each disconnect() acquires,
        which is only true because EVERY holder of that lock is itself bounded:
        connect() by its own timeout plus a bounded _cancel_owner and a bounded
        error-path _reap, and disconnect() by this timeout or the caller's. An
        unbounded wait anywhere under that lock re-hangs shutdown from behind
        it, where this method's own timeout cannot see it.
        """
        for worker in list(self._owners):
            try:
                await self.disconnect(worker, timeout=DEFAULT_OWNER_JOIN_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("worker %r: close_all disconnect failed", worker,
                               exc_info=True)
        # Final pass at anything abandoned earlier (here or by a manager-level
        # disconnect_timeout). Only for owners still running: a done() owner
        # finished client.close(), so anyio has already reaped its child and
        # that pid is free for the OS to reuse -- and the _orphans entry
        # outlives the task by one loop iteration, because its removal is a
        # call_soon done-callback. Killing on that window would signal an
        # unrelated process.
        for task, (worker, pid) in list(self._orphans.items()):
            if not task.done():
                self._kill_pid(worker, pid)
                task.cancel()
        self._clients.clear()
