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
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from typing import Any

from agent_core.workers.client import MCPClient
from agent_core.workers.types import WorkerSpec

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 10.0


class MCPClientPool:
    """Holds one MCPClient per worker name, each owned by its own task."""

    def __init__(self, specs: list[WorkerSpec]) -> None:
        self._specs: dict[str, WorkerSpec] = {s.name: s for s in specs}
        self._clients: dict[str, MCPClient] = {}
        self._owners: dict[str, asyncio.Task] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._stop: dict[str, asyncio.Event] = {}
        self._errors: dict[str, BaseException] = {}
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

    def _owner_pid(self, worker: str) -> int | None:
        """The stdio child's pid, for tests and for hard-kill on a wedged close.
        None for non-stdio transports or before connect completes."""
        client = self._clients.get(worker)
        ctx = getattr(client, "_transport_ctx", None)
        gen = getattr(ctx, "gen", None)
        frame = getattr(gen, "ag_frame", None)
        proc = frame.f_locals.get("process") if frame is not None else None
        return getattr(proc, "pid", None)

    # --- connection lifecycle --------------------------------------------
    async def _own(self, worker: str) -> None:
        """Own one worker's connection for its entire lifetime.

        Connect, publish, park, close — all in this one task, which is what
        makes teardown legal.
        """
        client = MCPClient.from_spec(self._specs[worker])
        try:
            await client.connect()
            await client.initialize()
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
            with contextlib.suppress(BaseException):
                await client.close()
            raise
        self._clients[worker] = client
        self._ready[worker].set()
        try:
            await self._stop[worker].wait()
        finally:
            self._clients.pop(worker, None)
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
            exc = self._errors.get(worker)
            if exc is not None:
                await self._reap(worker)
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
                        f"connecting to worker {worker!r} failed "
                        f"(cancelled internally: {exc!r})"
                    ) from exc
                raise exc

    async def _cancel_owner(self, worker: str) -> None:
        task = self._owners.pop(worker, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        self._cleanup(worker)

    async def _reap(self, worker: str) -> None:
        task = self._owners.pop(worker, None)
        if task is not None:
            with contextlib.suppress(BaseException):
                await task
        self._cleanup(worker)

    def _cleanup(self, worker: str) -> None:
        self._clients.pop(worker, None)
        self._ready.pop(worker, None)
        self._stop.pop(worker, None)

    async def disconnect(self, worker: str) -> None:
        """Stop the worker's owner task and wait for its close to finish.

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
            await self._reap(worker)
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
        for worker in list(self._owners):
            with contextlib.suppress(BaseException):
                await self.disconnect(worker)
        self._clients.clear()
