"""WorkerManager — the runtime worker lifecycle.

The only component that mutates the worker registry, the connection pool and
the tool executor together. Boot and runtime share one path: `load_autoload()`
calls the same `load()` an operator command does, so the two cannot drift.

Every public method returns a result object rather than raising. A worker that
fails to load must not take down the daemon or the caller's turn — the failure
belongs in `/worker list`, not in a traceback.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from agent_core.tools.base import Tool
from agent_core.workers.registry import WorkerNotFoundError, WorkerRegistry
from agent_core.workers.tool_factory import make_tool_class

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_DISCONNECT_TIMEOUT = 5.0

ErrorKind = Literal[
    "unknown_worker", "spawn_failed", "connect_timeout", "protocol_mismatch",
    "tool_collision", "list_tools_failed", "disconnect_timeout",
]


@dataclass
class WorkerOpResult:
    """One result shape for load/unload/reload.

    Reload needs somewhere to say "the unload half failed" — which is exactly
    the state a wedged teardown produces — so it cannot be a LoadResult.
    """
    op: Literal["load", "unload", "reload"]
    name: str
    ok: bool
    tool_count: int = 0
    tools: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: ErrorKind | None = None


@dataclass
class WorkerStatus:
    name: str
    loaded: bool
    tool_count: int
    transport: str
    risk_default: str
    capability_tags: list[str]
    autoload: bool
    last_error: str | None = None


class WorkerManager:
    def __init__(self, registry: WorkerRegistry, tool_pool, executor,
                 *, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
                 disconnect_timeout: float = DEFAULT_DISCONNECT_TIMEOUT) -> None:
        self._registry = registry
        self._pool = tool_pool
        self._executor = executor
        self._connect_timeout = connect_timeout
        self._disconnect_timeout = disconnect_timeout
        self._loaded: dict[str, list[str]] = {}
        self._errors: dict[str, str] = {}
        # Held across a whole load/unload/reload body. The pool's per-worker
        # lock only guards connect+initialize; without this, a concurrent
        # unload's remove_worker can land after a load's add_all, leaving a
        # connected worker whose tools are invisible.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # --- queries ----------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def status(self) -> list[WorkerStatus]:
        out = []
        for spec in self._registry.all():
            out.append(WorkerStatus(
                name=spec.name,
                loaded=self.is_loaded(spec.name),
                tool_count=len(self._loaded.get(spec.name, [])),
                transport=spec.transport,
                risk_default=spec.risk_default,
                capability_tags=list(spec.capability_tags),
                autoload=spec.autoload,
                last_error=self._errors.get(spec.name),
            ))
        return out

    def tools_of(self, name: str) -> list[str]:
        return list(self._loaded.get(name, []))

    def unavailable_reason(self, name: str) -> str | None:
        """Advisory only — enforcement stays at RiskAwareToolPool.call_tool.

        Reports "loaded" only once the executor registration has completed, so
        callers that bypass the executor cannot dispatch during the load window
        while the tier table is still empty.
        """
        if self.is_loaded(name):
            return None
        try:
            self._registry.get(name)
        except WorkerNotFoundError:
            return (f"worker {name!r} is not declared in workers.yaml — "
                    f"declared workers: {sorted(s.name for s in self._registry.all())}")
        err = self._errors.get(name)
        tail = f" (last error: {err})" if err else ""
        return (f"worker {name!r} is not loaded — its tools are unavailable this "
                f"session. Ask the operator to run /worker load {name}.{tail}")

    def worker_of(self, tool_name: str) -> str | None:
        """Map a prefixed tool name back to its declared worker, longest first
        so a worker named `x` cannot claim `x_y_z` belonging to `x_y`."""
        for spec in sorted(self._registry.all(), key=lambda s: -len(s.name)):
            if tool_name.startswith(f"{spec.name}_"):
                return spec.name
        return None

    # --- lifecycle --------------------------------------------------------
    async def load(self, name: str) -> WorkerOpResult:
        try:
            spec = self._registry.get(name)
        except WorkerNotFoundError:
            return WorkerOpResult(
                "load", name, False, error_kind="unknown_worker",
                error=(f"no worker named {name!r}; declared: "
                       f"{sorted(s.name for s in self._registry.all())}"))
        async with self._locks[name]:
            if self.is_loaded(name):
                return WorkerOpResult("load", name, True,
                                      tool_count=len(self._loaded[name]),
                                      tools=list(self._loaded[name]))
            return await self._load_locked(spec)

    async def _load_locked(self, spec) -> WorkerOpResult:
        name = spec.name
        self._pool.add_spec(spec)
        try:
            await self._pool.connect(name, timeout=self._connect_timeout)
        except asyncio.TimeoutError:
            return await self._fail(name, "connect_timeout",
                                    f"worker {name!r} did not connect within "
                                    f"{self._connect_timeout}s")
        except FileNotFoundError as exc:
            return await self._fail(name, "spawn_failed", str(exc))
        except Exception as exc:
            kind = "protocol_mismatch" if "version" in str(exc).lower() else "spawn_failed"
            return await self._fail(name, kind, f"{type(exc).__name__}: {exc}")

        try:
            listing = await self._pool.list_tools(name)
        except Exception as exc:
            return await self._fail(name, "list_tools_failed",
                                    f"{type(exc).__name__}: {exc}")

        classes: list[type[Tool]] = []
        for tool in getattr(listing, "tools", []) or []:
            tool_name = getattr(tool, "name", None)
            if tool_name is None:
                continue
            classes.append(make_tool_class(spec, {
                "name": tool_name,
                "description": getattr(tool, "description", "") or "",
                "inputSchema": getattr(tool, "inputSchema", None)
                or {"type": "object", "properties": {}},
            }, self._pool))
        try:
            self._executor.add_all(classes)
        except ValueError as exc:
            return await self._fail(name, "tool_collision", str(exc))
        except RuntimeError as exc:
            return await self._fail(name, "spawn_failed", str(exc))

        names = [c.name for c in classes]
        self._loaded[name] = names
        self._errors.pop(name, None)
        self._pool.emit_lifecycle(name, "worker_loaded",
                                  args={"transport": spec.transport,
                                        "command": spec.command or spec.endpoint,
                                        "tool_count": len(names)})
        logger.info("loaded worker %s (%d tools)", name, len(names))
        return WorkerOpResult("load", name, True, tool_count=len(names), tools=names)

    async def _fail(self, name: str, kind: ErrorKind, message: str) -> WorkerOpResult:
        """Roll a partial load back to the unloaded state."""
        with contextlib.suppress(Exception):
            await self._pool.disconnect(name)
        self._executor.remove_worker(name)
        self._pool.remove_spec(name)
        self._loaded.pop(name, None)
        self._errors[name] = message
        logger.warning("worker %s load failed (%s): %s", name, kind, message)
        return WorkerOpResult("load", name, False, error=message, error_kind=kind)

    async def unload(self, name: str) -> WorkerOpResult:
        async with self._locks[name]:
            if not self.is_loaded(name):
                return WorkerOpResult("unload", name, True)
            # Order matters: everything before the disconnect is unconditional
            # and cannot hang, so a wedged teardown can never leave a worker
            # whose tools are gone but whose spec and approvals remain.
            removed = self._executor.remove_worker(name)
            self._pool.remove_spec(name)          # bumps generation, evicts approvals
            self._loaded.pop(name, None)
            err = kind = None
            try:
                await asyncio.wait_for(self._pool.disconnect(name),
                                       timeout=self._disconnect_timeout)
            except asyncio.TimeoutError:
                kind, err = "disconnect_timeout", (
                    f"worker {name!r} did not shut down within "
                    f"{self._disconnect_timeout}s; its process may still be running")
                self._errors[name] = err
                logger.warning(err)
            self._pool.emit_lifecycle(name, "worker_unloaded",
                                      detail=err, args={"tools_removed": removed})
            return WorkerOpResult("unload", name, err is None,
                                  tool_count=removed, error=err, error_kind=kind)

    async def reload(self, name: str) -> WorkerOpResult:
        un = await self.unload(name)
        if not un.ok:
            return WorkerOpResult("reload", name, False,
                                  error=f"unload half failed: {un.error}",
                                  error_kind=un.error_kind)
        res = await self.load(name)
        return WorkerOpResult("reload", name, res.ok, res.tool_count, res.tools,
                              res.error, res.error_kind)

    async def load_autoload(self) -> list[WorkerOpResult]:
        """Boot path. Gathers so boot costs max(), not sum()."""
        targets = [s.name for s in self._registry.all() if s.autoload]
        if not targets:
            return []
        return list(await asyncio.gather(*(self.load(n) for n in targets)))

    async def close_all(self) -> None:
        for name in list(self._loaded):
            with contextlib.suppress(Exception):
                await self.unload(name)
        with contextlib.suppress(Exception):
            await self._pool.close_all()
