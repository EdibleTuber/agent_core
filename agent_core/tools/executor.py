"""Tool executor: registry + dispatch + exception containment.

The executor is constructed at agent startup via `ToolExecutor.build()`, which
unions builtins with agent-supplied tool classes, drops anything in the
`disabled` set, validates each tool's `requires` against the agent's attrs,
and instantiates the surviving classes. The executor is then attached to the
agent as `agent.tool_executor` and used by the agent's `handle_chat` to
dispatch tool calls returned by the model.

The registry is mutable at runtime: `add`/`add_all`/`remove`/`remove_worker`
let a worker's tools be loaded and unloaded while the daemon runs, without
rebuilding the executor.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agent_core.tools.base import Tool
from agent_core.tools.builtin import BUILTIN_TOOLS

if TYPE_CHECKING:
    from agent_core.agent import HandlerContext

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool], *, agent=None,
                 disabled: frozenset[str] = frozenset()) -> None:
        self._tools = tools
        # Retained so add() can run the same `requires` validation build() does
        # and honour the same `disabled` set. The executor is already stored on
        # the agent (runtime.py:50) and every tool receives it via ctx.agent, so
        # this reference adds no lifetime coupling that did not already exist.
        self._agent = agent
        self._disabled = disabled

    @classmethod
    def build(
        cls,
        agent,
        agent_tool_classes: list[type[Tool]],
        disabled: frozenset[str] = frozenset(),
    ) -> "ToolExecutor":
        all_classes = [
            t for t in BUILTIN_TOOLS + list(agent_tool_classes) if t.name not in disabled
        ]
        instances: dict[str, Tool] = {}
        for tool_cls in all_classes:
            cls._validate_requires(agent, tool_cls)
            if tool_cls.name in instances:
                # 1.8.0 warns; raising here would break a consumer that
                # deliberately shadows a builtin, which is a 2.0.0 change.
                logger.warning(
                    "tool name collision: %r from %s replaces %s — the earlier "
                    "tool is now unreachable",
                    tool_cls.name, tool_cls.__name__,
                    type(instances[tool_cls.name]).__name__,
                )
            instances[tool_cls.name] = tool_cls()
        return cls(instances, agent=agent, disabled=disabled)

    @staticmethod
    def _validate_requires(agent, tool_cls: type[Tool]) -> None:
        for attr in tool_cls.requires:
            if not hasattr(agent, attr):
                raise RuntimeError(
                    f"Tool {tool_cls.name!r} requires agent.{attr!r}, "
                    f"but {type(agent).__name__} has no such attribute. "
                    f"Add it in setup(), or remove {tool_cls.name!r} from tools / disabled_builtins."
                )

    # --- runtime mutation -------------------------------------------------
    def add(self, tool_cls: type[Tool]) -> None:
        """Register one tool. Raises on collision or unmet requires."""
        if tool_cls.name in self._disabled:
            return
        if tool_cls.name in self._tools:
            existing = type(self._tools[tool_cls.name])
            raise ValueError(
                f"tool name collision: {tool_cls.name!r} is already registered by "
                f"{existing.__name__}; refusing to shadow it silently"
            )
        self._validate_requires(self._agent, tool_cls)
        self._tools[tool_cls.name] = tool_cls()

    def add_all(self, tool_classes: list[type[Tool]]) -> None:
        """Validate every class, then commit — all or nothing.

        Atomicity lives here rather than in the caller because the dict lives
        here; a worker advertising one shadowing tool among ten is exactly the
        partial-add this prevents.
        """
        pending = [t for t in tool_classes if t.name not in self._disabled]
        seen: set[str] = set()
        for tool_cls in pending:
            if tool_cls.name in self._tools or tool_cls.name in seen:
                raise ValueError(
                    f"tool name collision: {tool_cls.name!r} is already registered"
                )
            self._validate_requires(self._agent, tool_cls)
            seen.add(tool_cls.name)
        for tool_cls in pending:
            self._tools[tool_cls.name] = tool_cls()

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def remove_worker(self, worker: str) -> int:
        """Remove every tool synthesized for `worker`, by provenance."""
        doomed = [n for n, t in self._tools.items()
                  if getattr(type(t), "worker", None) == worker]
        for name in doomed:
            del self._tools[name]
        return len(doomed)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def run(self, name: str, arguments: dict, ctx: "HandlerContext") -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            return await tool.run(arguments, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return f"Error in {name}: {exc}"

    def schemas(self) -> list[dict]:
        """Stable order: non-worker tools in insertion order, then worker tools
        sorted by (worker, name).

        This is a prompt prefix. Dict insertion order would reshuffle it on
        every load/unload, and load_autoload() registers workers concurrently,
        so registration order is not even deterministic. names() deliberately
        keeps insertion order — it has no consumer-visible meaning.
        """
        plain, owned = [], []
        for tool in self._tools.values():
            (owned if getattr(type(tool), "worker", None) else plain).append(tool)
        owned.sort(key=lambda t: (type(t).worker, type(t).name))
        return [type(t).to_openai_schema() for t in plain + owned]

    def names(self) -> list[str]:
        return list(self._tools)
