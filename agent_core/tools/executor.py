"""Tool executor: registry + dispatch + exception containment.

The executor is constructed at agent startup via `ToolExecutor.build()`, which
unions builtins with agent-supplied tool classes, drops anything in the
`disabled` set, validates each tool's `requires` against the agent's attrs,
and instantiates the surviving classes. The executor is then attached to the
agent as `agent.tool_executor` and used by the agent's `handle_chat` to
dispatch tool calls returned by the model.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agent_core.tools.base import Tool
from agent_core.tools.builtin import BUILTIN_TOOLS

if TYPE_CHECKING:
    from agent_core.agent import HandlerContext


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools

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
            for attr in tool_cls.requires:
                if not hasattr(agent, attr):
                    raise RuntimeError(
                        f"Tool {tool_cls.name!r} requires agent.{attr!r}, "
                        f"but {type(agent).__name__} has no such attribute. "
                        f"Set it in setup() or add {tool_cls.name!r} to disabled_builtins."
                    )
            instances[tool_cls.name] = tool_cls()
        return cls(instances)

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
        return [type(t).to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)
