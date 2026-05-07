"""Tool base class.

Tools are class-based extension points. An agent registers a list of Tool
subclasses on its class via `tools = [...]`. The framework instantiates them,
validates their `requires` against the agent's attributes, and exposes an
executor that dispatches by tool name.

Tools access dependencies through `ctx.agent.X` at runtime. The `requires`
tuple lists attribute names that must exist on the agent at registration time;
missing requirements fail fast inside `run_daemon()`, before any user message
is processed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from agent_core.agent import HandlerContext


class Tool:
    """Base class for agent tools.

    Subclasses set `name`, `description`, `parameters` (JSON Schema, OpenAI
    function-calling format), and optionally `requires` (a tuple of attribute
    names that must exist on the agent).

    Implement `run(args, ctx)` as an async method. It must return a string;
    errors that should reach the LLM are returned as descriptive strings, not
    raised. Unhandled exceptions are caught by the executor and converted.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]
    requires: ClassVar[tuple[str, ...]] = ()

    async def run(self, args: dict, ctx: "HandlerContext") -> str:
        raise NotImplementedError

    @classmethod
    def to_openai_schema(cls) -> dict:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters,
            },
        }
