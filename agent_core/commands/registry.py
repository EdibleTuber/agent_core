"""Command registry: registration + dispatch.

Mirror of ToolExecutor. `build()` validates requires and instantiates
commands. `dispatch()` looks up by name and yields the command's messages.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_core.commands.base import Command
from agent_core.commands.builtin import BUILTIN_COMMANDS
from agent_core.protocol.messages import ResponseMessage

if TYPE_CHECKING:
    from agent_core.agent import HandlerContext


class CommandRegistry:
    def __init__(self, commands: dict[str, Command]) -> None:
        self._commands = commands

    @classmethod
    def build(
        cls,
        agent,
        agent_command_classes: list[type[Command]],
        disabled: frozenset[str] = frozenset(),
    ) -> "CommandRegistry":
        all_classes = [
            c for c in BUILTIN_COMMANDS + list(agent_command_classes) if c.name not in disabled
        ]
        instances: dict[str, Command] = {}
        for cmd_cls in all_classes:
            for attr in cmd_cls.requires:
                if not hasattr(agent, attr):
                    raise RuntimeError(
                        f"Command {cmd_cls.name!r} requires agent.{attr!r}, "
                        f"but {type(agent).__name__} has no such attribute. "
                        f"Add it in setup(), or remove {cmd_cls.name!r} from commands / disabled_builtins."
                    )
            instances[cmd_cls.name] = cmd_cls()
        return cls(instances)

    async def dispatch(self, name: str, raw_args: str, ctx: "HandlerContext"):
        command = self._commands.get(name)
        if command is None:
            yield ResponseMessage(text=f"Unknown command: {name}")
            return
        async for msg in command.run(raw_args, ctx):
            yield msg

    def metadata(self) -> list[tuple[str, str, str]]:
        return [
            (type(c).name, type(c).args, type(c).description)
            for c in self._commands.values()
        ]

    def names(self) -> list[str]:
        return list(self._commands)
