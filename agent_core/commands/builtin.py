"""Builtin command list."""
from agent_core.commands.base import Command
from agent_core.commands._builtin_impls import (
    Clear,
    Context,
    Help,
    Learnings,
    Model,
    Profile,
    Promote,
    Quit,
    Rate,
    Scratch,
    Status,
    Think,
    Wisdom,
)

BUILTIN_COMMANDS: list[type[Command]] = [
    Help,
    Clear,
    Status,
    Context,
    Profile,
    Scratch,
    Wisdom,
    Learnings,
    Promote,
    Rate,
    Model,
    Think,
    Quit,
]
