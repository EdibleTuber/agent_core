"""Command base class.

Commands are user-typed slash commands. Each Command class declares its
`name`, `args` (template string for /help, e.g. "<title>"), `description`,
and optional `requires`. The `run` method takes a raw string arg and yields
zero or more messages.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from agent_core.agent import HandlerContext


class Command:
    name: ClassVar[str]
    args: ClassVar[str]
    description: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    async def run(self, raw_args: str, ctx: "HandlerContext") -> AsyncIterator:
        raise NotImplementedError
        yield   # makes this an async generator
