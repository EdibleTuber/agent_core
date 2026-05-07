"""Tests for CommandRegistry."""
from typing import AsyncIterator

import pytest

from agent_core.commands.base import Command
from agent_core.commands.registry import CommandRegistry
from agent_core.protocol.messages import ResponseMessage


class _Hello(Command):
    name = "hello"
    args = "[<name>]"
    description = "Say hello"

    async def run(self, raw_args, ctx) -> AsyncIterator:
        target = raw_args.strip() or "world"
        yield ResponseMessage(text=f"hi {target}")


class _Multi(Command):
    name = "multi"
    args = ""
    description = "Yields multiple"

    async def run(self, raw_args, ctx) -> AsyncIterator:
        yield ResponseMessage(text="one")
        yield ResponseMessage(text="two")
        yield ResponseMessage(text="three")


class _NeedsCompiler(Command):
    name = "needs_compiler"
    args = ""
    description = "Requires compiler"
    requires = ("compiler",)

    async def run(self, raw_args, ctx) -> AsyncIterator:
        yield ResponseMessage(text=f"compiler: {ctx.agent.compiler}")


class _StubAgent:
    pass


# Autouse fixture: neutralize BUILTIN_COMMANDS for all tests in this module
@pytest.fixture(autouse=True)
def _empty_builtin_commands(monkeypatch):
    monkeypatch.setattr("agent_core.commands.builtin.BUILTIN_COMMANDS", [])
    monkeypatch.setattr("agent_core.commands.registry.BUILTIN_COMMANDS", [])


async def _collect(it):
    out = []
    async for m in it:
        out.append(m)
    return out


async def test_dispatch_known_command():
    registry = CommandRegistry({"hello": _Hello()})
    msgs = await _collect(registry.dispatch("hello", "PAL", ctx=None))
    assert len(msgs) == 1
    assert msgs[0].text == "hi PAL"


async def test_dispatch_yields_multiple_messages():
    registry = CommandRegistry({"multi": _Multi()})
    msgs = await _collect(registry.dispatch("multi", "", ctx=None))
    assert [m.text for m in msgs] == ["one", "two", "three"]


async def test_dispatch_unknown_command_yields_response():
    registry = CommandRegistry({})
    msgs = await _collect(registry.dispatch("nope", "", ctx=None))
    assert len(msgs) == 1
    assert isinstance(msgs[0], ResponseMessage)
    assert "unknown" in msgs[0].text.lower()


def test_metadata_returns_tuples():
    registry = CommandRegistry({"hello": _Hello()})
    assert registry.metadata() == [("hello", "[<name>]", "Say hello")]


def test_metadata_preserves_registration_order():
    class A(Command):
        name = "a"; args = ""; description = ""
        async def run(self, raw_args, ctx): yield None
    class B(Command):
        name = "b"; args = ""; description = ""
        async def run(self, raw_args, ctx): yield None
    class C(Command):
        name = "c"; args = ""; description = ""
        async def run(self, raw_args, ctx): yield None
    agent = _StubAgent()
    registry = CommandRegistry.build(agent, [A, B, C])
    assert [m[0] for m in registry.metadata()] == ["a", "b", "c"]


def test_build_validates_requires():
    agent = _StubAgent()
    with pytest.raises(RuntimeError, match="needs_compiler.*compiler"):
        CommandRegistry.build(agent, [_NeedsCompiler])


def test_build_excludes_disabled():
    agent = _StubAgent()
    registry = CommandRegistry.build(agent, [_Hello], disabled=frozenset({"hello"}))
    assert registry.metadata() == []
