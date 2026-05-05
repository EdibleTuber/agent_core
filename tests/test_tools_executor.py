"""Tests for agent_core.tools.executor."""
import asyncio
import pytest

from agent_core.tools.base import Tool
from agent_core.tools.executor import ToolExecutor


@pytest.fixture(autouse=True)
def _empty_builtin_tools(monkeypatch):
    """Isolate executor tests from the live BUILTIN_TOOLS list.

    Tests in this module construct ToolExecutor.build() to assert exact
    membership; once Task 11 populates BUILTIN_TOOLS with 12 entries,
    those assertions would otherwise silently include the builtins.
    """
    monkeypatch.setattr("agent_core.tools.builtin.BUILTIN_TOOLS", [])
    monkeypatch.setattr("agent_core.tools.executor.BUILTIN_TOOLS", [])


class _Echo(Tool):
    name = "echo"
    description = "Echoes its input"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def run(self, args, ctx):
        return f"echo: {args['text']}"


class _NeedsCompiler(Tool):
    name = "needs_compiler"
    description = "Requires compiler"
    parameters = {}
    requires = ("compiler",)

    async def run(self, args, ctx):
        return f"compiler: {ctx.agent.compiler}"


class _Boom(Tool):
    name = "boom"
    description = "Raises"
    parameters = {}

    async def run(self, args, ctx):
        raise ValueError("kaboom")


class _CancelledTool(Tool):
    name = "cancel"
    description = "Raises CancelledError"
    parameters = {}

    async def run(self, args, ctx):
        raise asyncio.CancelledError()


class _StubAgent:
    pass


def test_build_validates_requires_present():
    agent = _StubAgent()
    agent.compiler = object()
    executor = ToolExecutor.build(agent, [_NeedsCompiler])
    assert "needs_compiler" in executor.names()


def test_build_fails_when_requires_missing():
    agent = _StubAgent()  # no .compiler
    with pytest.raises(RuntimeError, match="needs_compiler.*compiler"):
        ToolExecutor.build(agent, [_NeedsCompiler])


def test_build_excludes_disabled():
    agent = _StubAgent()
    executor = ToolExecutor.build(agent, [_Echo], disabled=frozenset({"echo"}))
    assert executor.names() == []


async def test_run_unknown_returns_string():
    executor = ToolExecutor({})
    result = await executor.run("nope", {}, ctx=None)
    assert result == "Unknown tool: nope"


async def test_run_executes_tool():
    executor = ToolExecutor({"echo": _Echo()})
    result = await executor.run("echo", {"text": "hi"}, ctx=None)
    assert result == "echo: hi"


async def test_run_catches_exceptions():
    executor = ToolExecutor({"boom": _Boom()})
    result = await executor.run("boom", {}, ctx=None)
    assert result == "Error in boom: kaboom"


async def test_run_does_not_swallow_cancellation():
    executor = ToolExecutor({"cancel": _CancelledTool()})
    with pytest.raises(asyncio.CancelledError):
        await executor.run("cancel", {}, ctx=None)


def test_schemas_returns_openai_format():
    executor = ToolExecutor({"echo": _Echo()})
    schemas = executor.schemas()
    assert schemas == [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echoes its input",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }]


def test_names_preserves_insertion_order():
    class A(Tool):
        name = "a"; description = ""; parameters = {}
        async def run(self, args, ctx): return ""
    class B(Tool):
        name = "b"; description = ""; parameters = {}
        async def run(self, args, ctx): return ""
    class C(Tool):
        name = "c"; description = ""; parameters = {}
        async def run(self, args, ctx): return ""
    agent = _StubAgent()
    executor = ToolExecutor.build(agent, [A, B, C])
    assert executor.names() == ["a", "b", "c"]
