"""Tests for the register_tools() lifecycle hook on the Agent base class.

register_tools() returns a list of Tool subclasses constructed at runtime,
unioned with the class-level cls.tools by the framework during registry
attachment. Existing agents (PAL) that use only declarative cls.tools
continue to work unchanged.
"""
from agent_core.agent import Agent
from agent_core.tools.base import Tool


class _Tool1(Tool):
    name = "tool1"
    description = "first"
    parameters = {"type": "object", "properties": {}}

    async def run(self, args, ctx):
        return "1"


class _Tool2(Tool):
    name = "tool2"
    description = "second"
    parameters = {"type": "object", "properties": {}}

    async def run(self, args, ctx):
        return "2"


def test_register_tools_default_returns_empty_list():
    """A bare Agent subclass returns no dynamic tools."""
    class _Bare(Agent):
        name = "bare"

    agent = _Bare()
    assert agent.register_tools() == []


def test_register_tools_override_returns_supplied_tools():
    """A subclass overriding register_tools() returns its list."""
    class _Dynamic(Agent):
        name = "dynamic"

        def register_tools(self):
            return [_Tool1, _Tool2]

    agent = _Dynamic()
    assert agent.register_tools() == [_Tool1, _Tool2]


def test_register_tools_coexists_with_class_tools():
    """Subclasses can declare cls.tools AND override register_tools()."""
    class _Hybrid(Agent):
        name = "hybrid"
        tools = [_Tool1]

        def register_tools(self):
            return [_Tool2]

    agent = _Hybrid()
    assert _Hybrid.tools == [_Tool1]
    assert agent.register_tools() == [_Tool2]
