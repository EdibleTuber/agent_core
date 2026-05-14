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


def test_runtime_unions_class_tools_with_register_tools():
    """_attach_registries should union class-level cls.tools with
    register_tools() output."""
    from unittest.mock import MagicMock

    from agent_core.runtime import _attach_registries

    class _Mixed(Agent):
        name = "mixed"
        tools = [_Tool1]

        def register_tools(self):
            return [_Tool2]

    agent = _Mixed()
    # Stub framework managers that _attach_registries reads during
    # ToolExecutor.build and SystemPromptBuilder construction.
    for attr in ["profile", "wisdom", "channels", "learning", "allowlist",
                 "approval_registry", "inference", "retrieval", "websearch",
                 "config", "fetcher"]:
        setattr(agent, attr, MagicMock())

    _attach_registries(agent)
    registered_names = {t.name for t in agent.tool_executor._tools.values()}
    assert "tool1" in registered_names
    assert "tool2" in registered_names
