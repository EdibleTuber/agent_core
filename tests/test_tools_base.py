"""Tests for agent_core.tools.base."""
import pytest

from agent_core.tools.base import Tool


def test_tool_subclass_inherits_classvars():
    class MyTool(Tool):
        name = "my_tool"
        description = "A test tool"
        parameters = {"type": "object", "properties": {}, "required": []}

    assert MyTool.name == "my_tool"
    assert MyTool.description == "A test tool"
    assert MyTool.requires == ()


def test_tool_to_openai_schema():
    class MyTool(Tool):
        name = "my_tool"
        description = "A test tool"
        parameters = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    schema = MyTool.to_openai_schema()
    assert schema == {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        },
    }


def test_tool_run_not_implemented():
    class MyTool(Tool):
        name = "my_tool"
        description = "A test tool"
        parameters = {}

    import asyncio
    with pytest.raises(NotImplementedError):
        asyncio.run(MyTool().run({}, ctx=None))
