"""Tests for Command base class."""
from agent_core.commands.base import Command


def test_command_subclass_inherits_classvars():
    class Help(Command):
        name = "help"
        args = ""
        description = "Show help"

    assert Help.name == "help"
    assert Help.args == ""
    assert Help.description == "Show help"
    assert Help.requires == ()
