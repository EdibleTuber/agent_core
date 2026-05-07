"""Tests for SystemPromptBuilder render helpers."""
from pathlib import Path
from unittest.mock import MagicMock

from agent_core.prompts.builder import SystemPromptBuilder


def _builder(profile=None, wisdom=None, channels=None, tool_executor=None,
             command_registry=None, agent=None):
    return SystemPromptBuilder(
        profile=profile or MagicMock(),
        wisdom=wisdom or MagicMock(),
        channels=channels or MagicMock(),
        tool_executor=tool_executor or MagicMock(),
        command_registry=command_registry or MagicMock(),
        agent=agent or MagicMock(),
    )


def test_render_profile_empty():
    profile = MagicMock(); profile.read.return_value = ""
    assert _builder(profile=profile).render_profile() == ""


def test_render_profile_populated():
    profile = MagicMock(); profile.read.return_value = "I am Shane."
    out = _builder(profile=profile).render_profile()
    assert "## About the User" in out
    assert "I am Shane." in out


def test_render_wisdom_empty():
    wisdom = MagicMock(); wisdom.bodies.return_value = []
    assert _builder(wisdom=wisdom).render_wisdom() == ""


def test_render_wisdom_populated():
    wisdom = MagicMock(); wisdom.bodies.return_value = ["W1.", "W2."]
    out = _builder(wisdom=wisdom).render_wisdom()
    assert "## Active Wisdom" in out
    assert "- W1." in out
    assert "- W2." in out


def test_render_scratchpad_empty(tmp_path, monkeypatch):
    # Scratchpad reads from disk; with no file, .read() returns "".
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=2048)
    agent = MagicMock(name="test", config=cfg)
    agent.name = "test"
    out = _builder(agent=agent).render_scratchpad("c1")
    assert out == ""


def test_render_scratchpad_populated(tmp_path):
    # Create a real Scratchpad file at the expected path so the builder's
    # construction reads back what we wrote.
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=2048)
    agent = MagicMock(config=cfg)
    agent.name = "test"
    # Construct a Scratchpad and write some content
    from agent_core.scratchpad import Scratchpad
    sp = Scratchpad(vault_path=tmp_path, agent_name="test",
                    channel_id="c1", max_bytes=2048)
    sp.write("scratch contents")
    out = _builder(agent=agent).render_scratchpad("c1")
    assert "## Channel Scratchpad" in out
    assert "scratch contents" in out


def test_render_commands_catalog():
    cr = MagicMock()
    cr.metadata.return_value = [("hello", "[<name>]", "Say hi"), ("quit", "", "Exit")]
    out = _builder(command_registry=cr).render_commands_catalog()
    assert "## Available Commands" in out
    assert "/hello [<name>]" in out
    assert "Say hi" in out
    assert "/quit" in out
    assert "Exit" in out


def test_render_commands_catalog_preserves_order():
    cr = MagicMock()
    cr.metadata.return_value = [("a", "", ""), ("b", "", ""), ("c", "", "")]
    out = _builder(command_registry=cr).render_commands_catalog()
    a_pos = out.index("/a")
    b_pos = out.index("/b")
    c_pos = out.index("/c")
    assert a_pos < b_pos < c_pos


def test_render_commands_catalog_empty():
    cr = MagicMock(); cr.metadata.return_value = []
    assert _builder(command_registry=cr).render_commands_catalog() == ""


def test_render_tools_catalog():
    te = MagicMock()
    te.schemas.return_value = [
        {"type": "function", "function": {"name": "cat", "description": "Read file", "parameters": {}}},
        {"type": "function", "function": {"name": "grep", "description": "Search files", "parameters": {}}},
    ]
    out = _builder(tool_executor=te).render_tools_catalog()
    assert "## Available Tools" in out
    assert "`cat`" in out
    assert "Read file" in out
    assert "`grep`" in out
    assert "Search files" in out


def test_render_tools_catalog_empty():
    te = MagicMock(); te.schemas.return_value = []
    assert _builder(tool_executor=te).render_tools_catalog() == ""
