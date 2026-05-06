"""Tests for the ls builtin tool."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_core.tools._shell import Ls


@dataclass
class _Config:
    vault_path: Path


class _Agent:
    def __init__(self, vault_path):
        self.config = _Config(vault_path)


def _ctx(agent):
    class _C:
        pass
    c = _C()
    c.agent = agent
    return c


async def test_ls_root(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "Notes").mkdir()
    agent = _Agent(tmp_path)
    result = await Ls().run({}, _ctx(agent))
    lines = set(result.splitlines())
    assert "a.md" in lines
    assert "b.md" in lines
    assert "Notes/" in lines


async def test_ls_subdir(tmp_path):
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "x.md").write_text("x")
    agent = _Agent(tmp_path)
    result = await Ls().run({"path": "Notes"}, _ctx(agent))
    assert result.strip() == "x.md"


async def test_ls_hides_system_paths_by_default(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "_index.md").write_text("internal")
    agent = _Agent(tmp_path)
    result = await Ls().run({}, _ctx(agent))
    assert "_index.md" not in result
    assert "a.md" in result


async def test_ls_show_hidden(tmp_path):
    (tmp_path / "_index.md").write_text("internal")
    agent = _Agent(tmp_path)
    result = await Ls().run({"show_hidden": True}, _ctx(agent))
    assert "_index.md" in result


async def test_ls_long_format(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    agent = _Agent(tmp_path)
    result = await Ls().run({"long": True}, _ctx(agent))
    assert "a.md" in result
    assert any(c.isdigit() for c in result)


async def test_ls_rejects_escape(tmp_path):
    agent = _Agent(tmp_path)
    result = await Ls().run({"path": "../.."}, _ctx(agent))
    assert "outside vault" in result.lower()


async def test_ls_caps_at_500_entries(tmp_path):
    for i in range(600):
        (tmp_path / f"f{i:04}.md").write_text(".")
    agent = _Agent(tmp_path)
    result = await Ls().run({}, _ctx(agent))
    assert "[output truncated:" in result or "more entries" in result.lower()
