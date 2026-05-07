"""Tests for the cat shell tool."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_core.tools._shell import Cat


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


async def test_cat_returns_file_contents(tmp_path):
    (tmp_path / "x.md").write_text("hello world")
    agent = _Agent(tmp_path)
    result = await Cat().run({"path": "x.md"}, _ctx(agent))
    assert result == "hello world"


async def test_cat_rejects_path_escape(tmp_path):
    agent = _Agent(tmp_path)
    result = await Cat().run({"path": "../../../etc/passwd"}, _ctx(agent))
    assert "outside vault" in result.lower() or "escape" in result.lower()


async def test_cat_rejects_system_path(tmp_path):
    (tmp_path / "_index.md").write_text("internal")
    agent = _Agent(tmp_path)
    result = await Cat().run({"path": "_index.md"}, _ctx(agent))
    assert "system path" in result.lower()


async def test_cat_missing_file(tmp_path):
    agent = _Agent(tmp_path)
    result = await Cat().run({"path": "nope.md"}, _ctx(agent))
    assert "not found" in result.lower()


async def test_cat_truncates_large_file(tmp_path):
    big = "x" * (40 * 1024)
    (tmp_path / "big.md").write_text(big)
    agent = _Agent(tmp_path)
    result = await Cat().run({"path": "big.md"}, _ctx(agent))
    assert "[output truncated:" in result


async def test_cat_requires_path(tmp_path):
    agent = _Agent(tmp_path)
    result = await Cat().run({}, _ctx(agent))
    assert "path" in result.lower() and "required" in result.lower()
