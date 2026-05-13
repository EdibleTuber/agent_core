"""Tests for head and tail builtin tools."""
from dataclasses import dataclass
from pathlib import Path

from agent_core.tools._shell import Head, Tail


@dataclass
class _Config:
    vault_path: Path


class _Agent:
    def __init__(self, vault_path):
        self.config = _Config(vault_path)


def _ctx(agent):
    class _C: pass
    c = _C(); c.agent = agent; return c


async def test_head_default_20_lines(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(f"line {i}" for i in range(50)))
    agent = _Agent(tmp_path)
    result = await Head().run({"path": "x.md"}, _ctx(agent))
    assert result.splitlines() == [f"line {i}" for i in range(20)]


async def test_head_explicit_lines(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(f"line {i}" for i in range(50)))
    agent = _Agent(tmp_path)
    result = await Head().run({"path": "x.md", "lines": 5}, _ctx(agent))
    assert result.splitlines() == [f"line {i}" for i in range(5)]


async def test_head_short_file(tmp_path):
    (tmp_path / "x.md").write_text("only\ntwo")
    agent = _Agent(tmp_path)
    result = await Head().run({"path": "x.md", "lines": 100}, _ctx(agent))
    assert result == "only\ntwo"


async def test_tail_default_20_lines(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(f"line {i}" for i in range(50)))
    agent = _Agent(tmp_path)
    result = await Tail().run({"path": "x.md"}, _ctx(agent))
    assert result.splitlines() == [f"line {i}" for i in range(30, 50)]


async def test_tail_explicit_lines(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(f"line {i}" for i in range(50)))
    agent = _Agent(tmp_path)
    result = await Tail().run({"path": "x.md", "lines": 3}, _ctx(agent))
    assert result.splitlines() == ["line 47", "line 48", "line 49"]


async def test_head_rejects_escape(tmp_path):
    agent = _Agent(tmp_path)
    result = await Head().run({"path": "../../etc/passwd"}, _ctx(agent))
    assert "outside vault" in result.lower()


async def test_tail_missing_file(tmp_path):
    agent = _Agent(tmp_path)
    result = await Tail().run({"path": "nope.md"}, _ctx(agent))
    assert "not found" in result.lower()


async def test_head_404_includes_suggestions(tmp_path):
    """Head 404 (via _read_safe) gets the suggestion treatment too."""
    (tmp_path / "vibe-coding.md").write_text("body")
    agent = _Agent(tmp_path)
    result = await Head().run({"path": "vibe_coding.md"}, _ctx(agent))
    assert "File not found: vibe_coding.md" in result
    assert "Did you mean: " in result
    assert "vibe-coding.md" in result


async def test_tail_404_includes_suggestions(tmp_path):
    """Tail 404 (via _read_safe) gets the suggestion treatment."""
    (tmp_path / "notes.md").write_text("body")
    agent = _Agent(tmp_path)
    result = await Tail().run({"path": "nottes.md"}, _ctx(agent))
    assert "File not found: nottes.md" in result
    assert "notes.md" in result
