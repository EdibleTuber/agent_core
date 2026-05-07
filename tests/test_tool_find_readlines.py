"""Tests for find and read_lines builtin tools."""
from dataclasses import dataclass
from pathlib import Path

from agent_core.tools._shell import Find, ReadLines


@dataclass
class _Config:
    vault_path: Path


class _Agent:
    def __init__(self, vault_path):
        self.config = _Config(vault_path)


def _ctx(agent):
    class _C: pass
    c = _C(); c.agent = agent; return c


async def test_find_glob_match(tmp_path):
    (tmp_path / "agent-1.md").write_text("a")
    (tmp_path / "agent-2.md").write_text("b")
    (tmp_path / "other.md").write_text("c")
    agent = _Agent(tmp_path)
    result = await Find().run({"pattern": "agent-*.md"}, _ctx(agent))
    lines = set(result.splitlines())
    assert "agent-1.md" in lines
    assert "agent-2.md" in lines
    assert "other.md" not in lines


async def test_find_recursive_glob(tmp_path):
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "quantum-1.md").write_text("a")
    (tmp_path / "Other").mkdir()
    (tmp_path / "Other" / "quantum-2.md").write_text("b")
    agent = _Agent(tmp_path)
    result = await Find().run({"pattern": "**/quantum*"}, _ctx(agent))
    lines = set(result.splitlines())
    assert "Notes/quantum-1.md" in lines
    assert "Other/quantum-2.md" in lines


async def test_find_skips_system_paths(tmp_path):
    (tmp_path / "_index.md").write_text("a")
    (tmp_path / "x.md").write_text("b")
    agent = _Agent(tmp_path)
    result = await Find().run({"pattern": "*.md"}, _ctx(agent))
    lines = set(result.splitlines())
    assert "_index.md" not in lines
    assert "x.md" in lines


async def test_find_type_filter(tmp_path):
    (tmp_path / "f.md").write_text("a")
    (tmp_path / "d").mkdir()
    agent = _Agent(tmp_path)
    files_only = await Find().run({"pattern": "*", "type": "f"}, _ctx(agent))
    dirs_only = await Find().run({"pattern": "*", "type": "d"}, _ctx(agent))
    assert "f.md" in files_only and "d" not in files_only.splitlines()
    assert "d" in dirs_only.splitlines() and "f.md" not in dirs_only


async def test_find_no_match(tmp_path):
    agent = _Agent(tmp_path)
    result = await Find().run({"pattern": "nope*"}, _ctx(agent))
    assert "no match" in result.lower() or result.strip() == ""


async def test_read_lines_range(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(f"line {i}" for i in range(1, 21)))
    agent = _Agent(tmp_path)
    result = await ReadLines().run({"path": "x.md", "start": 5, "end": 7}, _ctx(agent))
    assert "5: line 5" in result
    assert "6: line 6" in result
    assert "7: line 7" in result
    assert "4: line 4" not in result
    assert "8: line 8" not in result


async def test_read_lines_bounds_clamped(tmp_path):
    (tmp_path / "x.md").write_text("a\nb\nc")
    agent = _Agent(tmp_path)
    result = await ReadLines().run({"path": "x.md", "start": 1, "end": 100}, _ctx(agent))
    assert "1: a" in result
    assert "2: b" in result
    assert "3: c" in result


async def test_read_lines_invalid_range(tmp_path):
    (tmp_path / "x.md").write_text("a")
    agent = _Agent(tmp_path)
    result = await ReadLines().run({"path": "x.md", "start": 5, "end": 2}, _ctx(agent))
    assert "invalid" in result.lower() or "range" in result.lower()
