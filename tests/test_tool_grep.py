"""Tests for the grep builtin tool."""
from dataclasses import dataclass
from pathlib import Path

from agent_core.tools._shell import Grep


@dataclass
class _Config:
    vault_path: Path


class _Agent:
    def __init__(self, vault_path):
        self.config = _Config(vault_path)


def _ctx(agent):
    class _C: pass
    c = _C(); c.agent = agent; return c


async def test_grep_finds_substring(tmp_path):
    (tmp_path / "x.md").write_text("apple\nbanana\ncherry")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "banana"}, _ctx(agent))
    assert "x.md:2:" in result
    assert "banana" in result


async def test_grep_no_match(tmp_path):
    (tmp_path / "x.md").write_text("apple")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "zzz"}, _ctx(agent))
    assert "no match" in result.lower() or result.strip() == ""


async def test_grep_case_insensitive(tmp_path):
    (tmp_path / "x.md").write_text("APPLE")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "apple", "ignore_case": True}, _ctx(agent))
    assert "x.md:1:" in result


async def test_grep_regex(tmp_path):
    (tmp_path / "x.md").write_text("foo123\nfoo456\nbar")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": r"foo\d+", "regex": True}, _ctx(agent))
    assert "foo123" in result
    assert "foo456" in result
    assert "bar" not in result


async def test_grep_subdir(tmp_path):
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "x.md").write_text("hit")
    (tmp_path / "Other").mkdir()
    (tmp_path / "Other" / "y.md").write_text("hit")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "hit", "path": "Notes"}, _ctx(agent))
    assert "Notes/x.md" in result
    assert "Other/y.md" not in result


async def test_grep_skips_system_paths(tmp_path):
    (tmp_path / "_index.md").write_text("hit")
    (tmp_path / "x.md").write_text("hit")
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "hit"}, _ctx(agent))
    assert "_index.md" not in result
    assert "x.md" in result


async def test_grep_invalid_regex(tmp_path):
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "[", "regex": True}, _ctx(agent))
    assert "regex" in result.lower() or "invalid" in result.lower()


async def test_grep_max_hits(tmp_path):
    (tmp_path / "x.md").write_text("\n".join(["match"] * 200))
    agent = _Agent(tmp_path)
    result = await Grep().run({"pattern": "match", "max_hits": 10}, _ctx(agent))
    hit_lines = [l for l in result.splitlines() if l.startswith("x.md:")]
    assert len(hit_lines) == 10
