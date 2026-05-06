"""Test that BUILTIN_TOOLS is correctly populated."""
from agent_core.tools.builtin import BUILTIN_TOOLS


def test_builtin_tools_count():
    assert len(BUILTIN_TOOLS) == 12


def test_builtin_tools_names():
    names = {t.name for t in BUILTIN_TOOLS}
    expected = {
        "cat", "head", "tail", "ls", "grep", "find", "read_lines",  # shell
        "fetch_url", "search_vault", "search_web", "update_scratch", "add_learning",  # framework-backed
    }
    assert names == expected


def test_builtin_tools_unique_names():
    names = [t.name for t in BUILTIN_TOOLS]
    assert len(names) == len(set(names))
