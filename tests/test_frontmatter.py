"""Tests for YAML frontmatter parsing and serialization."""
from agent_core.utils.frontmatter import parse_frontmatter, serialize_frontmatter


def test_parse_frontmatter_basic():
    content = """---
title: Test Article
tags: [python, testing]
---

# Test Article

Body text here.
"""
    meta, body = parse_frontmatter(content)
    assert meta["title"] == "Test Article"
    assert meta["tags"] == ["python", "testing"]
    assert body.strip() == "# Test Article\n\nBody text here."


def test_parse_frontmatter_empty():
    content = "# No frontmatter\n\nJust body."
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert body == content


def test_parse_frontmatter_empty_yaml():
    content = "---\n---\n\nBody only."
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert body.strip() == "Body only."


def test_parse_frontmatter_preserves_body_whitespace():
    content = "---\ntitle: X\n---\n\nLine one.\n\nLine two.\n"
    meta, body = parse_frontmatter(content)
    assert meta["title"] == "X"
    assert "\n\nLine one.\n\nLine two.\n" == body


def test_serialize_frontmatter():
    meta = {"title": "My Article", "tags": ["ai", "wiki"]}
    body = "# My Article\n\nContent here.\n"
    result = serialize_frontmatter(meta, body)
    assert result.startswith("---\n")
    assert "title: My Article" in result
    assert result.endswith("\n# My Article\n\nContent here.\n")
    # Round-trip
    parsed_meta, parsed_body = parse_frontmatter(result)
    assert parsed_meta["title"] == "My Article"
    assert parsed_meta["tags"] == ["ai", "wiki"]
    assert parsed_body.strip() == "# My Article\n\nContent here."


def test_serialize_frontmatter_empty_meta():
    body = "# Just body\n"
    result = serialize_frontmatter({}, body)
    assert result == "# Just body\n"


def test_parse_frontmatter_with_dashes_in_body():
    content = "---\ntitle: Test\n---\n\nSome text with --- dashes in it.\n"
    meta, body = parse_frontmatter(content)
    assert meta["title"] == "Test"
    assert "--- dashes" in body
