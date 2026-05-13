"""Tests for shell tool helpers."""
import pytest

from agent_core.tools._shell_helpers import (
    OUTPUT_CAP_BYTES,
    cap_output,
    is_system_path,
    resolve_safe,
)


def test_resolve_safe_inside_vault(tmp_path):
    f = tmp_path / "Notes" / "x.md"
    f.parent.mkdir()
    f.write_text("hi")
    resolved = resolve_safe(tmp_path, "Notes/x.md")
    assert resolved == f.resolve()


def test_resolve_safe_rejects_escape(tmp_path):
    assert resolve_safe(tmp_path, "../../../etc/passwd") is None


def test_resolve_safe_rejects_absolute_outside(tmp_path):
    assert resolve_safe(tmp_path, "/etc/passwd") is None


def test_resolve_safe_handles_dotdot(tmp_path):
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "x.md").write_text("hi")
    # Notes/../Notes/x.md normalizes to Notes/x.md inside the vault
    assert resolve_safe(tmp_path, "Notes/../Notes/x.md") == (tmp_path / "Notes/x.md").resolve()


def test_is_system_path():
    assert is_system_path("_index.md")
    assert is_system_path("_channels/foo")
    assert is_system_path("Notes/_private.md")
    assert not is_system_path("Notes/x.md")
    assert not is_system_path("Notes")


def test_cap_output_under_limit():
    text = "hello"
    assert cap_output(text) == text


def test_cap_output_truncates_when_over():
    text = "x" * (OUTPUT_CAP_BYTES + 100)
    capped = cap_output(text)
    assert len(capped.encode("utf-8")) <= OUTPUT_CAP_BYTES + 200  # +footer
    assert "[output truncated:" in capped


from agent_core.tools._shell_helpers import (
    suggest_nearest_paths,
    format_not_found_with_suggestions,
)


def test_suggest_nearest_paths_finds_typo(tmp_path):
    """Underscore-vs-hyphen typo in directory name finds the real path."""
    (tmp_path / "Software-Development").mkdir()
    (tmp_path / "Software-Development" / "vibe-coding.md").write_text("body")
    matches = suggest_nearest_paths(tmp_path, "Software_Development/vibe-coding.md")
    assert "Software-Development/vibe-coding.md" in matches


def test_suggest_nearest_paths_respects_score_cutoff(tmp_path):
    """Totally unrelated query returns [] (nothing crosses 0.6 cutoff)."""
    (tmp_path / "foo.md").write_text("x")
    (tmp_path / "bar.md").write_text("x")
    assert suggest_nearest_paths(tmp_path, "totally-unrelated-xyz.md") == []


def test_suggest_nearest_paths_respects_max(tmp_path):
    """When many close matches exist, result is capped at max_suggestions."""
    for i in range(10):
        (tmp_path / f"vibe-coding-{i}.md").write_text("x")
    matches = suggest_nearest_paths(tmp_path, "vibe-coding.md", max_suggestions=3)
    assert len(matches) == 3


def test_suggest_nearest_paths_skips_system_paths(tmp_path):
    """Files under _archive (or any _-prefixed segment) are never suggested."""
    (tmp_path / "_archive").mkdir()
    (tmp_path / "_archive" / "foo.md").write_text("x")
    (tmp_path / "foo.md").write_text("x")
    matches = suggest_nearest_paths(tmp_path, "fooo.md")
    assert "foo.md" in matches
    assert all("_archive" not in m for m in matches)


def test_suggest_nearest_paths_skips_missing_path_itself(tmp_path):
    """Defensive: if the missing path happens to be in the candidate scan
    (race condition), it is not suggested as a match for itself."""
    (tmp_path / "foo.md").write_text("x")
    # Query for foo.md; even though it exists, it shouldn't suggest itself.
    matches = suggest_nearest_paths(tmp_path, "foo.md")
    assert "foo.md" not in matches


def test_suggest_nearest_paths_empty_vault(tmp_path):
    """No .md files in vault returns []."""
    assert suggest_nearest_paths(tmp_path, "anything.md") == []


def test_format_not_found_with_suggestions_appends_when_matches(tmp_path):
    """Formatter produces base + newline + 'Did you mean: ...' when matches exist."""
    (tmp_path / "foo.md").write_text("x")
    result = format_not_found_with_suggestions(
        tmp_path, "fooo.md", "File not found: fooo.md"
    )
    assert result.startswith("File not found: fooo.md")
    assert "\nDid you mean: " in result
    assert "foo.md" in result


def test_format_not_found_with_suggestions_unchanged_when_no_matches(tmp_path):
    """Formatter returns base verbatim when suggestions list is empty."""
    result = format_not_found_with_suggestions(
        tmp_path, "anything.md", "File not found: anything.md"
    )
    assert result == "File not found: anything.md"
