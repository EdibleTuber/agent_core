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
