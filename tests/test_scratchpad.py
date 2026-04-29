"""Tests for Scratchpad: per-channel free-form markdown file in the vault."""
import pytest
from pathlib import Path
from agent_core.scratchpad import Scratchpad, ScratchpadTooLarge


@pytest.fixture
def commit_recorder():
    """Returns (callback, calls_list) for recording commit_callback invocations."""
    calls: list[tuple[Path, str]] = []

    def cb(path: Path, message: str) -> None:
        calls.append((path, message))

    return cb, calls


def test_read_returns_empty_when_file_missing(tmp_path, commit_recorder):
    cb, _ = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    assert sp.read() == ""


def test_write_creates_directory_and_file(tmp_path, commit_recorder):
    cb, _ = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    sp.write("# hello\n")

    expected_path = tmp_path / "_channels" / "testagent" / "C1" / "scratch.md"
    assert expected_path.exists()
    assert expected_path.read_text() == "# hello\n"


def test_write_invokes_commit_callback(tmp_path, commit_recorder):
    cb, calls = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    sp.write("# hello\n")
    expected_path = tmp_path / "_channels" / "testagent" / "C1" / "scratch.md"
    assert calls == [(expected_path, "scratch: update C1")]


def test_read_after_write_round_trip(tmp_path, commit_recorder):
    cb, _ = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    sp.write("content")
    assert sp.read() == "content"


def test_write_raises_when_over_cap(tmp_path, commit_recorder):
    cb, calls = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=10,
        commit_callback=cb,
    )
    with pytest.raises(ScratchpadTooLarge) as exc_info:
        sp.write("x" * 11)
    assert "11" in str(exc_info.value)
    assert "10" in str(exc_info.value)
    assert not (tmp_path / "_channels" / "testagent" / "C1" / "scratch.md").exists()
    assert calls == []


def test_append_adds_to_existing_content(tmp_path, commit_recorder):
    cb, _ = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    sp.write("line1\n")
    sp.append("line2\n")
    assert sp.read() == "line1\nline2\n"


def test_append_respects_cap(tmp_path, commit_recorder):
    cb, _ = commit_recorder
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=10,
        commit_callback=cb,
    )
    sp.write("short")
    with pytest.raises(ScratchpadTooLarge):
        sp.append(" and more bytes than allowed")
    assert sp.read() == "short"


def test_read_unreadable_file_returns_empty(tmp_path, commit_recorder, monkeypatch, caplog):
    import logging
    cb, _ = commit_recorder
    scratch_path = tmp_path / "_channels" / "testagent" / "C1" / "scratch.md"
    scratch_path.parent.mkdir(parents=True)
    scratch_path.write_text("hi")

    real_open = Path.open
    def patched_open(self, *args, **kwargs):
        if self == scratch_path and "r" in (args[0] if args else kwargs.get("mode", "r")):
            raise OSError("simulated")
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", patched_open)

    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    with caplog.at_level(logging.WARNING):
        assert sp.read() == ""
    assert any("unreadable" in rec.message.lower() for rec in caplog.records)


def test_write_with_no_commit_callback_does_not_raise(tmp_path):
    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=None,
    )
    sp.write("hello")
    assert (tmp_path / "_channels" / "testagent" / "C1" / "scratch.md").read_text() == "hello"


def test_commit_callback_receives_path_and_message(tmp_path):
    captured: list[tuple[Path, str]] = []

    def cb(path: Path, message: str) -> None:
        captured.append((path, message))

    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    sp.write("hello")
    expected_path = tmp_path / "_channels" / "testagent" / "C1" / "scratch.md"
    assert captured == [(expected_path, "scratch: update C1")]


def test_commit_callback_failure_is_logged_not_raised(tmp_path, caplog):
    import logging

    def cb(path: Path, message: str) -> None:
        raise RuntimeError("boom")

    sp = Scratchpad(
        vault_path=tmp_path,
        agent_name="testagent",
        channel_id="C1",
        max_bytes=1024,
        commit_callback=cb,
    )
    with caplog.at_level(logging.WARNING):
        sp.write("hello")
    expected_path = tmp_path / "_channels" / "testagent" / "C1" / "scratch.md"
    assert expected_path.read_text() == "hello"
    assert any("commit callback failed" in rec.message.lower() for rec in caplog.records)
