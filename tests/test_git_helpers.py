"""Tests for agent_core.git_helpers.make_commit_callback."""
import subprocess
from pathlib import Path

import pytest

from agent_core.git_helpers import make_commit_callback


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_commit_callback_creates_commit(tmp_path):
    _init_git_repo(tmp_path)
    cb = make_commit_callback(tmp_path)
    file_path = tmp_path / "note.md"
    file_path.write_text("hello\n")
    cb(file_path, "scratch: add note")
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert log == "scratch: add note"


def test_commit_callback_handles_subsequent_writes(tmp_path):
    _init_git_repo(tmp_path)
    cb = make_commit_callback(tmp_path)
    file_path = tmp_path / "note.md"
    file_path.write_text("v1\n")
    cb(file_path, "v1")
    file_path.write_text("v2\n")
    cb(file_path, "v2")
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert log == ["v2", "v1"]


def test_commit_callback_no_op_when_no_changes(tmp_path):
    """Callback should not raise if there's nothing to commit."""
    _init_git_repo(tmp_path)
    file_path = tmp_path / "note.md"
    file_path.write_text("hello\n")
    cb = make_commit_callback(tmp_path)
    cb(file_path, "first")
    # Second call without changes should not raise
    cb(file_path, "second-noop")
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert log == ["first"]  # second commit skipped


def test_commit_callback_swallows_non_repo_error(tmp_path, caplog):
    """A non-repo vault_path produces a warning but does not raise."""
    cb = make_commit_callback(tmp_path)  # no _init_git_repo
    file_path = tmp_path / "note.md"
    file_path.write_text("hi")
    with caplog.at_level("WARNING", logger="agent_core.git_helpers"):
        cb(file_path, "should not raise")
    assert any("git commit failed" in r.message for r in caplog.records)


def test_commit_callback_ignores_global_gpgsign(tmp_path, monkeypatch):
    """If the user has commit.gpgsign=true globally, the helper still commits
    cleanly without trying to invoke a GPG agent. We force the global config
    via GIT_CONFIG_COUNT/KEY/VALUE so the test doesn't depend on the host's
    git config."""
    _init_git_repo(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "commit.gpgsign")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    cb = make_commit_callback(tmp_path)
    file_path = tmp_path / "note.md"
    file_path.write_text("hello\n")
    cb(file_path, "scratch: gpg-bypassed commit")
    log = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert log == "scratch: gpg-bypassed commit"


def test_commit_callback_handles_dash_prefixed_path(tmp_path):
    """Paths whose names start with '-' must not be interpreted as git options."""
    _init_git_repo(tmp_path)
    weird = tmp_path / "--weird-name.md"
    weird.write_text("hello\n")
    cb = make_commit_callback(tmp_path)
    cb(weird, "scratch: dash-prefixed name")
    log = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert log == "scratch: dash-prefixed name"
