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
