"""Tests for ProfileManager — user profile read/write."""
from pathlib import Path

import pytest

from agent_core.profile import ProfileManager


@pytest.fixture()
def vault(tmp_path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture()
def profile(vault) -> ProfileManager:
    return ProfileManager(vault, "test-agent", username="edible")


def test_profile_starts_empty(profile, vault):
    body = profile.read()
    assert body == ""


def test_profile_write_creates_file(profile, vault):
    profile.write("## World\n\nI run an inference server.\n")
    path = vault / "_profile" / "test-agent" / "edible.md"
    assert path.exists()
    content = path.read_text()
    assert "I run an inference server." in content
    assert "title: User Profile" in content


def test_profile_read_after_write(profile, vault):
    profile.write("## Bio\n\nSoftware engineer.\n")
    body = profile.read()
    assert "Software engineer." in body


def test_profile_write_updates_timestamp(profile, vault):
    profile.write("## World\n\nFirst version.\n")
    first = (vault / "_profile" / "test-agent" / "edible.md").read_text()
    profile.write("## World\n\nSecond version.\n")
    second = (vault / "_profile" / "test-agent" / "edible.md").read_text()
    assert first != second
    assert "Second version." in second
    assert "First version." not in second


def test_profile_directory_created_automatically(profile, vault):
    assert not (vault / "_profile").exists()
    profile.write("## Bio\n\nHi.\n")
    assert (vault / "_profile").is_dir()


def test_username_sanitized(vault):
    """Malicious usernames get sanitized to prevent path traversal."""
    bad = ProfileManager(vault, "test-agent", username="../../etc/passwd")
    # Should not escape vault
    assert ".." not in bad.username
    assert "/" not in bad.username
    bad.write("## Bio\n\nTest.\n")
    # File should be inside vault's _profile/<agent_name> dir
    assert bad.profile_path.parent == vault / "_profile" / "test-agent"


def test_empty_username_becomes_user(vault):
    p = ProfileManager(vault, "test-agent", username="")
    assert p.username == "user"
