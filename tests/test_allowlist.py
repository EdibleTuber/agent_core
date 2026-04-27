"""Tests for AllowlistManager — domain allowlist validation."""
from pathlib import Path

import pytest

from agent_core.allowlist import AllowlistManager


@pytest.fixture()
def vault(tmp_path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture()
def allowlist(vault) -> AllowlistManager:
    return AllowlistManager(vault, "test-agent")


def test_empty_allowlist_denies_all(allowlist):
    assert allowlist.is_allowed("https://example.com/page") is False


def test_seed_creates_file_with_starter_domains(allowlist, vault):
    allowlist.seed()
    path = vault / "_config" / "test-agent" / "allowlist.md"
    assert path.exists()
    content = path.read_text()
    assert "wikipedia.org" in content
    assert "arxiv.org" in content


def test_after_seed_allows_listed_domains(allowlist):
    allowlist.seed()
    assert allowlist.is_allowed("https://en.wikipedia.org/wiki/Python") is True
    assert allowlist.is_allowed("https://arxiv.org/abs/1706.03762") is True


def test_denies_unlisted_domains(allowlist):
    allowlist.seed()
    assert allowlist.is_allowed("https://evil.example.com/") is False


def test_wildcard_subdomain_match(allowlist, vault):
    (vault / "_config" / "test-agent").mkdir(parents=True)
    (vault / "_config" / "test-agent" / "allowlist.md").write_text(
        "# Allowlist\n\n- *.readthedocs.io\n"
    )
    assert allowlist.is_allowed("https://flask.readthedocs.io/en/stable/") is True
    assert allowlist.is_allowed("https://readthedocs.io/") is True
    assert allowlist.is_allowed("https://readthedocs.example.com/") is False


def test_exact_domain_match_no_subdomain(allowlist, vault):
    (vault / "_config" / "test-agent").mkdir(parents=True)
    (vault / "_config" / "test-agent" / "allowlist.md").write_text(
        "# Allowlist\n\n- github.com\n"
    )
    assert allowlist.is_allowed("https://github.com/user/repo") is True
    assert allowlist.is_allowed("https://raw.github.com/x") is False


def test_list_returns_all_patterns(allowlist):
    allowlist.seed()
    patterns = allowlist.list()
    assert "wikipedia.org" in patterns
    assert "arxiv.org" in patterns
    assert len(patterns) > 5


def test_rejects_non_http_schemes(allowlist):
    allowlist.seed()
    assert allowlist.is_allowed("ftp://wikipedia.org/file") is False
    assert allowlist.is_allowed("file:///etc/passwd") is False
    assert allowlist.is_allowed("javascript:alert(1)") is False


def test_rejects_malformed_urls(allowlist):
    allowlist.seed()
    assert allowlist.is_allowed("not a url") is False
    assert allowlist.is_allowed("") is False
