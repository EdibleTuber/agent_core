"""Tests for LearningManager — learning extraction storage."""
from pathlib import Path

import pytest

from agent_core.learning import LearningManager


@pytest.fixture()
def vault(tmp_path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture()
def learning(vault) -> LearningManager:
    return LearningManager(vault, "test-agent")


def test_list_empty(learning):
    assert learning.list() == []


def test_add_creates_file(learning, vault):
    slug = learning.add(
        title="Always test edge cases",
        body="Edge cases reveal assumptions. Test boundaries, empty inputs, and error paths.",
        source="conversation",
    )
    assert slug == "always-test-edge-cases"
    path = vault / "_learning" / "test-agent" / "always-test-edge-cases.md"
    assert path.exists()
    content = path.read_text()
    assert "title: Always test edge cases" in content
    assert "Edge cases reveal assumptions" in content
    assert "source: conversation" in content


def test_list_returns_entries(learning):
    learning.add(title="First", body="Body one.", source="conversation")
    learning.add(title="Second", body="Body two.", source="conversation")
    entries = learning.list()
    assert len(entries) == 2
    slugs = [e["slug"] for e in entries]
    assert "first" in slugs
    assert "second" in slugs


def test_get_returns_body(learning):
    learning.add(title="Rule", body="Always check.", source="conversation")
    body = learning.get("rule")
    assert body == "Always check."


def test_get_nonexistent_raises(learning):
    with pytest.raises(FileNotFoundError):
        learning.get("nonexistent")


def test_remove_deletes_file(learning, vault):
    learning.add(title="Temp", body="Will be removed.", source="conversation")
    assert (vault / "_learning" / "test-agent" / "temp.md").exists()
    learning.remove("temp")
    assert not (vault / "_learning" / "test-agent" / "temp.md").exists()


def test_remove_nonexistent_raises(learning):
    with pytest.raises(FileNotFoundError):
        learning.remove("nope")


def test_add_sanitizes_slug(learning, vault):
    slug = learning.add(title="Hello, World!", body="Test.", source="conversation")
    assert slug == "hello-world"
    assert (vault / "_learning" / "test-agent" / "hello-world.md").exists()


def test_add_stores_metadata(learning, vault):
    import yaml
    learning.add(title="Test", body="Body.", source="conversation")
    content = (vault / "_learning" / "test-agent" / "test.md").read_text()
    meta = yaml.safe_load(content.split("---")[1])
    assert meta["title"] == "Test"
    assert meta["source"] == "conversation"
    assert "created" in meta
    assert meta["status"] == "active"


from agent_core.wisdom import WisdomManager


def test_mark_promoted_updates_status(learning, vault):
    learning.add(title="Good Idea", body="This works.", source="conversation")
    learning.mark_promoted("good-idea")
    import yaml
    content = (vault / "_learning" / "test-agent" / "good-idea.md").read_text()
    meta = yaml.safe_load(content.split("---")[1])
    assert meta["status"] == "promoted"
    assert "promoted_at" in meta


def test_mark_promoted_nonexistent_raises(learning):
    with pytest.raises(FileNotFoundError):
        learning.mark_promoted("nope")


def test_add_rating(learning, vault):
    learning.add_rating("good", "Great session")
    ratings_path = vault / "_learning" / "test-agent" / "ratings.md"
    assert ratings_path.exists()
    content = ratings_path.read_text()
    assert "**good**" in content
    assert "Great session" in content


def test_add_rating_appends(learning, vault):
    learning.add_rating("good", "First")
    learning.add_rating("bad", "Second")
    content = (vault / "_learning" / "test-agent" / "ratings.md").read_text()
    assert "**good**" in content
    assert "**bad**" in content
    assert "First" in content
    assert "Second" in content


def test_list_excludes_ratings_file(learning):
    learning.add(title="Real Learning", body="Body.", source="conversation")
    learning.add_rating("good")
    entries = learning.list()
    slugs = [e["slug"] for e in entries]
    assert "ratings" not in slugs
    assert "real-learning" in slugs


def test_exists_returns_true_for_existing(tmp_path):
    from agent_core.learning import LearningManager
    lm = LearningManager(tmp_path, "test-agent")
    slug = lm.add("My Lesson", "body text", source="conversation")
    assert lm.exists(slug) is True


def test_exists_returns_false_for_missing(tmp_path):
    from agent_core.learning import LearningManager
    lm = LearningManager(tmp_path, "test-agent")
    assert lm.exists("no-such-slug") is False


def test_get_meta_returns_frontmatter(tmp_path):
    from agent_core.learning import LearningManager
    lm = LearningManager(tmp_path, "test-agent")
    slug = lm.add("My Lesson", "body text", source="conversation")
    meta = lm.get_meta(slug)
    assert meta["title"] == "My Lesson"
    assert meta["status"] == "active"


def test_get_meta_raises_for_missing(tmp_path):
    from agent_core.learning import LearningManager
    import pytest
    lm = LearningManager(tmp_path, "test-agent")
    with pytest.raises(FileNotFoundError):
        lm.get_meta("no-such-slug")


def test_two_agents_have_isolated_dirs(tmp_path):
    """Two managers with the same vault_path but different agent_names see only their own entries."""
    pal = LearningManager(tmp_path, "pal")
    relab = LearningManager(tmp_path, "re-lab")

    pal.add("PAL lesson", "Avoid the fire emoji.", source="chat")
    relab.add("RE Lab lesson", "Always pin clang versions.", source="chat")

    pal_slugs = [e["slug"] for e in pal.list()]
    relab_slugs = [e["slug"] for e in relab.list()]

    assert pal_slugs == ["pal-lesson"]
    assert relab_slugs == ["re-lab-lesson"]
    assert (tmp_path / "_learning" / "pal" / "pal-lesson.md").exists()
    assert (tmp_path / "_learning" / "re-lab" / "re-lab-lesson.md").exists()
