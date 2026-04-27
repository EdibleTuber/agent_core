"""LearningManager — extracted lessons from conversations.

Learnings live in _learning/ within the vault. Each is a short markdown
file with a title, body (the lesson), source, and status. Users can
review learnings and promote valuable ones to wisdom via /promote.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from agent_core.utils.frontmatter import parse_frontmatter, serialize_frontmatter

logger = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


class LearningManager:
    def __init__(self, vault_path: Path, agent_name: str) -> None:
        self.vault_path = vault_path
        self.agent_name = agent_name

    @property
    def learning_dir(self) -> Path:
        return self.vault_path / "_learning" / self.agent_name

    def list(self) -> list[dict]:
        """List all learnings, returning dicts with 'slug', 'title', 'status'."""
        if not self.learning_dir.exists():
            return []
        entries = []
        for md_file in sorted(self.learning_dir.glob("*.md")):
            if md_file.stem == "ratings":
                continue
            meta, _ = parse_frontmatter(md_file.read_text())
            entries.append({
                "slug": md_file.stem,
                "title": meta.get("title", md_file.stem),
                "status": meta.get("status", "active"),
            })
        return entries

    def add(self, title: str, body: str, source: str) -> str:
        """Add a new learning. Returns the slug."""
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(title)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = {
            "title": title,
            "source": source,
            "created": now,
            "status": "active",
        }
        content = serialize_frontmatter(meta, body if body.endswith("\n") else body + "\n")
        (self.learning_dir / f"{slug}.md").write_text(content)
        logger.info("Added learning: %s", slug)
        return slug

    def get(self, slug: str) -> str:
        """Return the body of a learning by slug."""
        path = self.learning_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Learning not found: {slug}")
        _, body = parse_frontmatter(path.read_text())
        return body.strip()

    def remove(self, slug: str) -> None:
        """Delete a learning."""
        path = self.learning_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Learning not found: {slug}")
        path.unlink()
        logger.info("Removed learning: %s", slug)

    def mark_promoted(self, slug: str) -> None:
        """Mark a learning as promoted (status → promoted)."""
        path = self.learning_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Learning not found: {slug}")
        meta, body = parse_frontmatter(path.read_text())
        meta["status"] = "promoted"
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(serialize_frontmatter(meta, body if body.endswith("\n") else body + "\n"))
        logger.info("Marked learning as promoted: %s", slug)

    def exists(self, slug: str) -> bool:
        """Return True if a learning with this slug exists."""
        return (self.learning_dir / f"{slug}.md").exists()

    def get_meta(self, slug: str) -> dict:
        """Return the frontmatter dict of a learning by slug."""
        path = self.learning_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Learning not found: {slug}")
        meta, _ = parse_frontmatter(path.read_text())
        return meta

    def add_rating(self, rating: str, comment: str = "") -> None:
        """Append a rating entry to the ratings log."""
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        ratings_path = self.learning_dir / "ratings.md"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry = f"- [{now}] **{rating}**"
        if comment:
            entry += f" — {comment}"
        entry += "\n"
        if not ratings_path.exists():
            ratings_path.write_text(
                "---\ntitle: Session Ratings\n---\n\n# Session Ratings\n\n"
            )
        with open(ratings_path, "a") as f:
            f.write(entry)
        logger.info("Added rating: %s", rating)
