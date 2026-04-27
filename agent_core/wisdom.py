"""WisdomManager — curated guidance entries injected into the system prompt.

Wisdom entries live in _wisdom/ within the vault. Each entry is a short
markdown file with a title and a body (the actual guidance). Entries are
small, focused, and human-editable.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from agent_core.utils.frontmatter import parse_frontmatter, serialize_frontmatter

logger = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    """Convert a title into a filesystem-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


class WisdomManager:
    def __init__(self, vault_path: Path, agent_name: str) -> None:
        self.vault_path = vault_path
        self.agent_name = agent_name

    @property
    def wisdom_dir(self) -> Path:
        return self.vault_path / "_wisdom" / self.agent_name

    def list(self) -> list[dict]:
        """List all wisdom entries, returning dicts with 'slug' and 'title'."""
        if not self.wisdom_dir.exists():
            return []
        entries = []
        for md_file in sorted(self.wisdom_dir.glob("*.md")):
            meta, _ = parse_frontmatter(md_file.read_text())
            entries.append({
                "slug": md_file.stem,
                "title": meta.get("title", md_file.stem),
            })
        return entries

    def add(self, title: str, body: str) -> str:
        """Add a new wisdom entry. Returns the slug."""
        self.wisdom_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(title)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = {"title": title, "created": now}
        content = serialize_frontmatter(meta, body if body.endswith("\n") else body + "\n")
        (self.wisdom_dir / f"{slug}.md").write_text(content)
        logger.info("Added wisdom: %s", slug)
        return slug

    def get(self, slug: str) -> str:
        """Return the body of a wisdom entry by slug."""
        path = self.wisdom_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Wisdom not found: {slug}")
        _, body = parse_frontmatter(path.read_text())
        return body.strip()

    def remove(self, slug: str) -> None:
        """Delete a wisdom entry."""
        path = self.wisdom_dir / f"{slug}.md"
        if not path.exists():
            raise FileNotFoundError(f"Wisdom not found: {slug}")
        path.unlink()
        logger.info("Removed wisdom: %s", slug)

    def bodies(self) -> list[str]:
        """Return all wisdom entry bodies, for injection into prompts."""
        if not self.wisdom_dir.exists():
            return []
        bodies = []
        for md_file in sorted(self.wisdom_dir.glob("*.md")):
            _, body = parse_frontmatter(md_file.read_text())
            bodies.append(body.strip())
        return bodies
