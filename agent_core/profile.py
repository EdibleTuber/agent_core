"""ProfileManager — read and write the user's profile.

The profile lives in _profile/<username>.md within the vault. It contains
world facts, biographical notes, and opinions that get injected into
PAL's system prompt on every chat.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from agent_core.utils.frontmatter import parse_frontmatter, serialize_frontmatter

logger = logging.getLogger(__name__)


def _sanitize_username(username: str) -> str:
    """Reduce username to a filesystem-safe identifier."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", username)
    safe = safe.strip("_")
    return safe or "user"


class ProfileManager:
    def __init__(self, vault_path: Path, agent_name: str, username: str) -> None:
        self.vault_path = vault_path
        self.agent_name = agent_name
        self.username = _sanitize_username(username)

    @property
    def profile_path(self) -> Path:
        return self.vault_path / "_profile" / self.agent_name / f"{self.username}.md"

    def read(self) -> str:
        """Return the profile body, or empty string if not yet written."""
        if not self.profile_path.exists():
            return ""
        _, body = parse_frontmatter(self.profile_path.read_text())
        return body.strip()

    def write(self, body: str) -> None:
        """Overwrite the profile with the given body."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = {"title": "User Profile", "updated": now}
        content = serialize_frontmatter(meta, body if body.endswith("\n") else body + "\n")
        self.profile_path.write_text(content)
        logger.info("Wrote profile for %s", self.username)
