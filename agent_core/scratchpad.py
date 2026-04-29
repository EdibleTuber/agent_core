"""Per-channel scratchpad: a free-form markdown file in the vault.

Lives at <vault>/_channels/<agent_name>/<channel_id>/scratch.md. Optionally
calls a commit callback after every write for git tracking. Size-capped to
prevent drift into a second wiki.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ScratchpadTooLarge(Exception):
    """Raised when a write would exceed the scratchpad size cap."""

    def __init__(self, current_bytes: int, proposed_bytes: int, max_bytes: int) -> None:
        super().__init__(
            f"scratchpad would be {proposed_bytes} bytes (cap {max_bytes}, "
            f"current {current_bytes})"
        )
        self.current_bytes = current_bytes
        self.proposed_bytes = proposed_bytes
        self.max_bytes = max_bytes


class Scratchpad:
    """File-backed free-form markdown owned by one channel."""

    def __init__(
        self,
        vault_path: Path,
        agent_name: str,
        channel_id: str,
        max_bytes: int,
        commit_callback: Callable[[Path, str], None] | None = None,
    ) -> None:
        self._vault_path = vault_path
        self._agent_name = agent_name
        self._channel_id = channel_id
        self._max_bytes = max_bytes
        self._commit_callback = commit_callback

    @property
    def _path(self) -> Path:
        return (
            self._vault_path
            / "_channels"
            / self._agent_name
            / self._channel_id
            / "scratch.md"
        )

    def read(self) -> str:
        """Return the scratchpad content, or empty string if missing/unreadable."""
        path = self._path
        if not path.exists():
            return ""
        try:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            logger.warning(
                "scratchpad %s unreadable (%s) treating as empty",
                path, exc,
            )
            return ""

    def write(self, content: str) -> None:
        """Replace scratchpad content. Raises ScratchpadTooLarge if over cap."""
        size = len(content.encode("utf-8"))
        if size > self._max_bytes:
            raise ScratchpadTooLarge(
                current_bytes=len(self.read().encode("utf-8")),
                proposed_bytes=size,
                max_bytes=self._max_bytes,
            )
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if self._commit_callback is not None:
            try:
                self._commit_callback(path, f"scratch: update {self._channel_id}")
            except Exception as exc:
                logger.warning(
                    "scratchpad commit callback failed for %s: %s",
                    self._channel_id, exc,
                )

    def append(self, text: str) -> None:
        """Append text. Raises ScratchpadTooLarge if resulting size over cap."""
        combined = self.read() + text
        self.write(combined)
