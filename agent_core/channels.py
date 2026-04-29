"""Per-channel Conversation container with on-disk persistence.

Each channel (identified by a free-form string, e.g. Discord channel ID,
`cli-default` for CLI) gets its own Conversation instance, backed by a jsonl
file at <vault>/_channels/<agent_name>/<channel_id>/history.jsonl. On first
access for a channel, if the file exists, its contents are replayed into a
fresh Conversation. Subsequent accesses return the same cached instance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from agent_core.conversation import Conversation

logger = logging.getLogger(__name__)

_CHANNEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_channel_id(channel_id: str) -> bool:
    """Return True if the id matches the allowed character set and is non-empty."""
    return bool(_CHANNEL_ID_PATTERN.match(channel_id))


class ChannelStore:
    """Caches Conversation instances per channel, loading from disk as needed."""

    def __init__(
        self,
        vault_path: Path,
        agent_name: str,
        history_depth: int,
    ) -> None:
        self._channels_dir = vault_path / "_channels" / agent_name
        self._history_depth = history_depth
        self._cache: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, channel_id: str) -> Conversation:
        """Return the Conversation for channel_id, loading or creating as needed."""
        if not validate_channel_id(channel_id):
            raise ValueError(f"invalid channel_id: {channel_id!r}")
        async with self._lock:
            if channel_id in self._cache:
                return self._cache[channel_id]

            channel_dir = self._channels_dir / channel_id
            channel_dir.mkdir(parents=True, exist_ok=True)
            history_path = channel_dir / "history.jsonl"

            conv = Conversation(
                history_depth=self._history_depth,
                history_path=history_path,
            )

            if history_path.exists():
                self._replay_into(conv, history_path)

            self._cache[channel_id] = conv
            return conv

    def _replay_into(self, conv: Conversation, history_path: Path) -> None:
        """Replay existing messages into the Conversation. Safe on bad data."""
        try:
            with history_path.open("r", encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            logger.warning(
                "slot=%s history unreadable (%s) renaming and starting fresh",
                history_path, exc,
            )
            self._rename_corrupt(history_path)
            return

        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "channel %s history.jsonl line %d malformed, skipping",
                    history_path.parent.name, lineno,
                )
                continue
            conv._messages.append(message)
        conv._truncate()

    def _rename_corrupt(self, history_path: Path) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt = history_path.with_name(f"{history_path.name}.corrupt-{ts}")
        try:
            history_path.rename(corrupt)
        except OSError as exc:
            logger.warning("could not rename corrupt history %s: %s", history_path, exc)
