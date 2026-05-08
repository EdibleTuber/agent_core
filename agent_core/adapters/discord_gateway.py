"""Discord gateway helpers for agent_core consumers.

Lifted from PAL's discord_adapter (Phase G). Provides the generic
machinery: per-user DaemonConnection management, `!cmd` vs chat parsing,
slash-prefix rewriting (`/cmd` -> `!cmd`), 2000-char message splitting,
and a generic tool-progress formatter that takes per-tool labels via
the `custom_formatters` argument.

Bot classes (subclasses of discord.Client) stay in consumer code -- this
module is composition primitives, not a bot base class. Each agent
that wants Discord brings its own discord.Client subclass and uses
these helpers as composition pieces.

Pure-Python helpers -- no `import discord` at module level. Consumers
that write a discord.Client subclass install discord.py via the
`agent_core[discord]` extras_require; this module itself has no
dependency on it.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from agent_core.client import DaemonConnection


class UserConnectionManager:
    """Manages per-user DaemonConnection connections to the daemon."""

    def __init__(self, allowed_users: set[str], socket_path: str | Path) -> None:
        self.allowed_users = allowed_users
        self.socket_path = Path(socket_path)
        self._clients: dict[str, DaemonConnection] = {}

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowed_users

    async def get_client(self, user_id: str) -> DaemonConnection:
        """Get or create a DaemonConnection for a Discord user."""
        if user_id in self._clients:
            client = self._clients[user_id]
            if client.is_connected:
                return client
            del self._clients[user_id]

        client = DaemonConnection(self.socket_path)
        await client.connect()
        self._clients[user_id] = client
        return client

    async def close_all(self) -> None:
        """Close all daemon connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")

_DISCORD_MSG_LIMIT = 2000


def rewrite_slash_prefixes(text: str, names: set[str]) -> str:
    """Translate `/cmd` to `!cmd` for commands in `names`.

    Skips content inside fenced and inline code. Only rewrites tokens at
    line start or immediately following whitespace/punctuation.

    `names` is required -- the caller supplies the set of known command names.
    Pass an empty set to leave the text unchanged.
    """
    if not names:
        return text
    # Build an alternation regex for known command names, longest first
    # so `compile-batch` wins over `compile`.
    sorted_names = sorted(names, key=len, reverse=True)
    pattern = re.compile(
        r"(?P<lead>^|[\s,.;:!?\(])/(?P<name>"
        + "|".join(re.escape(n) for n in sorted_names)
        + r")\b"
    )

    # Protect fenced code blocks and inline code by temporarily substituting.
    placeholders: dict[str, str] = {}

    def _stash(m: re.Match) -> str:
        key = f"\x00PLACEHOLDER{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    safe = _FENCED_CODE.sub(_stash, text)
    safe = _INLINE_CODE.sub(_stash, safe)

    rewritten = pattern.sub(lambda m: f"{m.group('lead')}!{m.group('name')}", safe)

    for key, original in placeholders.items():
        rewritten = rewritten.replace(key, original)
    return rewritten


def parse_discord_message(text: str) -> tuple | None:
    """Parse a Discord message into an intent tuple.

    Returns:
        ("chat", text) for regular messages
        ("command", name, args) for ! commands
        None for empty/invalid messages
    """
    text = text.strip()
    if not text:
        return None
    if text.startswith("!"):
        rest = text[1:].strip()
        if not rest:
            return None
        parts = rest.split(None, 1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        return ("command", name, args)
    return ("chat", text)


def format_tool_progress(
    tool: str,
    arguments: dict,
    custom_formatters: dict[str, Callable[[dict], str]] | None = None,
) -> str:
    """Format a tool progress message for Discord (italic). Generic: any
    tool that isn't in custom_formatters gets `f"{tool}..."`."""
    if custom_formatters and tool in custom_formatters:
        label = custom_formatters[tool](arguments)
    else:
        label = f"{tool}..."
    return f"*[{label}]*"


def split_message(text: str, limit: int = _DISCORD_MSG_LIMIT) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit.

    Prefers splitting at paragraph boundaries (double newline).
    Falls back to splitting at the last space before the limit.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at > 0:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at + 2:]
            continue

        split_at = remaining.rfind(" ", 0, limit)
        if split_at > 0:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at + 1:]
            continue

        chunks.append(remaining[:limit])
        remaining = remaining[limit:]

    return chunks
