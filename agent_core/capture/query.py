from __future__ import annotations

_ALLOWED_FIELDS = frozenset({"worker", "tool", "session_id"})


def fts_phrase(text: str) -> str:
    """Wrap a user string as a single FTS5 phrase so punctuation (.-:) is
    literal, not query syntax. Double embedded quotes per FTS5 rules."""
    return '"' + text.replace('"', '""') + '"'
