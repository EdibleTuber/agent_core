from __future__ import annotations

_COL_MAP = {"worker": "c.worker", "tool": "c.tool", "session_id": "c.session_id"}
_ALLOWED_FIELDS = frozenset(_COL_MAP)


def fts_phrase(text: str) -> str:
    """Wrap a user string as a single FTS5 phrase so punctuation (.-:) is
    literal, not query syntax. Double embedded quotes per FTS5 rules."""
    return '"' + text.replace('"', '""') + '"'
