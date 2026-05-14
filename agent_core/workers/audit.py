"""AuditLog — append-only JSONL with daily rotation.

Per-project; PARE creates one AuditLog per active project, writing to
~/.local/share/pare/projects/{project}/audit/audit-YYYY-MM-DD.jsonl.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent_core.workers.types import AuditEntry


def _today_utc() -> str:
    """Return UTC date in YYYY-MM-DD format. Monkeypatched in tests."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AuditLog:
    """Append-only JSONL audit log writer with date-based rotation."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def append(self, entry: AuditEntry) -> None:
        """Append one entry to today's log file."""
        path = self._directory / f"audit-{_today_utc()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json())
            f.write("\n")
