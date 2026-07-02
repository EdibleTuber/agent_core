from __future__ import annotations

from pathlib import Path


def resolve_capture_db(cwd: Path, marker: str | None, *, home: Path, xdg_state: Path,
                       channel_id: str) -> tuple[Path, bool]:
    """Resolve the capture db path. Walk up from cwd for `marker`, stopping
    before $HOME (a marker exactly at $HOME is ignored). Fall back to a
    per-launch path under xdg_state keyed by channel_id."""
    if marker:
        cwd = Path(cwd).resolve()
        home = Path(home).resolve()
        for d in [cwd, *cwd.parents]:
            if d == home or d == d.parent:  # $HOME ceiling / filesystem root
                break
            if (d / marker).is_dir():
                return d / marker / "capture.db", True
    fallback = Path(xdg_state) / "captures" / f"{channel_id}.db"
    return fallback, False
