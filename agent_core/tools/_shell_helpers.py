"""Shared helpers for shell-style builtin tools.

Path resolution rooted at the agent's vault. System paths (any component
starting with `_`) are rejected. Output is capped at 32 KB with a truncation
footer.
"""
from __future__ import annotations

from pathlib import Path

OUTPUT_CAP_BYTES = 32 * 1024


def resolve_safe(vault_path: Path, arg: str) -> Path | None:
    """Resolve `arg` against `vault_path`. Returns None if it escapes."""
    try:
        full = (vault_path / arg).resolve()
    except (OSError, ValueError):
        return None
    try:
        full.relative_to(vault_path.resolve())
    except ValueError:
        return None
    return full


def is_system_path(path: str) -> bool:
    """True if any path component starts with `_`."""
    return any(part.startswith("_") for part in Path(path).parts)


def cap_output(text: str) -> str:
    """Cap text at OUTPUT_CAP_BYTES, append a truncation footer if cut."""
    encoded = text.encode("utf-8")
    if len(encoded) <= OUTPUT_CAP_BYTES:
        return text
    truncated_bytes = encoded[:OUTPUT_CAP_BYTES]
    # Don't split a multi-byte character at the boundary
    truncated = truncated_bytes.decode("utf-8", errors="ignore")
    dropped = len(encoded) - len(truncated.encode("utf-8"))
    return truncated + f"\n\n[output truncated: {dropped} bytes dropped]"
