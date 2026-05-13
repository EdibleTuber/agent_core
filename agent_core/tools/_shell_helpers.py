"""Shared helpers for shell-style builtin tools.

Path resolution rooted at the agent's vault. System paths (any component
starting with `_`) are rejected. Output is capped at 32 KB with a truncation
footer.
"""
from __future__ import annotations

import difflib
from pathlib import Path

OUTPUT_CAP_BYTES = 32 * 1024


def resolve_safe(vault_path: Path, arg: str) -> Path | None:
    """Resolve `arg` against `vault_path`. Returns None if it escapes the vault.

    The returned Path is always fully resolved (symlinks collapsed, no `..`
    components), regardless of whether `vault_path` was passed resolved.
    Existence is NOT checked — callers must verify themselves.
    """
    try:
        full = (vault_path / arg).resolve()
    except (OSError, ValueError):
        return None
    try:
        full.relative_to(vault_path.resolve())
    except ValueError:
        return None
    return full


def is_system_path(path: str | Path) -> bool:
    """True if any path component starts with `_`. Accepts str or Path."""
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


def suggest_nearest_paths(
    vault_path: Path,
    missing_path: str,
    *,
    max_suggestions: int = 3,
    score_cutoff: float = 0.6,
) -> list[str]:
    """Return up to max_suggestions vault-relative paths similar to missing_path.

    Walks the vault for *.md files, scores each against the missing path
    via difflib.SequenceMatcher. Matches against the FULL vault-relative
    path (not just the stem) so directory-level typos like
    `Software_Development/` vs `Software-Development/` are caught.

    Skips paths whose any segment starts with `_` (matches is_system_path).
    Skips the missing path itself (defensive; could occur in a race).
    Returns [] when no candidate meets score_cutoff or the vault is empty.

    score_cutoff=0.6 is difflib's default; catches typos and word-order
    swaps but rejects unrelated names.
    """
    try:
        vault_resolved = vault_path.resolve()
    except (OSError, ValueError):
        return []

    candidates: list[str] = []
    for path in vault_resolved.rglob("*.md"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(vault_resolved)
        except ValueError:
            continue
        rel_str = str(rel)
        # Skip system paths (any segment starts with `_`).
        if any(part.startswith("_") for part in rel.parts):
            continue
        # Defensive: never suggest the path the caller said was missing.
        if rel_str == missing_path:
            continue
        candidates.append(rel_str)

    if not candidates:
        return []

    return difflib.get_close_matches(
        missing_path, candidates, n=max_suggestions, cutoff=score_cutoff
    )


def format_not_found_with_suggestions(
    vault_path: Path,
    missing_path: str,
    base_message: str,
) -> str:
    """Build the 404 error string, appending 'Did you mean: ...' when matches exist.

    Returns base_message verbatim when suggest_nearest_paths returns [].
    """
    suggestions = suggest_nearest_paths(vault_path, missing_path)
    if not suggestions:
        return base_message
    return f"{base_message}\nDid you mean: {', '.join(suggestions)}"
