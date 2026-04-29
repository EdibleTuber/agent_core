"""Git helpers for agent_core consumers.

Currently exposes a single factory: `make_commit_callback(vault_path)` returns
a callable suitable for `Scratchpad.commit_callback`. Other helpers may be
added as future agents need them.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def make_commit_callback(vault_path: Path) -> Callable[[Path, str], None]:
    """Return a callable that stages `path` and commits it with `message`.

    No-ops silently if the working tree has no changes (e.g. the file content
    didn't actually change between writes). Logs and swallows any subprocess
    failure so a transient git error doesn't break the surrounding operation.
    """

    def _commit(path: Path, message: str) -> None:
        try:
            subprocess.run(
                ["git", "-C", str(vault_path), "add", str(path)],
                check=True, capture_output=True,
            )
            # Use --allow-empty=false (default); if there's nothing to commit
            # git returns nonzero, which we treat as a benign no-op.
            result = subprocess.run(
                ["git", "-C", str(vault_path), "commit", "-m", message],
                capture_output=True, text=True,
            )
            if result.returncode != 0 and "nothing to commit" not in result.stdout:
                logger.warning(
                    "git commit failed in %s: rc=%d stderr=%s",
                    vault_path, result.returncode, result.stderr.strip(),
                )
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("git commit failed in %s: %s", vault_path, exc)

    return _commit
