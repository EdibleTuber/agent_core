"""Git helpers for agent_core consumers.

Currently exposes a single factory: `make_commit_callback(vault_path)` returns
a callable suitable for `Scratchpad.commit_callback`. Other helpers may be
added as future agents need them.

Failure policy: the callback never raises. It logs a warning on subprocess
errors (missing git binary, non-repo vault, hook rejections, etc.) and
returns. Callers who need authoritative success/failure should use git
directly rather than this helper.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def make_commit_callback(vault_path: Path) -> Callable[[Path, str], None]:
    """Return a callable that stages `path` and commits it with `message`.

    Behavior:
    - Stages the path with `git add -- <path>`. The `--` sentinel guards
      against paths that look like git options.
    - Skips the commit when the staged tree matches HEAD (detected via
      `git diff --cached --quiet`, which is locale-independent).
    - Disables GPG signing on the commit invocation. Scratchpad writes
      should not trigger interactive passphrase prompts.
    - Logs and swallows any subprocess failure so a transient git error
      doesn't break the surrounding operation.
    """

    def _commit(path: Path, message: str) -> None:
        try:
            subprocess.run(
                ["git", "-C", str(vault_path), "add", "--", str(path)],
                check=True, capture_output=True,
            )
            # Locale-independent "anything staged?" check. Returns 0 when
            # nothing is staged (treat as benign no-op), 1 when there's a
            # diff to commit.
            diff = subprocess.run(
                ["git", "-C", str(vault_path), "diff", "--cached", "--quiet"],
                capture_output=True,
            )
            if diff.returncode == 0:
                return
            result = subprocess.run(
                [
                    "git", "-C", str(vault_path),
                    "-c", "commit.gpgsign=false",
                    "commit", "-m", message,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    "git commit failed in %s: rc=%d stdout=%s stderr=%s",
                    vault_path,
                    result.returncode,
                    result.stdout.strip()[:200],
                    result.stderr.strip()[:200],
                )
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("git commit failed in %s: %s", vault_path, exc)

    return _commit
