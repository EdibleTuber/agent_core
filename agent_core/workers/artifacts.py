"""The daemon's half of the artifact contract.

A tool that declares `produces: artifact` returns a DESCRIPTOR of a file it
wrote on its own machine, not the file's contents. This module states the wire
constant and validates what comes back.
"""
from __future__ import annotations

import re

PRODUCES_META_KEY = "agent_core/produces"
"""Stated here and in pare-worker-kit, with a guard test on each side. See
RISK_TIER_META_KEY for the same arrangement and the same reasoning: the two
packages are installed separately on machines that never share a Python
environment."""

PRODUCES_RESULT = "result"
PRODUCES_ARTIFACT = "artifact"

VALID_PRODUCES = (PRODUCES_RESULT, PRODUCES_ARTIFACT)

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_HOST_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,252}\Z")
"""A hostname or IP that will be interpolated into `scp <host>:<path>`.

Leading-alphanumeric for the same reason the slug rule requires it: a host
beginning with `-` is an ssh/scp ARGUMENT, not a destination, and
`-oProxyCommand=...` is remote code execution on the operator's own machine.
No colon, so it cannot forge the host:path split; no whitespace or shell
metacharacters. Max 253 characters to comply with DNS hostname limits.
"""

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
"""NUL, newline, tab, the escape byte and DEL. Refused in `path` for the
reasons in validate_descriptor's docstring -- none of them are fixable by a
caller that quotes correctly."""

_REQUIRED = ("host", "path", "size", "sha256")


class DescriptorError(ValueError):
    """A tool declared `produces: artifact` and returned something else."""


def validate_descriptor(payload, *, worker: str, tool: str) -> dict:
    """Check the SHAPE of an artifact descriptor. Not its truthfulness.

    Everything here is self-reported by the worker. This rejects a malformed
    descriptor, a confused one, and a buggy one. It does not make a hostile
    worker honest -- nothing on this side of the wire can.

    WHAT sha256 BUYS, EXACTLY. It is INTEGRITY, NOT AUTHENTICITY. It detects
    corruption ACROSS THE TRANSFER, so a corrupted or half-finished `scp` is
    caught rather than silently acted on, and it gives the artifact a stable
    identity for audit, dedup and later reference -- both real and worth
    having. What it cannot do is attest that the producer chose the RIGHT
    bytes: the worker supplies both the file and the digest, so a worker that
    truncates a dump AT PRODUCTION returns a perfectly correct sha256 OF THE
    TRUNCATED BYTES, and the operator verifies it successfully --
    indistinguishable, by hash alone, from a complete one. (The two cases are
    kept in different words on purpose: the transfer one is a half-finished
    `scp`, which the digest catches; the production one is a truncated dump,
    which it cannot.) Authenticity
    needs a digest the producer did not supply: a vendor's published hash,
    one recorded before the worker could have been compromised, or a
    signature over a key the worker does not hold. The same paragraph is in
    pare_worker_kit.artifacts.artifact_path, which is what the worker-side
    implementer reads; keep the two saying the same thing.

    WHY `path` IS CHECKED, AND WHY ONLY THIS MUCH. Both halves of the
    descriptor land in the operator's shell -- retrieval has the shape
    `scp <host>:<path>`. `host` has had a full grammar since the previous fix
    round; `path` had only isinstance and startswith('/'). The checks added
    here are exactly the ones a CORRECT CALLER CANNOT FIX BY QUOTING:

      - A `..` component. `scp host:'/mnt/store/../../etc/shadow'` retrieves
        /etc/shadow however it is quoted, because `..` is resolved by the
        kernel and never by the shell. Checked component-wise, never as a
        substring: `fw..bin` and `..hidden` are legitimate filenames and pass.
      - A basename beginning with `-`. This is argument injection, the one
        class quoting does not touch: `scp host:/mnt/s/-rf .` lands a local
        file named `-rf`, and the next `rm *` or `tar cf x *` in that
        directory hands `-rf` to the command as an OPTION. The check uses the
        last non-empty component, so a trailing slash does not hide the name.
      - NUL, newline and other control characters. NUL truncates the path in
        every C-level path API, so the bytes checked are not the bytes
        opened; a newline breaks any line-oriented audit record or `while
        read` loop; an escape sequence rewrites what the operator SEES while
        confirming the path. Quotes do not make any of these visible or safe.

    SHELL METACHARACTERS (`;`, `$`, backtick, `|`, `&`) ARE DELIBERATELY
    ACCEPTED. Refusing them would look like shell-safety and could not
    deliver it: space, quote, backslash, `*`, `?`, `[` and `~` are ordinary
    in filenames and each is enough to break an unquoted splice, so the
    blocklist would leave the hole open while advertising it shut -- which is
    worse than not claiming it, because it invites the unquoted splice it
    cannot protect. They also occur in real filenames (`$` in a Samba share
    path, `&` and `;` in names taken from vendor strings), so refusing them
    is a usability cost paid for no gain.

    WHAT THE CALL SITE HAS TO DO INSTEAD, STATED PRECISELY. Use a transfer
    whose argument is never re-parsed by a REMOTE shell: `scp` in its default
    SFTP mode (that is, WITHOUT `-O`), or `sftp`, or `rsync -s`. Building an
    argv list and calling shlex.quote are NOT sufficient on their own -- they
    protect the operator's LOCAL shell only. Under the legacy SCP protocol,
    scp(1)'s own CAVEATS section says the remote user's shell is executed to
    perform glob(3) matching, so `scp -O bench-b:'/mnt/s/$(id)' .` survives
    the operator's quoting intact and the substitution then runs on bench-b
    -- a host the descriptor NAMED and the compromised worker need not
    control. `ssh host "cat <path>"` and rsync without `-s` have the same
    shape. OpenSSH >= 9.0 defaults to SFTP, so the default path is safe; this
    is why the requirement is "use an SFTP-mode transfer", not "quote it".

    So this function refuses what NO caller can fix, and leaves what the
    right transfer mode fixes completely.

    NOT CHECKED HERE, AND IT IS A REAL GAP: containment of `path` under the
    worker's operator-declared `artifact_root`. This signature is handed a
    worker NAME, not its WorkerSpec, so there is no root to compare against
    and containment is not expressible here -- `/etc/shadow` passes every
    check in this function. THE CALLER THAT HOLDS THE ROOT MUST DO IT: the
    dispatch path that routes on the produces declaration resolves the
    WorkerSpec, and it must refuse a descriptor whose path is not under
    `spec.artifact_root`, and refuse any descriptor at all from a worker
    whose `artifact_root` is None. Nothing dispatches artifacts yet; that
    check has to land with the wiring, not be assumed to exist already.

    THE SECOND GAP, SAME FAMILY: `host` is checked for SHAPE and never
    against the worker it came from. `_HOST_RE` accepts any well-formed
    hostname, so a compromised worker A can return `host: "bench-b"` and aim
    the operator's retrieval at a machine of its choosing -- which is what
    makes the remote-shell caveat above reachable at all. As with
    containment, the daemon CAN check this and this function cannot: the
    endpoint lives on the WorkerSpec, and this signature has only the worker
    name. The dispatch path must reconcile a descriptor's `host` with the
    spec it dispatched to.
    """
    where = f"{worker}.{tool}"
    if not isinstance(payload, dict):
        raise DescriptorError(
            f"{where} declared produces=artifact but returned "
            f"{type(payload).__name__}, not a JSON object")
    for field in _REQUIRED:
        if field not in payload:
            raise DescriptorError(f"{where}: descriptor is missing {field!r}")

    size = payload["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise DescriptorError(
            f"{where}: descriptor size must be a non-negative int, got {size!r}")

    sha = payload["sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise DescriptorError(
            f"{where}: descriptor sha256 must be 64 lowercase hex chars, "
            f"got {sha!r}")

    path = payload["path"]
    if not isinstance(path, str) or not path.startswith("/"):
        raise DescriptorError(
            f"{where}: descriptor path must be absolute, got {path!r}")
    if _CONTROL_RE.search(path):
        raise DescriptorError(
            f"{where}: descriptor path contains a control character, "
            f"got {path!r}")
    parts = path.split("/")
    if ".." in parts:
        raise DescriptorError(
            f"{where}: descriptor path contains a '..' component, "
            f"got {path!r}")
    # The last NON-EMPTY component: `/mnt/s/-rf/` names `-rf` just as
    # `/mnt/s/-rf` does, and os.path.basename would return "" for the first.
    name = next((c for c in reversed(parts) if c), "")
    if name.startswith("-"):
        raise DescriptorError(
            f"{where}: descriptor path names {name!r}, which begins with '-' "
            f"and is read as an option rather than a filename by the commands "
            f"an operator runs on it; got {path!r}")

    host = payload["host"]
    if not isinstance(host, str) or not _HOST_RE.match(host):
        raise DescriptorError(
            f"{where}: descriptor host must be a hostname or address, "
            f"got {host!r}")

    return dict(payload)


SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
"""ArcticBase's own rule, verbatim (its storage layer enforces the anchored
form). Adopted rather than invented because the slug names three things -- a
workbench, a capture store and a directory on the bench drive -- and the
strictest consumer has to win. A project accepted here but rejected there
would silently have no approval or report surface at all.

Note `fullmatch` below, not `match`: an unanchored check accepts
`proj/../../etc`, which is the single most common way this is written wrong.
The leading-alphanumeric requirement is what keeps a slug out of
argument-injection range in the scp/tar commands an operator later runs by
hand -- `-rf` and `--checkpoint-action=exec=sh` are legal directory names.
"""


def validate_slug(slug) -> str:
    """The project slug supplied to a worker. Raises ValueError if unusable."""
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"invalid project slug {slug!r}: must match {SLUG_RE.pattern!r} "
            f"(lowercase, starts alphanumeric, max 64 chars)")
    return slug
