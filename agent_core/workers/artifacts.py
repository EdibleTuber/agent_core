"""The daemon's half of the artifact contract.

A tool that declares `produces: artifact` returns a DESCRIPTOR of a file it
wrote on its own machine, not the file's contents. This module states the wire
constant and validates what comes back.
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.workers.types import WorkerSpec

PRODUCES_META_KEY = "agent_core/produces"
"""Stated here and in pare-worker-kit, with a guard test on each side. See
RISK_TIER_META_KEY for the same arrangement and the same reasoning: the two
packages are installed separately on machines that never share a Python
environment."""

PRODUCES_RESULT = "result"
PRODUCES_ARTIFACT = "artifact"

VALID_PRODUCES = (PRODUCES_RESULT, PRODUCES_ARTIFACT)

ARTIFACT_DESCRIPTOR_FIELDS = ("host", "path", "size", "sha256", "hashed_at",
                              "media_type", "drive_id")
"""The seven fields a `produces: artifact` tool must return, in wire order.

Stated here AND in pare-worker-kit, with a guard test on each side, for the
same reason as PRODUCES_META_KEY above: the two packages are separately
installed and never share a Python environment. validate_descriptor requires
exactly these fields and pare_worker_kit's open_artifact builds exactly
these; a field present on one side and not the other is a descriptor that
validates on one machine and is refused on the other, with no useful error
anywhere. All seven are required; none is optional. The eighth field of the
object that gets published, produced_by, is added by the daemon AFTER
validation and never travels.
"""

RESERVED_SLUG_ARG = "project_slug"
"""The tool-argument name dispatch injects with the project's ArcticBase
slug. RESERVED_DRIVE_ID_ARG is the same arrangement for the drive id the
artifact must land on; see it below.

Reserved means the daemon supplies the value: injected at the dispatch
chokepoint, overwriting whatever the model supplied, so the model never sees
it as an input it may choose. A worker names its tool-handler parameter after
this value so the injection lands where the handler expects it -- which is
why the value must be a legal Python identifier as well as wire vocabulary.
Stated here AND in pare-worker-kit, with a guard test on each side. Changing
either value is a wire-breaking change.
"""

RESERVED_DRIVE_ID_ARG = "expected_drive_id"
"""The same arrangement as RESERVED_SLUG_ARG, for the drive id. A descriptor
whose drive_id differs from the injected value is refused.
"""

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

_HASHED_AT_RE = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z\Z")
"""RFC 3339 UTC in the strict `Z` form, optional fractional seconds.

"Required" without a format is not a contract, so the wire states one. The
check is FORMAT, not calendar: whether the timestamp is true is a custody
record the daemon cannot verify (the worker's clock is what it is), and this
function checks shape, not truthfulness, like every other field. A non-UTC
offset is refused because the field IS UTC, not "a time with a zone"; the
uppercase `Z` is pinned because the producer (open_artifact) emits that
spelling and a second accepted spelling is comparison surface the operator's
eye cannot audit.
"""

_MEDIA_TYPE_RE = re.compile(
    r"\A[A-Za-z0-9!#$&^_.+-]{1,126}/[A-Za-z0-9!#$&^_.+-]{1,126}\Z")
"""An IANA type/subtype, and nothing else.

No parameters (`;q=`, `;charset=`): the field is the type/subtype, not a full
media-type production. No wildcards (`*`): a descriptor names what the file
IS, not a range of what it might be. The character class is the RFC 2045
token set minus the characters no registered type uses, and the 126 cap is
the RFC token length limit. Case-insensitive by RFC 2046, so
`application/octet-stream` and `APPLICATION/OCTET-STREAM` both pass.
"""

_DRIVE_ID_RE = re.compile(
    r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
"""Canonical lowercase UUID -- the grammar of the sentinel file
(`{artifact_root}/.bench-store-id`) the value is read from.

Lowercase and dashed, and only that: the sentinel is written that way, and a
second accepted spelling is a second comparison the operator's eye cannot
audit. drive_id also gets the explicit _CONTROL_RE check: the value is
printed into the mismatch message an operator reads, and an escape sequence
in it would rewrite what that message says.
"""


class DescriptorError(ValueError):
    """A tool declared `produces: artifact` and returned something else."""


def validate_descriptor(payload, *, spec: WorkerSpec, tool: str, slug: str) -> dict:
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

    CONTAINMENT IS CHECKED HERE, AGAINST THE PROJECT DIRECTORY, NOT THE ROOT.
    The slug arrives as a keyword because it is not in the payload: the
    daemon validated it, injected it into the tool call as the reserved slug
    argument, and hands it back here. The rule is `path` under
    `{artifact_root}/{slug}`, decided by `commonpath` on normalised paths --
    a LEXICAL check, D7's division of labour: the worker enforces the real
    containment (it is the only side that can see symlinks and mount points),
    and this is the daemon's shadow of the same rule, so a descriptor that
    lies about its project is refused at the chokepoint rather than at
    retrieval time. Against the root alone the check would bind the injected
    slug to nothing: `/mnt/bench-store/other-proj/fw.bin` is well contained
    by the root and is another project's dump.

    A worker whose `artifact_root` is None refuses every descriptor, as does
    one whose `artifact_drive_id` is None: the dispatch path refuses earlier,
    with the operator-facing message, and these refusals keep the function
    safe to call standalone.

    THE DRIVE ID IS COMPARED, NOT JUST FORM-CHECKED. `drive_id` must match
    the sentinel's UUID grammar AND equal `spec.artifact_drive_id`; a
    mismatch means the bytes went to a different drive than workers.yaml
    names, and the error names both values so the operator can see which is
    which.

    HOST IS SHAPE-CHECKED HERE, RECONCILED AT DISPATCH. `_HOST_RE` accepts
    any well-formed hostname, so this function alone cannot stop a
    compromised worker from returning `host: "bench-b"` and aiming the
    operator's retrieval at a machine of its choosing -- which is what
    makes the remote-shell caveat above reachable at all. The dispatch path
    closes the gap: RiskAwareToolPool.call_tool notes a host mismatch in
    the audit detail when the descriptor's host differs from
    `spec.artifact_host`, and the handoff descriptor carries the
    operator-declared value, never the worker's claim.
    """
    where = f"{spec.name}.{tool}"
    if not isinstance(payload, dict):
        raise DescriptorError(
            f"{where} declared produces=artifact but returned "
            f"{type(payload).__name__}, not a JSON object")
    for field in ARTIFACT_DESCRIPTOR_FIELDS:
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

    if spec.artifact_root is None:
        raise DescriptorError(
            f"{where}: worker {spec.name!r} declares no artifact_root, so no "
            f"artifact it returns can be contained; refused, not validated")
    project = os.path.normpath(os.path.join(spec.artifact_root, slug))
    if os.path.commonpath([os.path.normpath(path), project]) != project:
        raise DescriptorError(
            f"{where}: descriptor path {path!r} is not under {project!r}: "
            f"containment is against the project directory "
            f"({spec.artifact_root!r}/{slug!r}), not the root, so the "
            f"injected slug binds the path to the project it names")

    hashed_at = payload["hashed_at"]
    if not isinstance(hashed_at, str) or not _HASHED_AT_RE.match(hashed_at):
        raise DescriptorError(
            f"{where}: descriptor hashed_at must be RFC 3339 UTC "
            f"(e.g. 2026-09-06T12:34:56Z), got {hashed_at!r}")

    media_type = payload["media_type"]
    if not isinstance(media_type, str) or not _MEDIA_TYPE_RE.match(media_type):
        raise DescriptorError(
            f"{where}: descriptor media_type must be an IANA type/subtype, "
            f"got {media_type!r}")

    drive_id = payload["drive_id"]
    if (not isinstance(drive_id, str) or _CONTROL_RE.search(drive_id)
            or not _DRIVE_ID_RE.match(drive_id)):
        raise DescriptorError(
            f"{where}: descriptor drive_id must be the canonical lowercase "
            f"UUID read from the sentinel file, got {drive_id!r}")
    if spec.artifact_drive_id is None:
        raise DescriptorError(
            f"{where}: worker {spec.name!r} declares artifact_root without "
            f"artifact_drive_id; a descriptor it returns cannot be checked "
            f"against the drive it names, so it is refused")
    if drive_id != spec.artifact_drive_id:
        raise DescriptorError(
            f"{where}: descriptor drive_id {drive_id!r} is not the declared "
            f"{spec.artifact_drive_id!r}: the artifact was written to a "
            f"different drive than workers.yaml names")

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
