"""Pydantic types for the agent_core worker contract.

These types define the shape of workers.yaml entries, audit log records,
error responses, and contract-version negotiation. They are
transport-agnostic — the same models apply whether the worker is reached
over MCP-Streamable-HTTP, an HTTP /jobs API, or an in-process stub.
"""
from __future__ import annotations

import os

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_core.workers.artifacts import _HOST_RE
from urllib.parse import urlsplit


WORKER_CONTRACT_VERSION = 1
"""Contract major version. Workers and agents exchange this at initialize-time.

Same major: interoperate (optional new fields ignored older-side).
Different major: connection refused with -32005 protocol mismatch.
"""


RiskTier = Literal["low", "medium", "high", "critical"]
"""Per-tool risk classification.

- low: auto-execute, audit log only.
- medium: auto-execute with audit log + structured event.
- high: HITL approval required.
- critical: HITL approval + non-empty justification required.
"""


Transport = Literal["streamable_http", "http_job_api", "stdio"]
"""Worker transport. streamable_http is the MCP 2025-03-26 standard;
http_job_api is for legacy workers like apk-re-agents that ship their
own /jobs HTTP contract; stdio is for future co-located workers."""


class WorkerErrorCode(IntEnum):
    """Reserved error codes returned by workers in MCP error payloads."""
    WORKER_INTERNAL = -32000
    UPSTREAM_UNREACHABLE = -32001
    SESSION_EXPIRED = -32002
    HITL_DENIED = -32003
    RESOURCE_LIMIT = -32004
    PROTOCOL_VERSION_MISMATCH = -32005
    CONTRACT_VIOLATION = -32006


class WorkerError(BaseModel):
    """Structured error returned by a worker tool call."""
    code: WorkerErrorCode
    message: str
    data: dict[str, Any] | None = None


class WorkerSpec(BaseModel):
    """A single worker entry from workers.yaml."""

    model_config = ConfigDict(extra="forbid")
    """An unknown key is an ERROR, not a silent drop.

    pydantic's default extra="ignore" meant an older agent_core reading a
    newer workers.yaml quietly discarded keys it did not understand. That was
    a documented annoyance for `autoload`. It is not acceptable for
    `artifact_root`, which is a security control: a dropped root would
    silently remove the anchor that descriptor containment is to be checked
    against, with no error anywhere.
    """

    name: str
    endpoint: str | None = None
    transport: Transport
    risk_default: RiskTier
    container: str | None = None
    capability_tags: list[str] = Field(default_factory=list)
    kind: Literal["internal", "external_mcp"] = "internal"
    """external_mcp workers don't ship contract metadata; risk_default is
    raised one tier and name-pattern overrides apply aggressively."""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    connect_timeout: float | None = None
    """Wall-clock bound on this worker's connect, in seconds. None uses the
    pool default. Per-spec because a worker across a tailnet legitimately
    needs longer than one on the same box, and because a sleeping host does
    not refuse a connection -- it drops the SYN, so the bound is the only
    thing that ends the wait."""

    read_timeout: float | None = None
    """Wall-clock bound on every REQUEST this worker answers -- initialize,
    tools/list and each tools/call -- in seconds. None leaves the SDK's
    default, which is no session-level bound at all and a 300s transport read.
    Separate from connect_timeout because httpx times the dial and the
    response read independently; one field cannot bound both."""

    autoload: bool = True
    """Connect this worker at daemon startup. False means declared-but-not-
    loaded: it appears in the catalog and can be loaded at runtime.

    NOTE: an OLDER agent_core (pre model_config extra="forbid") reading a
    workers.yaml that sets autoload: false silently dropped the field and
    autoloaded the worker anyway. Consumers must bump their agent_core pin
    before adding the key.
    """

    artifact_root: str | None = None
    """Absolute directory on the WORKER's machine under which that worker may
    write artifacts. Operator-declared here, in workers.yaml, because the trust
    anchor has to be the file the worker cannot touch -- the same reasoning as
    the risk pins.

    None means the worker may not produce artifacts at all.

    DECLARED, ENFORCED AT ALL THREE LAYERS. The absolute-path validator
    below runs at config load, and agent_core.workers.artifacts.
    validate_descriptor now takes this object (as `spec`) and refuses a
    descriptor whose path is not under `{artifact_root}/{slug}` --
    lexically, D7's daemon-side shadow of the worker's real check. A
    worker with no root refuses artifact dispatch pre-gate in
    RiskAwareToolPool.call_tool (validation_failed audit row; no approval
    prompt; no dispatch).
    """

    artifact_drive_id: str | None = None
    """Expected contents of `{artifact_root}/.bench-store-id`.

    os.path.ismount() cannot tell one project's removable drive from another's,
    so writing a dump to the wrong stick would otherwise be silent.

    ENFORCED ON THE DAEMON SIDE.
    agent_core.workers.artifacts.validate_descriptor COMPARES a
    descriptor's drive_id against this value and refuses a mismatch -- and
    refuses outright when this field is None. The dispatch path carries
    this value: RiskAwareToolPool.call_tool injects the declared drive id
    into the tool arguments, refuses the dispatch pre-gate when the field
    is unset, and the descriptor's drive_id is compared against it. Note
    where the OTHER half has to live when it is built -- the daemon cannot
    see the worker's filesystem, so whatever reads `.bench-store-id` runs
    on the WORKER, next to artifact_path in pare-worker-kit.
    """

    artifact_host: str | None = None
    """Operator-declared reachable address for this worker's artifact host.

    The host is the destination name used in ``scp <host>:<path>`` when
    retrieving artifacts written by this worker.  It is validated at config
    load against the same grammar that the daemon uses for descriptor
    validation, so a malformed value is caught early.

    **Defaulting.**  When ``artifact_root`` is set, the transport is
    ``streamable_http`` or ``http_job_api``, and ``artifact_host`` is
    unset, it defaults to the
    hostname parsed from ``endpoint`` (via ``urlsplit(endpoint).hostname``).
    The operator does not need to repeat it.

    **stdio constraint.**  When the transport is ``stdio`` and
    ``artifact_root`` is set (meaning the worker may produce artifacts), the
    operator *must* declare ``artifact_host`` explicitly -- it cannot be
    defaulted, because stdio workers have no endpoint to derive one from.

    **Inert without a root.**  When ``artifact_root`` is ``None`` the field
    is inert: a declared value is accepted and stored, but no completeness
    error is raised.  This reflects the fact that the host is only meaningful
    in the context of artifact dispatch.

    **IPv6 / dash-leading refusal.**  The host grammar (``_HOST_RE``)
    rejects IPv6 literals (no colons or brackets) and any host beginning
    with a dash (argument injection range).  This is a fail-closed check
    at load time.
    """

    @field_validator("artifact_root")
    @classmethod
    def artifact_root_is_usable(cls, v: str | None) -> str | None:
        """Apply the same LEXICAL rule the worker applies, at config load.

        Containment is enforced on the worker (D7) and cannot be enforced here:
        the artifact is on another machine, so resolving this path daemon-side
        resolves it against the wrong namespace. This validator is not
        enforcement -- it is the operator finding out now rather than at
        hardware-run time with a dump half written.

        The three checks mirror pare_worker_kit.artifacts, whose reasoning is
        recorded there in full. A guard test runs one table of roots through
        both implementations and asserts identical verdicts, so this cannot
        drift into being a second, subtly different rule.
        """
        if v is None:
            return None
        if not v.startswith("/"):
            raise ValueError(
                f"artifact_root must be an absolute path, got {v!r}")
        if ".." in v.split("/"):
            # Refused even though the root is trusted: containment is checked
            # lexically and the kernel's resolution is not. For `/a/b/../c`
            # where `b` is a symlink, normpath says `/a/c` and the kernel says
            # somewhere else, so containment would be checked against a
            # directory that is not the one written to.
            raise ValueError(
                f"artifact_root must be normalised, got {v!r}: a '..' component "
                f"does not name the directory the operator declared")
        base = os.path.normpath(v)
        if os.path.commonpath([base, base]) != base:
            # A root that is not a commonpath prefix of ITSELF cannot have
            # containment checked against it. Today only paths beginning with
            # exactly two slashes have that shape: POSIX leaves them
            # implementation-defined, normpath preserves the `//` and
            # commonpath collapses it. Refused rather than collapsed, because
            # on a platform where `//host` names another filesystem,
            # collapsing would silently relocate the artifact root.
            raise ValueError(
                f"ambiguous artifact_root {v!r}: a path beginning with exactly "
                f"two slashes is implementation-defined in POSIX and is not a "
                f"prefix of itself, so containment cannot be checked against it")
        return v

    @field_validator("name")
    @classmethod
    def name_is_valid_identifier(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"worker name {v!r} must be alphanumeric/underscore (MCP-safe)"
            )
        return v

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "WorkerSpec":
        if self.transport in ("streamable_http", "http_job_api"):
            if not self.endpoint:
                raise ValueError(
                    f"worker {self.name!r}: transport {self.transport!r} requires endpoint"
                )
        elif self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"worker {self.name!r}: transport 'stdio' requires command"
                )
        return self

    @model_validator(mode="after")
    def validate_artifact_declaration(self) -> "WorkerSpec":
        """Enforce the artifact-host / artifact-drive-id declaration rules.

        R9 (artifact_host):

        1. **Endpoint transport defaulting.**  When ``artifact_root`` is
           set, the transport is ``streamable_http`` or ``http_job_api``,
           and ``artifact_host`` is unset, default it to
           ``urlsplit(endpoint).hostname``.  If the
           endpoint has no hostname (empty or absent), raise a
           ``ValidationError`` -- the operator must declare it explicitly.

        2. **stdio constraint.**  When the transport is ``stdio`` and
           ``artifact_root`` is set, ``artifact_host`` *must* be declared.
           There is no endpoint to default from, so the operator must name
           the host explicitly.  A missing host raises a
           ``ValidationError``.

        3. **Host grammar validation.**  Any declared (or defaulted)
           ``artifact_host`` is validated against ``_HOST_RE`` at load
           time.  IPv6 literals, dash-leading names, and other malformed
           values are refused (fail-closed).

        4. **Inert without a root.**  When ``artifact_root`` is ``None``
           and ``artifact_host`` is valid, no completeness error is raised.
           The field is accepted but inert -- the host only matters when
           artifact dispatch is active.

        R10 / A5 (artifact_drive_id):

        5. **Drive-id required with root.**  When ``artifact_root`` is set,
           ``artifact_drive_id`` must also be set (non-``None``).  A missing
           drive-id raises a ``ValidationError`` naming the missing field.
        """
        # --- R9d: endpoint transport defaulting -------------------------
        if self.artifact_root is not None:
            if self.transport in ("streamable_http", "http_job_api"):
                if self.artifact_host is None:
                    parsed = urlsplit(self.endpoint or "")
                    hostname = parsed.hostname or ""
                    if not hostname:
                        raise ValueError(
                            f"worker {self.name!r}: endpoint {self.endpoint!r} "
                            f"has no hostname; declare artifact_host explicitly"
                        )
                    self.artifact_host = hostname

        # --- R9c: stdio + root requires explicit host -------------------
        if self.transport == "stdio" and self.artifact_root is not None:
            if self.artifact_host is None:
                raise ValueError(
                    f"worker {self.name!r}: transport 'stdio' with "
                    f"artifact_root requires artifact_host to be declared"
                )

        # --- R9e: host grammar validation (fail-closed) -----------------
        if self.artifact_host is not None:
            if not _HOST_RE.match(self.artifact_host):
                raise ValueError(
                    f"worker {self.name!r}: artifact_host "
                    f"{self.artifact_host!r} does not match the required "
                    f"hostname grammar"
                )

        # --- R10 / A5: drive-id required with root --------------------
        if self.artifact_root is not None and self.artifact_drive_id is None:
            raise ValueError(
                f"worker {self.name!r}: artifact_root is set but "
                f"artifact_drive_id is not; both must be declared together"
            )

        return self


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


Outcome = Literal[
    "ok",
    "error",
    "hitl_approved",
    "hitl_denied",
    "validation_failed",
    "timeout",
    "cancelled",
    "approval_undeliverable",
    "worker_loaded",
    "worker_unloaded",
]


class AuditEntry(BaseModel):
    """One row in PARE's per-project audit log."""
    ts: datetime = Field(default_factory=_utc_now)
    request_id: str
    """The MCP request ID (also propagated to worker logs via _meta for
    cross-stream correlation)."""
    worker: str
    tool: str | None = None
    """None for control-plane rows (worker_loaded / worker_unloaded)."""
    args: dict[str, Any]
    """PARE-controlled redaction is applied before storing here."""
    declared_tier: RiskTier
    effective_tier: RiskTier
    override_reason: str | None = None
    detail: str | None = None
    tier_source: str | None = None
    """Provenance of declared_tier — one of:

      "wire"                the worker's advertised per-tool tier (or the
                            session high-water mark derived from it) escalated
                            above the worker-wide floor
      "floor"              the worker's risk_default floor (including the
                            session floor ratchet, and every external_mcp
                            worker, which is floor-only by contract)
      "invalid_advertised"  the worker sent a malformed tier or a malformed
                            meta container; the floor was used and the
                            contract violation is flagged here
      "unknown_worker"      no spec registered for this worker -> "high"
      None                  control-plane rows, and pre-v1.6 entries

    Forensic honesty: lets an auditor tell a low-tier dispatch advertised-low
    apart from a floor default, and either apart from a tampering signal."""
    outcome: Outcome
    latency_ms: int
    session_guid: str
    """The daemon-session boundary GUID, stamped per entry so audit
    trails group cleanly by session (§4.10.1)."""
    worker_contract_version: int

    # Reserved for v1.x recipes; nullable in v1.
    recipe_id: str | None = None
    parent_call_id: str | None = None
