"""Pydantic types for the agent_core worker contract.

These types define the shape of workers.yaml entries, audit log records,
error responses, and contract-version negotiation. They are
transport-agnostic — the same models apply whether the worker is reached
over MCP-Streamable-HTTP, an HTTP /jobs API, or an in-process stub.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    `artifact_root`, which is a security control: a dropped root would leave
    descriptor containment silently unenforced with no error anywhere.
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

    None means the worker may not produce artifacts at all: a
    produces="artifact" dispatch against a worker with no root is REFUSED
    rather than accepted unvalidated.
    """

    artifact_drive_id: str | None = None
    """Expected contents of `{artifact_root}/.bench-store-id`.

    os.path.ismount() cannot tell one project's removable drive from another's,
    so writing a dump to the wrong stick would otherwise be silent.
    """

    @field_validator("artifact_root")
    @classmethod
    def artifact_root_is_absolute(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("/"):
            raise ValueError(
                f"artifact_root must be an absolute path, got {v!r}")
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
