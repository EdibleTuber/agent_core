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

from pydantic import BaseModel, Field, field_validator, model_validator


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
]


class AuditEntry(BaseModel):
    """One row in PARE's per-project audit log."""
    ts: datetime = Field(default_factory=_utc_now)
    request_id: str
    """The MCP request ID (also propagated to worker logs via _meta for
    cross-stream correlation)."""
    worker: str
    tool: str
    args: dict[str, Any]
    """PARE-controlled redaction is applied before storing here."""
    declared_tier: RiskTier
    effective_tier: RiskTier
    override_reason: str | None = None
    detail: str | None = None
    tier_source: str | None = None
    """Provenance of declared_tier: "wire" | "floor" | "fallback_safe" |
    "unknown_worker" | None (pre-v1.6 entries). Forensic honesty: lets an
    auditor tell a low-tier dispatch advertised-low apart from a floor default."""
    outcome: Outcome
    latency_ms: int
    session_guid: str
    """The daemon-session boundary GUID, stamped per entry so audit
    trails group cleanly by session (§4.10.1)."""
    worker_contract_version: int

    # Reserved for v1.x recipes; nullable in v1.
    recipe_id: str | None = None
    parent_call_id: str | None = None
