"""Pydantic types for the agent_core worker contract.

These types define the shape of workers.yaml entries, audit log records,
error responses, and contract-version negotiation. They are
transport-agnostic — the same models apply whether the worker is reached
over MCP-Streamable-HTTP, an HTTP /jobs API, or an in-process stub.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    endpoint: str
    transport: Transport
    risk_default: RiskTier
    container: str | None = None
    capability_tags: list[str] = Field(default_factory=list)
    kind: Literal["internal", "external_mcp"] = "internal"
    """external_mcp workers don't ship contract metadata; risk_default is
    raised one tier and name-pattern overrides apply aggressively."""

    @field_validator("name")
    @classmethod
    def name_is_valid_identifier(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"worker name {v!r} must be alphanumeric/underscore (MCP-safe)"
            )
        return v
