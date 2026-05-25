"""Tests for the worker contract Pydantic types."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_core.workers.types import (
    RiskTier,
    WorkerSpec,
    WorkerError,
    WorkerErrorCode,
    WORKER_CONTRACT_VERSION,
    AuditEntry,
)


def test_worker_contract_version_is_int():
    assert isinstance(WORKER_CONTRACT_VERSION, int)
    assert WORKER_CONTRACT_VERSION >= 1


def test_risk_tier_values():
    from typing import get_args
    assert {"low", "medium", "high", "critical"} == set(get_args(RiskTier))


def test_worker_spec_minimal_valid():
    spec = WorkerSpec(
        name="android",
        endpoint="http://localhost:9100/mcp",
        transport="streamable_http",
        risk_default="medium",
    )
    assert spec.name == "android"
    assert spec.risk_default == "medium"
    assert spec.capability_tags == []  # default


def test_worker_spec_rejects_invalid_tier():
    with pytest.raises(ValidationError):
        WorkerSpec(
            name="bad",
            endpoint="http://localhost:1/x",
            transport="streamable_http",
            risk_default="lethal",  # not a valid tier
        )


def test_worker_spec_rejects_invalid_transport():
    with pytest.raises(ValidationError):
        WorkerSpec(
            name="bad",
            endpoint="http://localhost:1/x",
            transport="carrier_pigeon",
            risk_default="low",
        )


def test_worker_error_codes_in_reserved_range():
    """Error codes -32000 to -32006 are reserved by the contract."""
    for code in WorkerErrorCode:
        assert -32099 <= code.value <= -32000


def test_worker_error_constructs():
    err = WorkerError(
        code=WorkerErrorCode.WORKER_INTERNAL,
        message="something broke",
        data={"hint": "retry"},
    )
    assert err.code == WorkerErrorCode.WORKER_INTERNAL
    assert err.data == {"hint": "retry"}


def test_audit_entry_minimal_valid():
    entry = AuditEntry(
        request_id="req-abc",
        worker="android",
        tool="attach",
        args={"package": "com.example"},
        declared_tier="low",
        effective_tier="low",
        outcome="ok",
        latency_ms=42,
        session_guid="11111111-1111-4111-9111-111111111111",
        worker_contract_version=1,
    )
    assert entry.recipe_id is None  # reserved, nullable
    assert entry.parent_call_id is None
    assert entry.override_reason is None
    assert isinstance(entry.ts, datetime)


def test_audit_entry_serializes_to_jsonlines_friendly_dict():
    entry = AuditEntry(
        request_id="r1",
        worker="w",
        tool="t",
        args={},
        declared_tier="medium",
        effective_tier="high",
        override_reason="name pattern *write* forces high",
        outcome="hitl_denied",
        latency_ms=10,
        session_guid="22222222-2222-4222-9222-222222222222",
        worker_contract_version=1,
    )
    d = entry.model_dump(mode="json")
    assert d["override_reason"] == "name pattern *write* forces high"
    assert d["effective_tier"] == "high"
    assert isinstance(d["ts"], str)  # ISO format


# Tests for stdio-transport WorkerSpec fields.


def test_worker_spec_stdio_minimal_valid():
    spec = WorkerSpec(
        name="frida",
        transport="stdio",
        risk_default="medium",
        command="frida-mcp",
    )
    assert spec.command == "frida-mcp"
    assert spec.args == []
    assert spec.env == {}
    assert spec.cwd is None
    # endpoint not required for stdio
    assert spec.endpoint is None


def test_worker_spec_stdio_with_args_env():
    spec = WorkerSpec(
        name="frida",
        transport="stdio",
        risk_default="medium",
        command="python",
        args=["-m", "frida_mcp"],
        env={"FRIDA_DEBUG": "1"},
    )
    assert spec.args == ["-m", "frida_mcp"]
    assert spec.env == {"FRIDA_DEBUG": "1"}


def test_worker_spec_stdio_requires_command():
    """transport=stdio without command must fail."""
    with pytest.raises(ValidationError, match="command"):
        WorkerSpec(
            name="bad",
            transport="stdio",
            risk_default="low",
        )


def test_worker_spec_http_requires_endpoint():
    """transport=streamable_http without endpoint must fail."""
    with pytest.raises(ValidationError, match="endpoint"):
        WorkerSpec(
            name="bad",
            transport="streamable_http",
            risk_default="low",
        )


def test_worker_spec_endpoint_field_optional_at_field_level():
    """endpoint becomes optional at the field level (None allowed) so stdio
    specs can omit it. The transport↔fields invariant is in the validator."""
    fields = WorkerSpec.model_fields
    assert fields["endpoint"].is_required() is False
