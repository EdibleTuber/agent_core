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


def _spec(**kw):
    base = dict(name="hardware", transport="stdio", command="/bin/true",
                risk_default="high")
    base.update(kw)
    return WorkerSpec(**base)


def test_artifact_root_defaults_to_none():
    assert _spec().artifact_root is None
    assert _spec().artifact_drive_id is None


def test_artifact_root_must_be_absolute():
    """Containment is checked against this root; a relative path cannot be
    compared against one."""
    with pytest.raises(ValidationError, match="absolute"):
        _spec(artifact_root="bench-store")


def test_a_typo_in_workers_yaml_is_an_error_not_a_silent_drop():
    """pydantic's default extra='ignore' meant an older agent_core silently
    dropped a key it did not know. For autoload that was a documented
    annoyance; for artifact_root it would mean a security control absent with
    no error anywhere."""
    with pytest.raises(ValidationError):
        _spec(artifact_roots="/mnt/bench-store")


def test_the_real_workers_yaml_still_loads():
    """extra='forbid' is a behaviour change for every consumer. This is the
    canary: PARE's live catalog must still parse."""
    from pathlib import Path

    from agent_core.workers.registry import WorkerRegistry

    live = Path("/mnt/secondary/projects/PARE/workers.yaml")
    if not live.is_file():
        pytest.skip("PARE checkout not present next to agent_core")
    reg = WorkerRegistry.load(live)
    assert reg.all(), "the live catalog parsed to nothing"


# ---------------------------------------------------------------------------
# Task 1: WorkerSpec.artifact_host  (D1 – D11)
# ---------------------------------------------------------------------------
from urllib.parse import urlsplit

_D5_DRIVE = "12345678-90ab-4cd0-8e12-34567890abcd"
"""A valid UUID for artifact_drive_id used by D5 and D6."""


def test_d1_endpoint_defaults_artifact_host_from_url_hostname():
    """D1: endpoint transport (streamable_http), artifact_root set, no
    artifact_host ⇒ loads, and spec.artifact_host == urlsplit(endpoint).
    hostname (fixture endpoint http://100.97.133.126:9101/mcp ⇒ "100.97.
    133.126")."""
    spec = WorkerSpec(
        name="bench",
        endpoint="http://100.97.133.126:9101/mcp",
        transport="streamable_http",
        risk_default="medium",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=_D5_DRIVE,
    )
    assert spec.artifact_host == "100.97.133.126"


def test_d2_stdio_with_root_requires_artifact_host():
    """D2: transport=stdio, root set, no host ⇒ ValidationError naming
    artifact_host (and that stdio cannot default it)."""
    with pytest.raises(ValidationError, match="artifact_host"):
        WorkerSpec(
            name="bench",
            transport="stdio",
            risk_default="medium",
            command="frida-mcp",
            artifact_root="/mnt/bench-store",
            artifact_drive_id=_D5_DRIVE,
        )


def test_d3_stdio_with_root_and_explicit_host_loads():
    """D3: stdio + root + artifact_host="pare-bench" ⇒ loads,
    spec.artifact_host == "pare-bench"."""
    spec = WorkerSpec(
        name="bench",
        transport="stdio",
        risk_default="medium",
        command="frida-mcp",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=_D5_DRIVE,
        artifact_host="pare-bench",
    )
    assert spec.artifact_host == "pare-bench"


def test_d4_endpoint_explicit_host_wins_over_default():
    """D4: endpoint + root + explicit host ⇒ loads; explicit wins over the
    default."""
    spec = WorkerSpec(
        name="bench",
        endpoint="http://100.97.133.126:9101/mcp",
        transport="streamable_http",
        risk_default="medium",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=_D5_DRIVE,
        artifact_host="my-bench-host",
    )
    assert spec.artifact_host == "my-bench-host"


def test_d5_ipv6_literal_host_refused():
    """D5: artifact_host="::1" (IPv6 literal) ⇒ ValidationError
    (_HOST_RE admits no colons/brackets)."""
    with pytest.raises(ValidationError, match="artifact_host"):
        WorkerSpec(
            name="bench",
            endpoint="http://100.97.133.126:9101/mcp",
            transport="streamable_http",
            risk_default="medium",
            artifact_root="/mnt/bench-store",
            artifact_drive_id=_D5_DRIVE,
            artifact_host="::1",
        )


def test_d6_dash_leading_host_refused():
    """D6: artifact_host="-evil" (dash-leading — an ssh/scp argument, not a
    destination) ⇒ ValidationError."""
    with pytest.raises(ValidationError, match="artifact_host"):
        WorkerSpec(
            name="bench",
            endpoint="http://100.97.133.126:9101/mcp",
            transport="streamable_http",
            risk_default="medium",
            artifact_root="/mnt/bench-store",
            artifact_drive_id=_D5_DRIVE,
            artifact_host="-evil",
        )


def test_d7_endpoint_no_hostname_no_host_error():
    """D7: root set, endpoint with no hostname (e.g. http://:9101/mcp), no
    host ⇒ ValidationError — cannot be defaulted; declare explicitly."""
    with pytest.raises(ValidationError, match="artifact_host"):
        WorkerSpec(
            name="bench",
            endpoint="http://:9101/mcp",
            transport="streamable_http",
            risk_default="medium",
            artifact_root="/mnt/bench-store",
            artifact_drive_id=_D5_DRIVE,
        )


def test_d8_root_without_artifact_drive_id_fails():
    """D8 (A5): root set, artifact_drive_id None ⇒ ValidationError naming
    artifact_drive_id."""
    with pytest.raises(ValidationError, match="artifact_drive_id"):
        WorkerSpec(
            name="bench",
            endpoint="http://100.97.133.126:9101/mcp",
            transport="streamable_http",
            risk_default="medium",
            artifact_root="/mnt/bench-store",
            artifact_drive_id=None,
        )


def test_d9_root_with_valid_drive_id_loads():
    """D9 (P-pin): root set + a valid artifact_drive_id ⇒ loads."""
    spec = WorkerSpec(
        name="bench",
        endpoint="http://100.97.133.126:9101/mcp",
        transport="streamable_http",
        risk_default="medium",
        artifact_root="/mnt/bench-store",
        artifact_drive_id=_D5_DRIVE,
    )
    assert spec.artifact_root == "/mnt/bench-store"
    assert spec.artifact_drive_id == _D5_DRIVE


def test_d10_no_root_no_host_no_drive_id_loads():
    """D10 (P-pin): no root, no host, no drive id ⇒ loads, and
    getattr(spec, "artifact_host", None) is None."""
    spec = WorkerSpec(
        name="bench",
        transport="stdio",
        risk_default="medium",
        command="frida-mcp",
    )
    assert getattr(spec, "artifact_host", None) is None


def test_d11_no_root_declared_host_loads_inert():
    """D11: no root + declared artifact_host="pare-bench" ⇒ loads, inert
    (R9d)."""
    spec = WorkerSpec(
        name="bench",
        transport="stdio",
        risk_default="medium",
        command="frida-mcp",
        artifact_host="pare-bench",
    )
    assert spec.artifact_host == "pare-bench"
    assert spec.artifact_root is None
