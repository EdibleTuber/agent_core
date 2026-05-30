import pytest
from agent_core.workers.risk import resolve_declared_tier, RISK_TIER_META_KEY
from agent_core.workers.types import WorkerSpec


def _spec(risk_default="medium", kind="internal"):
    return WorkerSpec(
        name="frida", transport="stdio", command="x",
        risk_default=risk_default, kind=kind,
    )


def test_meta_key_is_namespaced():
    assert RISK_TIER_META_KEY == "agent_core/risk_tier"


def test_advertised_escalates_above_floor():
    # floor=medium, advertised=critical -> critical
    out = resolve_declared_tier(_spec("medium"), "critical")
    assert out == ("critical", "wire")


def test_floor_dominates_when_advertised_is_lower():
    # floor=high, advertised=low -> high (escalate-only relative to floor)
    out = resolve_declared_tier(_spec("high"), "low")
    assert out == ("high", "floor")


def test_missing_tier_falls_back_to_floor():
    # internal worker, no advertised tier -> use the risk_default floor (Option C:
    # no dispatch-time fail-safe; safety is via operator pins + conformance).
    out = resolve_declared_tier(_spec("low"), None)
    assert out == ("low", "floor")


def test_invalid_string_tier_falls_back_to_floor_flagged():
    # malformed string tier -> floor, but flagged as invalid_advertised for audit
    out = resolve_declared_tier(_spec("medium"), "lowww")
    assert out == ("medium", "invalid_advertised")


def test_unhashable_advertised_tier_does_not_crash():
    # a hostile worker can send a dict/list/bool as the tier; must not raise on
    # the `in _VALID_TIERS` membership test, must fall back to floor + flag it.
    for hostile in ({"tier": "low"}, ["low"], 123, True):
        out = resolve_declared_tier(_spec("low"), hostile)
        assert out == ("low", "invalid_advertised")


def test_external_mcp_uses_risk_default_unchanged():
    out = resolve_declared_tier(_spec("medium", kind="external_mcp"), "low")
    assert out == ("medium", "floor")


def test_unknown_worker_spec_none_is_high():
    out = resolve_declared_tier(None, "low")
    assert out == ("high", "unknown_worker")


from agent_core.workers.types import AuditEntry


def test_audit_entry_accepts_tier_source():
    e = AuditEntry(
        request_id="r", worker="w", tool="t", args={},
        declared_tier="high", effective_tier="high", outcome="ok",
        latency_ms=1, session_guid="s", worker_contract_version=1,
        tier_source="wire",
    )
    assert e.tier_source == "wire"


def test_audit_entry_tier_source_defaults_none():
    e = AuditEntry(
        request_id="r", worker="w", tool="t", args={},
        declared_tier="high", effective_tier="high", outcome="ok",
        latency_ms=1, session_guid="s", worker_contract_version=1,
    )
    assert e.tier_source is None
