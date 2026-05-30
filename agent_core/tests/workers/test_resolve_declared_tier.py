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


def test_missing_tier_internal_is_failsafe_high():
    # internal worker, no advertised tier -> max(floor, "high")
    out = resolve_declared_tier(_spec("low"), None)
    assert out == ("high", "fallback_safe")


def test_invalid_tier_internal_is_failsafe_high():
    out = resolve_declared_tier(_spec("medium"), "lowww")
    assert out == ("high", "fallback_safe")


def test_external_mcp_uses_risk_default_unchanged():
    out = resolve_declared_tier(_spec("medium", kind="external_mcp"), "low")
    assert out == ("medium", "floor")


def test_unknown_worker_spec_none_is_high():
    out = resolve_declared_tier(None, "low")
    assert out == ("high", "unknown_worker")
