"""Tests for RiskGate — declared tier + override-up only."""
import pytest

from agent_core.workers.risk import RiskGate, TierDecision


def test_no_overrides_returns_declared_tier():
    gate = RiskGate(overrides=[])
    decision = gate.evaluate(
        worker="android",
        tool="attach",
        declared_tier="low",
    )
    assert decision.effective_tier == "low"
    assert decision.override_reason is None


def test_override_can_raise_tier():
    """A pattern matching *write* raises 'low' to 'high'."""
    gate = RiskGate(overrides=[
        ("*write*", "high"),
    ])
    decision = gate.evaluate(
        worker="android",
        tool="write_memory",
        declared_tier="low",
    )
    assert decision.effective_tier == "high"
    assert decision.override_reason is not None
    assert "write" in decision.override_reason


def test_override_cannot_lower_tier():
    """An override of 'low' on a declared 'high' stays high (override-up only)."""
    gate = RiskGate(overrides=[
        ("attach*", "low"),
    ])
    decision = gate.evaluate(
        worker="android",
        tool="attach",
        declared_tier="high",
    )
    assert decision.effective_tier == "high"  # not lowered
    assert decision.override_reason is None  # no upgrade applied


def test_multiple_overrides_take_highest():
    gate = RiskGate(overrides=[
        ("*memory*", "medium"),
        ("*dump_*", "high"),
    ])
    decision = gate.evaluate(
        worker="android",
        tool="dump_memory",
        declared_tier="low",
    )
    assert decision.effective_tier == "high"


def test_match_is_against_worker_tool_combined():
    """Override patterns can scope by worker.tool combination."""
    gate = RiskGate(overrides=[
        ("ios_keychain_*", "high"),
    ])
    decision = gate.evaluate(
        worker="ios",
        tool="keychain_dump",
        declared_tier="medium",
    )
    assert decision.effective_tier == "high"


def test_invalid_override_tier_raises():
    """Overrides at construction time must use valid tiers."""
    with pytest.raises(ValueError):
        RiskGate(overrides=[("*write*", "ultracritical")])
