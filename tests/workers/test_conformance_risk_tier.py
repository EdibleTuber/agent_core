"""Unit tests for the wire risk_tier conformance assertion.

The live stdio/streamable_http conformance suites delegate the per-tool
"advertises a valid risk_tier in _meta" check to a shared helper; we test
that helper directly here (fast, no server spawn). The live fixture tests
(test_conformance_stdio / _streamable_http) prove real servers WITH _meta
pass end-to-end.
"""
import pytest

from agent_core.workers.conformance import _assert_valid_risk_tier_meta
from agent_core.workers.risk import RISK_TIER_META_KEY


class _Tool:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta


def test_tool_with_valid_tier_passes():
    _assert_valid_risk_tier_meta(_Tool("t", {RISK_TIER_META_KEY: "low"}))  # no raise


def test_tool_with_no_meta_fails():
    with pytest.raises(AssertionError, match="risk_tier"):
        _assert_valid_risk_tier_meta(_Tool("t", None))


def test_tool_with_meta_but_no_tier_key_fails():
    with pytest.raises(AssertionError, match="risk_tier"):
        _assert_valid_risk_tier_meta(_Tool("t", {"something_else": 1}))


def test_tool_with_invalid_tier_value_fails():
    with pytest.raises(AssertionError, match="risk_tier"):
        _assert_valid_risk_tier_meta(_Tool("t", {RISK_TIER_META_KEY: "LOW"}))


def test_the_meta_key_agrees_with_the_worker_kits():
    """The other half of the pare-worker-kit cross-check.

    RISK_TIER_META_KEY is a WIRE constant: the daemon reads it out of _meta
    that a worker on another machine wrote. Both packages state the literal
    because they are installed separately -- often on different hardware --
    and this test is what keeps the two statements the same. If it fails, a
    worker's advertised tiers stop being read and every tool silently
    resolves at its floor: safe, but silent, which is the bad kind.
    """
    import pytest
    kit_risk = pytest.importorskip(
        "pare_worker_kit.risk",
        reason="pare-worker-kit is not installed here; the worker-side half "
               "of this check runs in that package's own suite")
    assert kit_risk.RISK_TIER_META_KEY == RISK_TIER_META_KEY
