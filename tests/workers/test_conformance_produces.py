"""Build-time enforcement, matching how the risk tier is handled.

_assert_valid_risk_tier_meta is described in conformance.py as "the
compensating control that lets dispatch fall back to the risk_default floor
without a runtime fail-safe". `produces` has the same shape: dispatch falls
back to "result", so something has to reject a typo before it ships.
"""
import pytest

from agent_core.workers.artifacts import PRODUCES_META_KEY
from agent_core.workers.conformance import _assert_valid_produces_meta


class _Tool:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta


def test_absent_is_valid_because_result_is_the_default():
    _assert_valid_produces_meta(_Tool("read_uart", {}))
    _assert_valid_produces_meta(_Tool("read_uart", None))


@pytest.mark.parametrize("value", ["result", "artifact"])
def test_the_two_declared_values_are_valid(value):
    _assert_valid_produces_meta(_Tool("t", {PRODUCES_META_KEY: value}))


@pytest.mark.parametrize("bad", ["ARTIFACT", "Artifact", "artifacts", "file",
                                 "", 1, True, ["artifact"]])
def test_anything_else_fails_the_build(bad):
    with pytest.raises(AssertionError, match="produces"):
        _assert_valid_produces_meta(_Tool("dump_firmware", {PRODUCES_META_KEY: bad}))


def test_the_message_names_the_tool_and_the_bad_value():
    with pytest.raises(AssertionError) as e:
        _assert_valid_produces_meta(_Tool("dump_firmware",
                                          {PRODUCES_META_KEY: "ARTIFACT"}))
    assert "dump_firmware" in str(e.value) and "ARTIFACT" in str(e.value)
