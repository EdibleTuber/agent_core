import pytest

from agent_core.workers.artifacts import (PRODUCES_ARTIFACT, PRODUCES_META_KEY,
                                          PRODUCES_RESULT, VALID_PRODUCES)


def test_the_daemon_states_the_wire_constant_itself():
    assert PRODUCES_META_KEY == "agent_core/produces"
    assert VALID_PRODUCES == ("result", "artifact")
    assert (PRODUCES_RESULT, PRODUCES_ARTIFACT) == ("result", "artifact")


def test_it_agrees_with_the_worker_kit():
    """The two packages are installed separately, usually on different
    machines. This test is what keeps the two statements the same, in
    whichever environment happens to have both."""
    kit = pytest.importorskip(
        "pare_worker_kit.artifacts",
        reason="pare-worker-kit is not installed here; the worker-side half "
               "of this check runs in that package's own suite")
    assert kit.PRODUCES_META_KEY == PRODUCES_META_KEY
    assert kit.VALID_PRODUCES == VALID_PRODUCES
