"""Self-test: agent_core's conformance suite passes against the
MockWorkerContract stub. Future workers import the same suite into
their own test packages and run it against their real implementation.
"""
from agent_core.workers.conformance import (
    MockWorkerContract,
    assert_conformance,
)


def test_mock_worker_passes_conformance():
    worker = MockWorkerContract()
    # Should not raise.
    assert_conformance(worker)


def test_conformance_rejects_invalid_tier():
    """A worker that declares an invalid risk tier fails conformance."""
    import pytest
    worker = MockWorkerContract()
    worker._tools["bad_tool"] = {
        "name": "bad_tool",
        "risk_tier": "lethal",  # invalid
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }
    with pytest.raises(AssertionError, match="risk_tier"):
        assert_conformance(worker)


def test_conformance_rejects_missing_version():
    """A worker that doesn't expose worker_contract_version fails."""
    import pytest
    worker = MockWorkerContract()
    worker._version = None
    with pytest.raises(AssertionError, match="contract version"):
        assert_conformance(worker)
