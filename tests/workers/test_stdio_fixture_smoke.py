"""Sanity check that the stdio fixture spec validates as a WorkerSpec
and the subprocess command resolves on this machine."""
from pathlib import Path

from agent_core.workers.types import WorkerSpec


def test_stdio_fixture_spec_is_valid_workerspec(stdio_fixture_spec):
    """The fixture returns a valid WorkerSpec for stdio transport."""
    assert isinstance(stdio_fixture_spec, WorkerSpec)
    assert stdio_fixture_spec.transport == "stdio"
    assert stdio_fixture_spec.command  # command path resolved
    assert "tests.workers.fixtures.stdio_stub" in stdio_fixture_spec.args


def test_stdio_stub_module_exists():
    """The stub file exists at the expected path."""
    path = Path(__file__).parent / "fixtures" / "stdio_stub.py"
    assert path.exists(), f"missing stdio stub at {path}"
