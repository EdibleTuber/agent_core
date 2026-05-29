"""Per-package pytest configuration. Re-exports fixtures from the
fixtures subpackage so test files can use them without explicit imports."""
import sys

import pytest

from agent_core.workers.types import WorkerSpec
from tests.workers.fixtures import streamable_http_fixture  # noqa: F401


@pytest.fixture
def stdio_stub_spec():
    """Factory: build a stdio-stub WorkerSpec with a given name + risk tier."""
    def _make(name: str, tier: str) -> WorkerSpec:
        return WorkerSpec(
            name=name,
            transport="stdio",
            risk_default=tier,
            command=sys.executable,
            args=["-m", "tests.workers.fixtures.stdio_stub"],
        )
    return _make


@pytest.fixture
def stdio_fixture_spec() -> WorkerSpec:
    """A WorkerSpec pointing at the stdio FastMCP stub.

    Uses `sys.executable` to ensure the subprocess runs in the same Python
    environment (with fastmcp installed). The stub is launched as a module
    so its sys.path is consistent regardless of CWD.
    """
    return WorkerSpec(
        name="stdio_stub",
        transport="stdio",
        risk_default="low",
        command=sys.executable,
        args=["-m", "tests.workers.fixtures.stdio_stub"],
    )
