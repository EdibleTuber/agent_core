"""Per-package pytest configuration. Re-exports fixtures from the
fixtures subpackage so test files can use them without explicit imports."""
from tests.workers.fixtures import streamable_http_fixture  # noqa: F401
