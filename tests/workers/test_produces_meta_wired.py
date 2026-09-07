"""Pins that both live conformance suites actually CALL
_assert_valid_produces_meta on every listed tool, not just that the helper
itself is correct.

Without this, `_assert_valid_produces_meta` could be deleted from either
`assert_streamable_http_conformance` or `assert_stdio_conformance` and the
rest of the suite would stay green -- the same silent-fallback failure mode
this whole check exists to prevent, one level up. Each test below patches the
helper to unconditionally raise a distinctive AssertionError and asserts that
the corresponding live conformance function propagates it; that only happens
if the call site still invokes the (now-patched) function. The two tests are
independent so that removing either call site alone fails only the matching
test.
"""
import pytest

import agent_core.workers.conformance as conformance


@pytest.fixture(autouse=True)
def _make_produces_check_blow_up(monkeypatch):
    def _raise(tool):
        raise AssertionError(f"WIRING-PROOF-CALLED for {getattr(tool, 'name', tool)!r}")
    monkeypatch.setattr(conformance, "_assert_valid_produces_meta", _raise)


@pytest.mark.asyncio
async def test_stdio_call_site_invokes_it(stdio_fixture_spec):
    with pytest.raises(AssertionError, match="WIRING-PROOF-CALLED"):
        await conformance.assert_stdio_conformance(stdio_fixture_spec)


@pytest.mark.asyncio
async def test_streamable_http_call_site_invokes_it(streamable_http_fixture):
    with pytest.raises(AssertionError, match="WIRING-PROOF-CALLED"):
        await conformance.assert_streamable_http_conformance(streamable_http_fixture)
