"""Conformance: _assert_artifact_reserved_args for artifact-declaring tools.

Tests D36-D40 from the P3 artifact-wiring plan.

D36: artifact tool missing expected_drive_id → AssertionError naming the arg.
D37: artifact tool with both reserved args → no raise.
D38: non-artifact tool with no reserved args → no raise (no-op).
D39: non-dict _meta → fails closed, identical to sibling.
D40: wiring — the helper is invoked by both live-suite functions.
"""
import asyncio

import pytest

from agent_core.workers.artifacts import (
    PRODUCES_ARTIFACT,
    PRODUCES_META_KEY,
    PRODUCES_RESULT,
    RESERVED_DRIVE_ID_ARG,
    RESERVED_SLUG_ARG,
)
from agent_core.workers.conformance import (
    _assert_artifact_reserved_args,
    assert_stdio_conformance,
    assert_streamable_http_conformance,
)
from agent_core.workers.risk import RISK_TIER_META_KEY


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

class _Tool:
    """Minimal tool stub for _assert_artifact_reserved_args tests."""

    def __init__(self, name, meta, input_schema):
        self.name = name
        self.meta = meta
        self.inputSchema = input_schema


_ARTIFACT_META = {PRODUCES_META_KEY: PRODUCES_ARTIFACT}
_RESULT_META = {PRODUCES_META_KEY: PRODUCES_RESULT}
_EMPTY_META = {}


def _artifact_tool(name="dump_firmware", properties=None):
    """Build an artifact-declaring tool.

    If *properties* is None the tool has NO properties (missing both reserved
    args).  Pass a dict to override the default properties.
    """
    if properties is None:
        properties = {"size": {"type": "string"}}
    return _Tool(
        name,
        _ARTIFACT_META,
        {"type": "object", "properties": properties},
    )


def _non_artifact_tool(name="read_uart", meta=_EMPTY_META, properties=None):
    """Build a non-artifact tool (default: no produces meta)."""
    if properties is None:
        properties = {"port": {"type": "string"}}
    return _Tool(name, meta, {"type": "object", "properties": properties})


# ---------------------------------------------------------------------------
# D36 — missing expected_drive_id raises AssertionError naming the arg
# ---------------------------------------------------------------------------

def test_d36_artifact_tool_missing_expected_drive_id_raises():
    """A tool that declares `produces: artifact` but whose inputSchema
    properties are missing the `expected_drive_id` reserved argument must
    fail at conformance time, naming the missing argument."""
    tool = _artifact_tool(name="dump_firmware", properties={
        "size": {"type": "string"},
        RESERVED_SLUG_ARG: {"type": "string"},
    })
    with pytest.raises(AssertionError, match=RESERVED_DRIVE_ID_ARG):
        _assert_artifact_reserved_args(tool)


def test_d36b_artifact_tool_missing_project_slug_raises():
    """A tool that declares `produces: artifact` but whose inputSchema
    properties are missing the `project_slug` reserved argument must
    fail at conformance time, naming the missing argument."""
    tool = _artifact_tool(name="dump_firmware", properties={
        "size": {"type": "string"},
        RESERVED_DRIVE_ID_ARG: {"type": "string"},
    })
    with pytest.raises(AssertionError, match=RESERVED_SLUG_ARG):
        _assert_artifact_reserved_args(tool)


# ---------------------------------------------------------------------------
# D37 — artifact tool with BOTH reserved args → no raise
# ---------------------------------------------------------------------------

def test_d37_artifact_tool_with_both_reserved_args_passes():
    """An artifact tool that declares BOTH reserved arguments in its
    inputSchema properties must pass (no raise)."""
    tool = _artifact_tool(name="dump_firmware", properties={
        "size": {"type": "string"},
        RESERVED_SLUG_ARG: {"type": "string"},
        RESERVED_DRIVE_ID_ARG: {"type": "string"},
    })
    # Should not raise
    _assert_artifact_reserved_args(tool)


# ---------------------------------------------------------------------------
# D38 — non-artifact tool with no reserved args → no raise (no-op)
# ---------------------------------------------------------------------------

def test_d38_non_artifact_tool_no_reserved_args_is_noop():
    """A non-artifact tool (no produces meta) with no reserved arguments
    must be a no-op — _assert_artifact_reserved_args does not inspect
    non-artifact tools."""
    tool = _non_artifact_tool(name="read_uart")
    # Should not raise
    _assert_artifact_reserved_args(tool)


def test_d38b_result_tool_no_reserved_args_is_noop():
    """A tool that declares `produces: result` with no reserved arguments
    must also be a no-op."""
    tool = _non_artifact_tool(name="read_uart", meta=_RESULT_META)
    # Should not raise
    _assert_artifact_reserved_args(tool)


# ---------------------------------------------------------------------------
# D39 — non-dict _meta fails closed, identical to sibling
# ---------------------------------------------------------------------------

def test_d39_non_dict_meta_fails_closed():
    """A `_meta` that is present but not a dict is malformed, not absent.

    Mirrors _assert_valid_produces_meta, which asserts isinstance(meta,
    dict) directly whenever _meta is present: a non-dict _meta is
    malformed, not absent, and this helper must fail the same way.
    """
    for bad_meta in [[], "nope", 42, True]:
        tool = _Tool("t", bad_meta, {"type": "object", "properties": {}})
        with pytest.raises(AssertionError, match="non-dict"):
            _assert_artifact_reserved_args(tool)


# ---------------------------------------------------------------------------
# D40 — wiring: the helper is invoked by both live-suite functions
# ---------------------------------------------------------------------------

def test_d40_wiring_assert_streamable_http_conformance_calls_helper():
    """assert_streamable_http_conformance must invoke
    _assert_artifact_reserved_args for each listed tool.

    We monkeypatch the helper with a counting wrapper that records each
    call.  We pass a minimal fake that makes list_tools return one artifact
    tool (with both reserved args, so the helper succeeds) and one
    non-artifact tool (also succeeds).  The helper runs once per listed
    tool (a no-op on non-artifact ones), so two tools give two calls."""
    from unittest.mock import patch

    call_count = [0]

    def _counting_wrapper(tool):
        call_count[0] += 1
        _assert_artifact_reserved_args(tool)

    # Build a fake client that returns our tools
    # (meta must include risk_tier because the live suite checks it first)
    fake_tool_artifact = _Tool(
        "dump_firmware",
        {PRODUCES_META_KEY: PRODUCES_ARTIFACT, RISK_TIER_META_KEY: "low"},
        {
            "type": "object",
            "properties": {
                "size": {"type": "string"},
                RESERVED_SLUG_ARG: {"type": "string"},
                RESERVED_DRIVE_ID_ARG: {"type": "string"},
            },
        },
    )
    fake_tool_result = _Tool(
        "read_uart",
        {RISK_TIER_META_KEY: "low"},
        {"type": "object", "properties": {"port": {"type": "string"}}},
    )

    class _FakeListResult:
        tools = [fake_tool_artifact, fake_tool_result]

    class _FakeClient:
        def __init__(self, endpoint):
            pass
        async def connect(self):
            pass
        async def initialize(self):
            pass
        async def list_tools(self):
            return _FakeListResult()
        async def close(self):
            pass

    with patch(
        "agent_core.workers.client.MCPClient", _FakeClient
    ), patch(
        "agent_core.workers.conformance._assert_artifact_reserved_args",
        _counting_wrapper,
    ):
        asyncio.run(assert_streamable_http_conformance("http://localhost:9999"))

    assert call_count[0] == 2, (
        f"expected _assert_artifact_reserved_args to be called once per tool "
        f"(2 tools = 2 calls), but got {call_count[0]}"
    )


def test_d40b_wiring_assert_stdio_conformance_calls_helper():
    """assert_stdio_conformance must invoke
    _assert_artifact_reserved_args for each listed tool.

    Same pattern as D40 but for the stdio conformance function.  We
    monkeypatch the helper and pass a minimal fake WorkerSpec whose
    transport is "stdio".  The fake client returns one artifact tool and
    one non-artifact tool; the helper runs once per listed tool (a no-op
    on non-artifact ones), so two tools give two calls."""
    from unittest.mock import patch

    call_count = [0]

    def _counting_wrapper(tool):
        call_count[0] += 1
        _assert_artifact_reserved_args(tool)

    # (meta must include risk_tier because the live suite checks it first)
    fake_tool_artifact = _Tool(
        "dump_firmware",
        {PRODUCES_META_KEY: PRODUCES_ARTIFACT, RISK_TIER_META_KEY: "low"},
        {
            "type": "object",
            "properties": {
                "size": {"type": "string"},
                RESERVED_SLUG_ARG: {"type": "string"},
                RESERVED_DRIVE_ID_ARG: {"type": "string"},
            },
        },
    )
    fake_tool_result = _Tool(
        "read_uart",
        {RISK_TIER_META_KEY: "low"},
        {"type": "object", "properties": {"port": {"type": "string"}}},
    )

    class _FakeListResult:
        tools = [fake_tool_artifact, fake_tool_result]

    class _FakeClient:
        def __init__(self, spec):
            pass
        @classmethod
        def from_spec(cls, spec):
            return cls(spec)
        async def connect(self):
            pass
        async def initialize(self):
            pass
        async def list_tools(self):
            return _FakeListResult()
        async def close(self):
            pass

    from agent_core.workers.types import WorkerSpec

    spec = WorkerSpec(
        name="test_worker",
        transport="stdio",
        risk_default="low",
        command="python -m test_worker",
    )

    with patch(
        "agent_core.workers.client.MCPClient", _FakeClient
    ), patch(
        "agent_core.workers.conformance._assert_artifact_reserved_args",
        _counting_wrapper,
    ):
        asyncio.run(assert_stdio_conformance(spec))

    assert call_count[0] == 2, (
        f"expected _assert_artifact_reserved_args to be called once per tool "
        f"(2 tools = 2 calls), but got {call_count[0]}"
    )


# ---------------------------------------------------------------------------
# Deferred-minor-19: None inputSchema fails closed (same as non-dict)
# ---------------------------------------------------------------------------

def test_none_input_schema_fails_closed():
    """A ``None`` inputSchema is malformed, not absent — the helper must fail
    closed the same way a non-dict schema does (an artifact tool with no
    schema cannot expose the reserved arguments).

    Before the fix the helper returned silently on ``schema is None``.
    """
    tool = _artifact_tool(name="dump_firmware", properties={
        "size": {"type": "string"},
    })
    # Override inputSchema to None
    tool.inputSchema = None
    with pytest.raises(AssertionError, match="project_slug"):
        _assert_artifact_reserved_args(tool)
