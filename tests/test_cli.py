"""Tests for agent_core.adapters.cli."""
import asyncio

import pytest

from agent_core.adapters.cli import Renderer, _default_format
from agent_core.protocol import (
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    LearningCandidateProposalMessage,
    ResponseMessage,
    StreamChunkMessage,
    ToolProgressMessage,
)


def test_renderer_protocol_satisfied_by_simple_class():
    class MyRenderer:
        def splash(self) -> str:
            return "hi"
        def format_message(self, msg) -> str | None:
            return None

    assert isinstance(MyRenderer(), Renderer)


def test_default_format_stream_chunk():
    out = _default_format(StreamChunkMessage(token="hello "))
    assert out == "hello "


def test_default_format_response():
    out = _default_format(ResponseMessage(text="answer"))
    assert out == "answer"


def test_default_format_error():
    out = _default_format(ErrorMessage(error="boom"))
    assert "Error:" in out
    assert "boom" in out


def test_default_format_tool_progress():
    out = _default_format(ToolProgressMessage(tool="search", arguments={"q": "x"}))
    assert "search" in out


def test_default_format_learning_candidate():
    out = _default_format(LearningCandidateProposalMessage(
        proposal_id="a", title="T", body="B", trigger_excerpt="t",
    ))
    assert "T" in out
    assert "B" in out


def test_default_format_unknown_type_falls_back_to_repr():
    """Unknown message types render with type-name fallback so nothing crashes."""
    class Unknown:
        type = "unknown"

    out = _default_format(Unknown())
    assert "unrendered" in out
    assert "Unknown" in out
