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


# ---------------------------------------------------------------------------
# run_repl loop lifecycle
# ---------------------------------------------------------------------------

from pathlib import Path

from agent_core.adapters import cli


class _FakeConn:
    """Fake DaemonConnection: yields one batch of messages per receive() call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.sent = []
        self.closed = False

    async def connect(self):
        pass

    async def send(self, msg):
        self.sent.append(msg)

    async def receive(self):
        batch = self._turns.pop(0) if self._turns else []
        for m in batch:
            yield m

    async def close(self):
        self.closed = True


class _FakePrompt:
    """Fake PromptSession: returns queued lines, then EOFError; counts calls."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.calls = 0

    async def prompt_async(self, prompt=""):
        self.calls += 1
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


class _NullRenderer:
    def splash(self) -> str:
        return ""

    def format_message(self, msg):
        return None


def _patch_cli(monkeypatch, conn, prompt):
    monkeypatch.setattr(cli, "DaemonConnection", lambda *a, **k: conn)
    monkeypatch.setattr(cli, "PromptSession", lambda *a, **k: prompt)
    monkeypatch.setattr(cli, "FileHistory", lambda *a, **k: None)


async def test_run_repl_exits_on_end_session(monkeypatch, capsys):
    conn = _FakeConn([[ResponseMessage(text="Goodbye.", end_session=True)]])
    prompt = _FakePrompt(["/quit"])
    _patch_cli(monkeypatch, conn, prompt)

    await cli.run_repl(Path("/ignored.sock"), _NullRenderer())

    # Exited after the quit turn: never prompted a second time.
    assert prompt.calls == 1
    assert conn.closed is True
    assert isinstance(conn.sent[0], CommandMessage) and conn.sent[0].name == "quit"
    assert "Goodbye." in capsys.readouterr().out


async def test_run_repl_stamps_channel_id_on_outgoing_messages(monkeypatch):
    """Given a channel_id, run_repl tags every outgoing chat and command message
    with it so the daemon routes to a per-launch channel instead of cli-default."""
    conn = _FakeConn([
        [ResponseMessage(text="ok")],
        [ResponseMessage(text="bye", end_session=True)],
    ])
    prompt = _FakePrompt(["hello", "/quit"])
    _patch_cli(monkeypatch, conn, prompt)

    await cli.run_repl(
        Path("/ignored.sock"), _NullRenderer(), channel_id="cli-20260619-120000"
    )

    chats = [m for m in conn.sent if isinstance(m, ChatMessage)]
    cmds = [m for m in conn.sent if isinstance(m, CommandMessage)]
    assert chats and chats[0].channel_id == "cli-20260619-120000"
    assert cmds and cmds[0].channel_id == "cli-20260619-120000"


async def test_run_repl_defaults_channel_id_to_none(monkeypatch):
    """Backward compat: without a channel_id, outgoing messages carry None so the
    daemon's existing cli-default fallback is unchanged for other callers."""
    conn = _FakeConn([[ResponseMessage(text="bye", end_session=True)]])
    prompt = _FakePrompt(["hi"])
    _patch_cli(monkeypatch, conn, prompt)

    await cli.run_repl(Path("/ignored.sock"), _NullRenderer())

    assert conn.sent and conn.sent[0].channel_id is None


async def test_run_repl_keeps_looping_on_normal_response(monkeypatch):
    # A normal (non-terminal) response must NOT exit the loop. Two turns:
    # a chat that replies, then an empty prompt that EOFs to end the test.
    conn = _FakeConn([[ResponseMessage(text="hi there")], []])
    prompt = _FakePrompt(["hello"])  # then exhausted -> EOFError ends the loop
    _patch_cli(monkeypatch, conn, prompt)

    await cli.run_repl(Path("/ignored.sock"), _NullRenderer())

    # Prompted again after the normal response (call 2 raised EOFError to exit).
    assert prompt.calls == 2
    assert conn.closed is True
