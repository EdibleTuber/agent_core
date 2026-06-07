"""Smoke tests for builtin commands.

Each test verifies the command yields at least one message and calls the
correct manager method. API deviations from the plan are noted inline.

Key deviations discovered:
  - InferenceClient: .default_model not .model
  - LearningManager: .list() not .list_candidates(); .add_rating(rating, comment)
    not .rate(id, score); .mark_promoted(slug) not .promote()
  - ChannelStore: .get_or_create(channel_id) is async; no sync .conversation()
  - Scratchpad: constructed directly from config attrs (no ChannelStore.scratchpad())
  - HandlerContext: .agent is a duck-typed extra attr, not a dataclass field
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_core.commands._builtin_impls import (
    Clear,
    Context,
    Help,
    Learnings,
    Model,
    Profile,
    Promote,
    Quit,
    Rate,
    Scratch,
    Status,
    Think,
    Wisdom,
)
from agent_core.inference import Usage
from agent_core.conversation import Conversation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(it):
    return [m async for m in it]


def _body(msgs):
    return "\n".join(getattr(m, "text", "") for m in msgs)


def _ctx(agent, channel_id="default"):
    """Minimal duck-typed HandlerContext for command tests."""
    class _C:
        pass
    c = _C()
    c.agent = agent
    c.channel_id = channel_id
    # Commands that use /think write to ctx.conversation.overrides
    c.conversation = Conversation(history_depth=10)
    return c


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

async def test_help_lists_commands():
    cr = MagicMock()
    cr.metadata.return_value = [
        ("hello", "[<name>]", "Say hi"),
        ("quit", "", "Exit"),
    ]
    agent = MagicMock(command_registry=cr)
    msgs = await _collect(Help().run("", _ctx(agent)))
    assert len(msgs) >= 1
    body = _body(msgs)
    assert "hello" in body
    assert "Say hi" in body
    assert "quit" in body


async def test_help_formats_args():
    cr = MagicMock()
    cr.metadata.return_value = [("status", "", "Show status")]
    agent = MagicMock(command_registry=cr)
    msgs = await _collect(Help().run("", _ctx(agent)))
    body = _body(msgs)
    assert "/status" in body
    assert "Show status" in body


# ---------------------------------------------------------------------------
# /clear
# ---------------------------------------------------------------------------

async def test_clear_resets_conversation():
    # ChannelStore.get_or_create is async; channels must be an AsyncMock.
    conv = MagicMock()
    channels = MagicMock()
    channels.get_or_create = AsyncMock(return_value=conv)
    agent = MagicMock(channels=channels)

    msgs = await _collect(Clear().run("", _ctx(agent)))
    channels.get_or_create.assert_awaited_once_with("default")
    conv.clear.assert_called_once()
    assert len(msgs) == 1
    assert "cleared" in msgs[0].text.lower()


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

async def test_status_shows_basic_info():
    cfg = MagicMock(model="m1", vault_path="/v")
    agent = MagicMock(config=cfg, name="pal")
    msgs = await _collect(Status().run("", _ctx(agent)))
    body = _body(msgs)
    assert "pal" in body
    assert "m1" in body
    assert "/v" in body


async def test_status_shows_channel():
    cfg = MagicMock(model="x", vault_path="/x")
    agent = MagicMock(config=cfg, name="bot")
    msgs = await _collect(Status().run("", _ctx(agent, channel_id="chan42")))
    body = _body(msgs)
    assert "chan42" in body


# ---------------------------------------------------------------------------
# /profile
# ---------------------------------------------------------------------------

async def test_profile_read():
    profile = MagicMock()
    profile.read.return_value = "I am a user."
    agent = MagicMock(profile=profile)
    msgs = await _collect(Profile().run("", _ctx(agent)))
    body = _body(msgs)
    assert "I am a user." in body


async def test_profile_empty():
    profile = MagicMock()
    profile.read.return_value = ""
    agent = MagicMock(profile=profile)
    msgs = await _collect(Profile().run("", _ctx(agent)))
    body = _body(msgs)
    assert "empty" in body.lower()


# ---------------------------------------------------------------------------
# /scratch
# ---------------------------------------------------------------------------

async def test_scratch_read_empty(tmp_path):
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=2048)
    agent = MagicMock(config=cfg, name="pal")
    msgs = await _collect(Scratch().run("", _ctx(agent, channel_id="ch1")))
    body = _body(msgs)
    assert "empty" in body.lower()


async def test_scratch_append_and_read(tmp_path):
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=2048)
    agent = MagicMock(config=cfg, name="pal")
    ctx = _ctx(agent, channel_id="ch2")

    msgs = await _collect(Scratch().run("hello scratch", ctx))
    assert "appended" in _body(msgs).lower()

    msgs = await _collect(Scratch().run("", ctx))
    assert "hello scratch" in _body(msgs)


async def test_scratch_clear(tmp_path):
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=2048)
    agent = MagicMock(config=cfg, name="pal")
    ctx = _ctx(agent, channel_id="ch3")

    await _collect(Scratch().run("some text", ctx))
    msgs = await _collect(Scratch().run("clear", ctx))
    assert "cleared" in _body(msgs).lower()

    msgs = await _collect(Scratch().run("", ctx))
    assert "empty" in _body(msgs).lower()


async def test_scratch_too_large(tmp_path):
    cfg = MagicMock(vault_path=tmp_path, scratchpad_max_bytes=10)
    agent = MagicMock(config=cfg, name="pal")
    ctx = _ctx(agent, channel_id="ch4")

    msgs = await _collect(Scratch().run("x" * 20, ctx))
    body = _body(msgs)
    assert "too large" in body.lower() or "error" in body.lower()


# ---------------------------------------------------------------------------
# /wisdom
# ---------------------------------------------------------------------------

async def test_wisdom_list_empty():
    wisdom = MagicMock()
    wisdom.list.return_value = []
    agent = MagicMock(wisdom=wisdom)
    msgs = await _collect(Wisdom().run("", _ctx(agent)))
    body = _body(msgs)
    assert "no wisdom" in body.lower()


async def test_wisdom_list_entries():
    wisdom = MagicMock()
    wisdom.list.return_value = [
        {"slug": "be-kind", "title": "Be kind"},
        {"slug": "think-first", "title": "Think first"},
    ]
    agent = MagicMock(wisdom=wisdom)
    msgs = await _collect(Wisdom().run("", _ctx(agent)))
    body = _body(msgs)
    assert "be-kind" in body
    assert "Be kind" in body
    assert "think-first" in body


async def test_wisdom_add():
    wisdom = MagicMock()
    wisdom.add.return_value = "be-helpful"
    agent = MagicMock(wisdom=wisdom)
    msgs = await _collect(Wisdom().run("add Be helpful", _ctx(agent)))
    wisdom.add.assert_called_once_with(title="Be helpful", body="")
    body = _body(msgs)
    assert "be-helpful" in body


async def test_wisdom_remove():
    wisdom = MagicMock()
    agent = MagicMock(wisdom=wisdom)
    msgs = await _collect(Wisdom().run("remove be-kind", _ctx(agent)))
    wisdom.remove.assert_called_once_with("be-kind")
    assert "removed" in _body(msgs).lower()


async def test_wisdom_remove_not_found():
    wisdom = MagicMock()
    wisdom.remove.side_effect = FileNotFoundError("nope")
    agent = MagicMock(wisdom=wisdom)
    msgs = await _collect(Wisdom().run("remove ghost", _ctx(agent)))
    body = _body(msgs)
    assert "not found" in body.lower()


# ---------------------------------------------------------------------------
# /learnings
# ---------------------------------------------------------------------------

async def test_learnings_list_empty():
    learning = MagicMock()
    # Real method is .list() -> [{slug, title, status}]
    learning.list.return_value = []
    agent = MagicMock(learning=learning)
    msgs = await _collect(Learnings().run("", _ctx(agent)))
    body = _body(msgs)
    assert "no learnings" in body.lower()


async def test_learnings_list_entries():
    learning = MagicMock()
    learning.list.return_value = [
        {"slug": "lesson-a", "title": "Lesson A", "status": "active"},
        {"slug": "lesson-b", "title": "Lesson B", "status": "promoted"},
    ]
    agent = MagicMock(learning=learning)
    msgs = await _collect(Learnings().run("", _ctx(agent)))
    body = _body(msgs)
    assert "lesson-a" in body
    assert "active" in body
    assert "promoted" in body


# ---------------------------------------------------------------------------
# /promote
# ---------------------------------------------------------------------------

async def test_promote_happy_path():
    learning = MagicMock()
    learning.get.return_value = "The lesson body."
    learning.get_meta.return_value = {"title": "My Lesson", "status": "active"}
    wisdom = MagicMock()
    wisdom.add.return_value = "my-lesson"
    agent = MagicMock(learning=learning, wisdom=wisdom)

    msgs = await _collect(Promote().run("my-lesson-slug", _ctx(agent)))

    learning.get.assert_called_once_with("my-lesson-slug")
    learning.get_meta.assert_called_once_with("my-lesson-slug")
    wisdom.add.assert_called_once_with(title="My Lesson", body="The lesson body.")
    learning.mark_promoted.assert_called_once_with("my-lesson-slug")
    body = _body(msgs)
    assert "my-lesson" in body


async def test_promote_not_found():
    learning = MagicMock()
    learning.get.side_effect = FileNotFoundError("nope")
    wisdom = MagicMock()
    agent = MagicMock(learning=learning, wisdom=wisdom)

    msgs = await _collect(Promote().run("ghost", _ctx(agent)))
    body = _body(msgs)
    assert "not found" in body.lower()


async def test_promote_no_args():
    agent = MagicMock()
    msgs = await _collect(Promote().run("", _ctx(agent)))
    body = _body(msgs)
    assert "usage" in body.lower()


# ---------------------------------------------------------------------------
# /rate
# ---------------------------------------------------------------------------

async def test_rate_records_rating():
    # Real API: LearningManager.add_rating(rating, comment) -- NOT .rate(id, score)
    # The plan's "/rate <id> <1-5>" semantics are not supported; the command
    # uses add_rating(label, comment) instead.
    learning = MagicMock()
    agent = MagicMock(learning=learning)
    msgs = await _collect(Rate().run("good", _ctx(agent)))
    learning.add_rating.assert_called_once_with("good", "")
    body = _body(msgs)
    assert "good" in body


async def test_rate_with_comment():
    learning = MagicMock()
    agent = MagicMock(learning=learning)
    msgs = await _collect(Rate().run("5/5 very helpful session", _ctx(agent)))
    learning.add_rating.assert_called_once_with("5/5", "very helpful session")
    body = _body(msgs)
    assert "5/5" in body


async def test_rate_no_args():
    agent = MagicMock()
    msgs = await _collect(Rate().run("", _ctx(agent)))
    body = _body(msgs)
    assert "usage" in body.lower()


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------

async def test_model_show():
    # InferenceClient stores model as .default_model, not .model
    inf = MagicMock(default_model="model-a")
    agent = MagicMock(inference=inf)
    msgs = await _collect(Model().run("", _ctx(agent)))
    body = _body(msgs)
    assert "model-a" in body


async def test_model_switch():
    inf = MagicMock(default_model="old-model")
    agent = MagicMock(inference=inf)
    msgs = await _collect(Model().run("new-model-x", _ctx(agent)))
    assert inf.default_model == "new-model-x"
    body = _body(msgs)
    assert "new-model-x" in body


# ---------------------------------------------------------------------------
# /think
# ---------------------------------------------------------------------------

async def test_think_on_sets_override():
    agent = MagicMock()
    ctx = _ctx(agent)
    msgs = await _collect(Think().run("on", ctx))
    assert ctx.conversation.overrides.get("reasoning") == "on"
    body = _body(msgs)
    assert "on" in body.lower()


async def test_think_off_sets_override():
    agent = MagicMock()
    ctx = _ctx(agent)
    msgs = await _collect(Think().run("off", ctx))
    assert ctx.conversation.overrides.get("reasoning") == "off"
    body = _body(msgs)
    assert "off" in body.lower()


async def test_think_auto_clears_override():
    agent = MagicMock()
    ctx = _ctx(agent)
    ctx.conversation.overrides["reasoning"] = "on"
    msgs = await _collect(Think().run("auto", ctx))
    assert "reasoning" not in ctx.conversation.overrides
    body = _body(msgs)
    assert "auto" in body.lower()


async def test_think_show():
    agent = MagicMock()
    ctx = _ctx(agent)
    msgs = await _collect(Think().run("show", ctx))
    body = _body(msgs)
    assert "show" in body.lower()


async def test_think_hide():
    agent = MagicMock()
    ctx = _ctx(agent)
    msgs = await _collect(Think().run("hide", ctx))
    body = _body(msgs)
    assert "hide" in body.lower()


async def test_think_status_no_args():
    agent = MagicMock()
    ctx = _ctx(agent)
    ctx.conversation.overrides["reasoning"] = "on"
    msgs = await _collect(Think().run("", ctx))
    body = _body(msgs)
    assert "on" in body.lower()


async def test_think_invalid_arg():
    agent = MagicMock()
    ctx = _ctx(agent)
    msgs = await _collect(Think().run("maybe", ctx))
    body = _body(msgs)
    assert "usage" in body.lower()


# ---------------------------------------------------------------------------
# /quit
# ---------------------------------------------------------------------------

async def test_quit_yields_response():
    agent = MagicMock()
    msgs = await _collect(Quit().run("", _ctx(agent)))
    assert len(msgs) >= 1
    body = _body(msgs)
    assert "goodbye" in body.lower() or len(body) > 0


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

def _context_agent(tmp_path: Path, *, last_usage=None, schemas=None, prompt="System prompt body"):
    """Build a minimal agent for /context tests."""
    @dataclass
    class _Cfg:
        vault_path: Path
        scratchpad_max_bytes: int = 2048

    class _Agent:
        name = "test"
        config = _Cfg(tmp_path)

        def __init__(self, prompt_text):
            self._prompt = prompt_text
            self.last_usage = {}

        def system_prompt(self, ctx) -> str:
            return self._prompt

    agent = _Agent(prompt)
    if last_usage is not None:
        agent.last_usage = dict(last_usage)
    if schemas is not None:
        executor = MagicMock()
        executor.schemas.return_value = schemas
        agent.tool_executor = executor
    return agent


async def test_context_reports_no_usage_when_unrecorded(tmp_path):
    agent = _context_agent(tmp_path, schemas=[])
    msgs = await _collect(Context().run("", _ctx(agent, channel_id="ch-1")))
    body = _body(msgs)
    assert "no usage recorded" in body.lower()
    assert "ch-1" in body


async def test_context_reports_recorded_usage(tmp_path):
    usage = Usage(prompt_tokens=1234, completion_tokens=56, total_tokens=1290, model="gemma-x")
    agent = _context_agent(
        tmp_path,
        last_usage={"ch-1": usage},
        schemas=[{"type": "function", "function": {"name": "x", "description": "y"}}],
    )
    msgs = await _collect(Context().run("", _ctx(agent, channel_id="ch-1")))
    body = _body(msgs)
    assert "1234" in body
    assert "56" in body
    assert "1290" in body
    assert "gemma-x" in body


async def test_context_reports_other_channel_separately(tmp_path):
    usage = Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    agent = _context_agent(tmp_path, last_usage={"ch-other": usage}, schemas=[])
    msgs = await _collect(Context().run("", _ctx(agent, channel_id="ch-mine")))
    body = _body(msgs)
    # Channel mine has nothing recorded; should NOT show ch-other's totals.
    assert "no usage recorded" in body.lower()
    assert "ch-mine" in body


async def test_context_includes_component_byte_breakdown(tmp_path):
    schemas = [{"type": "function", "function": {"name": "x", "description": "y" * 200}}]
    agent = _context_agent(tmp_path, schemas=schemas, prompt="A" * 500)
    msgs = await _collect(Context().run("", _ctx(agent, channel_id="ch-1")))
    body = _body(msgs)
    assert "System prompt" in body
    assert "Tool schemas" in body
    assert "History" in body
    assert "Scratchpad" in body
    # System prompt is 500 'A' bytes; the byte count must appear.
    assert "500" in body


async def test_context_handles_missing_tool_executor(tmp_path):
    """No tool_executor attached: tool schemas should be reported as 0 bytes / 0 tools."""
    agent = _context_agent(tmp_path)  # no schemas arg => no tool_executor attr
    msgs = await _collect(Context().run("", _ctx(agent, channel_id="ch-1")))
    body = _body(msgs)
    assert "0 tools" in body


# ---------------------------------------------------------------------------
# Agent.record_usage / Agent.last_usage
# ---------------------------------------------------------------------------


def test_record_usage_stores_per_channel():
    """Multi-channel usage is keyed by channel_id."""
    from agent_core.agent import Agent

    class _A(Agent):
        name = "test"

    a = _A()
    a.record_usage("ch-1", Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12))
    a.record_usage("ch-2", Usage(prompt_tokens=20, completion_tokens=4, total_tokens=24))
    assert a.last_usage["ch-1"].total_tokens == 12
    assert a.last_usage["ch-2"].total_tokens == 24


def test_record_usage_none_is_noop():
    """record_usage(None) doesn't crash and doesn't insert anything."""
    from agent_core.agent import Agent

    class _A(Agent):
        name = "test"

    a = _A()
    a.record_usage("ch-1", None)
    assert a.last_usage == {}


def test_record_usage_lazy_init_when_super_init_skipped():
    """Subclasses that override __init__ without calling super still work."""
    from agent_core.agent import Agent

    class _A(Agent):
        name = "test"

        def __init__(self):
            # Deliberately do NOT call super().__init__()
            pass

    a = _A()
    assert not hasattr(a, "last_usage")  # confirms super init was skipped
    a.record_usage("ch-1", Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    assert a.last_usage["ch-1"].total_tokens == 2


# ---------------------------------------------------------------------------
# /quit
# ---------------------------------------------------------------------------

async def test_quit_signals_end_session():
    msgs = await _collect(Quit().run("", _ctx(agent=None)))
    assert len(msgs) == 1
    assert msgs[0].text == "Goodbye."
    assert msgs[0].end_session is True
