"""Tests for agent_core.agent.Agent + HandlerContext."""
import asyncio

import pytest

from agent_core.agent import Agent, HandlerContext
from agent_core.config import BaseConfig
from agent_core.conversation import Conversation


def test_agent_setup_default_is_noop():
    """Default Agent.setup() does nothing and doesn't raise."""
    class MyAgent(Agent):
        name = "test"

    a = MyAgent()
    a.setup()  # should not raise


def test_agent_handle_chat_default_raises():
    class MyAgent(Agent):
        name = "test"

    a = MyAgent()

    async def consume():
        async for _ in a.handle_chat(None, None):
            pass

    with pytest.raises(NotImplementedError):
        asyncio.run(consume())


def test_agent_handle_command_default_raises():
    class MyAgent(Agent):
        name = "test"

    a = MyAgent()

    async def consume():
        async for _ in a.handle_command(None, None):
            pass

    with pytest.raises(NotImplementedError):
        asyncio.run(consume())


def test_agent_system_prompt_default_raises():
    class MyAgent(Agent):
        name = "test"

    a = MyAgent()
    with pytest.raises(NotImplementedError):
        a.system_prompt(None)


def test_agent_decide_mode_delegates_to_reasoning():
    """Default decide_mode returns whatever agent_core.reasoning.decide_mode returns."""
    class MyAgent(Agent):
        name = "test"

    a = MyAgent()
    conv = Conversation(history_depth=10)
    # No reasoning override on conversation; default mode is "auto".
    result = a.decide_mode(conv)
    assert result in ("on", "off", "auto")


def test_agent_subclass_can_override_setup():
    class MyAgent(Agent):
        name = "test"

        def setup(self):
            self.custom = "wired"

    a = MyAgent()
    a.setup()
    assert a.custom == "wired"


def test_handler_context_carries_conversation_and_channel():
    conv = Conversation(history_depth=10)
    ctx = HandlerContext(conversation=conv, channel_id="C1", writer=None)
    assert ctx.conversation is conv
    assert ctx.channel_id == "C1"
    assert ctx.writer is None


def test_agent_classvars_default():
    from agent_core.agent import Agent
    assert Agent.tools == []
    assert Agent.commands == []
    assert Agent.disabled_builtins == frozenset()


def test_agent_classvars_subclass():
    from agent_core.agent import Agent
    from agent_core.tools.base import Tool
    from agent_core.commands.base import Command

    class T1(Tool):
        name = "t1"
        description = ""
        parameters = {}
        async def run(self, args, ctx): return ""

    class C1(Command):
        name = "c1"
        args = ""
        description = ""
        async def run(self, raw_args, ctx):
            yield None

    class MyAgent(Agent):
        name = "myagent"
        tools = [T1]
        commands = [C1]
        disabled_builtins = frozenset({"grep"})

    assert MyAgent.tools == [T1]
    assert MyAgent.commands == [C1]
    assert MyAgent.disabled_builtins == frozenset({"grep"})


def test_handler_context_has_agent_and_emit():
    from agent_core.agent import HandlerContext
    fields = {f.name for f in HandlerContext.__dataclass_fields__.values()}
    assert "agent" in fields
    assert "emit" in fields


def test_handler_context_defaults_for_new_fields():
    """Existing call sites that pass only conversation/channel_id/writer
    should still work — agent and emit default to None."""
    from agent_core.agent import HandlerContext
    from agent_core.conversation import Conversation
    ctx = HandlerContext(
        conversation=Conversation(history_depth=10),
        channel_id="c1",
        writer=object(),
    )
    assert ctx.agent is None
    assert ctx.emit is None


def test_handler_context_accepts_agent_and_emit():
    """Daemon._handle_connection (Task 18) will populate these."""
    from agent_core.agent import HandlerContext
    from agent_core.conversation import Conversation

    async def fake_emit(_msg):
        pass

    sentinel_agent = object()
    ctx = HandlerContext(
        conversation=Conversation(history_depth=10),
        channel_id="c1",
        writer=object(),
        agent=sentinel_agent,
        emit=fake_emit,
    )
    assert ctx.agent is sentinel_agent
    assert ctx.emit is fake_emit
