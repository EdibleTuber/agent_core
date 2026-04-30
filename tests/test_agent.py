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
