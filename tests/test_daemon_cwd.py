import pytest
from agent_core.agent import HandlerContext
from agent_core.protocol.messages import ChatMessage


def test_handler_context_has_cwd_default_none():
    ctx = HandlerContext(conversation=None, channel_id="c1", writer=None)
    assert ctx.cwd is None


@pytest.mark.asyncio
async def test_daemon_populates_ctx_cwd_from_message():
    class _Agent:
        name = "t"

        async def handle_chat(self, msg, ctx):
            return
            yield  # async generator

    ctx = HandlerContext(conversation=object(), channel_id="c1", writer=None,
                         agent=_Agent(), cwd="/home/op/target-b")
    async for _ in _Agent().handle_chat(ChatMessage(text="x", cwd="/home/op/target-b"), ctx):
        pass
    assert ctx.cwd == "/home/op/target-b"
