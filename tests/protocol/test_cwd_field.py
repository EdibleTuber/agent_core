from agent_core.protocol.messages import ChatMessage, CommandMessage
from agent_core.protocol.transport import decode_message, encode_message
from agent_core.config import BaseConfig


def test_chat_message_carries_cwd_and_roundtrips():
    msg = ChatMessage(text="hi", channel_id="c1", cwd="/home/op/target-a")
    back = decode_message(encode_message(msg).rstrip(b"\n"))
    assert isinstance(back, ChatMessage)
    assert back.cwd == "/home/op/target-a"


def test_command_message_cwd_defaults_none():
    assert CommandMessage(name="ps", args="").cwd is None


def test_config_has_context_window_tokens_default():
    assert BaseConfig().context_window_tokens == 32768
