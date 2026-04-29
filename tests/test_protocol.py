"""Tests for agent_core.protocol: transport + generic messages + registration."""
import json
import pytest

from agent_core.protocol import (
    STREAM_BUFFER_LIMIT,
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    LearningCandidateProposalMessage,
    ResponseMessage,
    StreamChunkMessage,
    ToolProgressMessage,
    decode_message,
    encode_message,
    register_message,
)


def test_stream_buffer_limit_is_16_mib():
    assert STREAM_BUFFER_LIMIT == 16 * 1024 * 1024


def test_chat_round_trip():
    msg = ChatMessage(text="hello", channel_id="cli-default")
    line = encode_message(msg)
    assert line.endswith(b"\n")
    parsed = decode_message(line[:-1])
    assert isinstance(parsed, ChatMessage)
    assert parsed.text == "hello"
    assert parsed.channel_id == "cli-default"


def test_command_round_trip():
    msg = CommandMessage(name="research", args="topic foo", channel_id="C1")
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, CommandMessage)
    assert parsed.name == "research"
    assert parsed.args == "topic foo"


def test_stream_chunk_round_trip():
    msg = StreamChunkMessage(token="hello ")
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, StreamChunkMessage)
    assert parsed.token == "hello "


def test_response_round_trip():
    msg = ResponseMessage(text="done", command="research", reasoning="thought")
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, ResponseMessage)
    assert parsed.text == "done"
    assert parsed.reasoning == "thought"


def test_error_round_trip():
    msg = ErrorMessage(error="boom")
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, ErrorMessage)
    assert parsed.error == "boom"


def test_tool_progress_round_trip():
    msg = ToolProgressMessage(tool="search", arguments={"q": "foo"})
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, ToolProgressMessage)
    assert parsed.tool == "search"
    assert parsed.arguments == {"q": "foo"}


def test_learning_candidate_proposal_round_trip():
    msg = LearningCandidateProposalMessage(
        proposal_id="abc",
        title="Use venv",
        body="The user runs everything in .venv.",
        trigger_excerpt="just use the venv",
    )
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, LearningCandidateProposalMessage)
    assert parsed.title == "Use venv"


def test_decode_unknown_type_raises():
    raw = json.dumps({"type": "not_a_real_type", "x": 1}).encode("utf-8")
    with pytest.raises(ValueError, match="Unknown message type"):
        decode_message(raw)


def test_register_message_extends_registry():
    """Verify a downstream consumer can register their own message type."""
    from dataclasses import dataclass

    @dataclass
    class CustomMessage:
        payload: str
        type: str = "custom_test_message"

    register_message(CustomMessage)

    msg = CustomMessage(payload="hello")
    parsed = decode_message(encode_message(msg)[:-1])
    assert isinstance(parsed, CustomMessage)
    assert parsed.payload == "hello"


def test_encode_uses_ndjson_format():
    msg = ChatMessage(text="hi")
    line = encode_message(msg)
    assert line.endswith(b"\n")
    obj = json.loads(line[:-1])
    assert obj["type"] == "chat"
    assert obj["text"] == "hi"
