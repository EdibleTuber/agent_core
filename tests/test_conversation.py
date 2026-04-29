"""Tests for conversation history management."""
from agent_core.conversation import Conversation


def test_empty_conversation():
    conv = Conversation(history_depth=10)
    assert conv.messages == []


def test_add_user_message():
    conv = Conversation(history_depth=10)
    conv.add_user("hello")
    assert len(conv.messages) == 1
    assert conv.messages[0] == {"role": "user", "content": "hello"}


def test_add_assistant_message():
    conv = Conversation(history_depth=10)
    conv.add_assistant("hi there")
    assert len(conv.messages) == 1
    assert conv.messages[0] == {"role": "assistant", "content": "hi there"}


def test_history_depth_truncation():
    conv = Conversation(history_depth=4)
    for i in range(6):
        conv.add_user(f"msg {i}")
        conv.add_assistant(f"reply {i}")
    # 12 messages added, depth=4 means keep last 4
    assert len(conv.messages) == 4
    assert conv.messages[0] == {"role": "user", "content": "msg 4"}
    assert conv.messages[-1] == {"role": "assistant", "content": "reply 5"}


def test_get_messages_for_api():
    """get_messages_for_api returns system prompt + conversation history."""
    conv = Conversation(history_depth=10)
    conv.add_user("hello")
    conv.add_assistant("hi")
    system = "You are PAL."
    messages = conv.get_messages_for_api(system_prompt=system)
    assert messages[0] == {"role": "system", "content": "You are PAL."}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi"}


def test_add_tool_call_and_result():
    """Conversation stores assistant tool_calls and tool results."""
    conv = Conversation(history_depth=50)
    conv.add_user("look at quantum.md")

    conv.add_assistant_tool_calls([{
        "id": "call_001",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "Research/quantum.md"}'},
    }])

    conv.add_tool_result("call_001", "# Quantum Computing\n\nQubits are neat.")

    messages = conv.messages
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_001"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_001"
    assert "Qubits" in messages[2]["content"]


def test_truncation_drops_orphaned_tool_messages():
    """Truncation should not leave orphaned tool result messages at the start."""
    conv = Conversation(history_depth=4)
    # user -> assistant(tool_calls) -> tool result -> assistant text = 4 msgs
    conv.add_user("first question")
    conv.add_assistant_tool_calls([{
        "id": "call_001",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    conv.add_tool_result("call_001", "file contents")
    conv.add_assistant("here's what I found")
    # Now add more to trigger truncation
    conv.add_user("second question")
    conv.add_assistant("second answer")
    # depth=4 keeps last 4: tool_result, assistant, user, assistant
    # But tool_result is orphaned — should be dropped
    messages = conv.messages
    assert all(m.get("role") != "tool" for m in messages)
    # Should not start with an orphaned assistant tool_calls either
    if messages:
        first = messages[0]
        assert not (first.get("role") == "assistant" and first.get("tool_calls"))


def test_clear():
    conv = Conversation(history_depth=10)
    conv.add_user("hello")
    conv.clear()
    assert conv.messages == []


def test_overrides_default_empty():
    conv = Conversation(history_depth=10)
    assert conv.overrides == {}


def test_overrides_can_be_set_and_read():
    conv = Conversation(history_depth=10)
    conv.overrides["reasoning"] = "on"
    assert conv.overrides["reasoning"] == "on"


def test_overrides_independent_per_conversation():
    a = Conversation(history_depth=10)
    b = Conversation(history_depth=10)
    a.overrides["reasoning"] = "on"
    assert b.overrides == {}


def test_overrides_can_hold_arbitrary_keys():
    conv = Conversation(history_depth=10)
    conv.overrides["reasoning"] = "off"
    conv.overrides["foo"] = "bar"
    conv.overrides["count"] = 42
    assert conv.overrides == {"reasoning": "off", "foo": "bar", "count": 42}


def test_conversation_without_history_path_is_in_memory_only(tmp_path):
    """Backward compat: no history_path means no file written."""
    conv = Conversation(history_depth=10)
    conv.add_user("hello")
    conv.add_assistant("hi back")
    # No file should exist anywhere.
    assert not list(tmp_path.iterdir())


def test_conversation_with_history_path_appends_every_message(tmp_path):
    import json
    history_path = tmp_path / "history.jsonl"
    conv = Conversation(history_depth=10, history_path=history_path)

    conv.add_user("hello")
    conv.add_assistant("hi back")
    conv.add_assistant_tool_calls([{"id": "c1", "type": "function",
                                     "function": {"name": "foo", "arguments": "{}"}}])
    conv.add_tool_result("c1", "result")

    lines = history_path.read_text().splitlines()
    assert len(lines) == 4

    parsed = [json.loads(line) for line in lines]
    assert parsed[0] == {"role": "user", "content": "hello"}
    assert parsed[1] == {"role": "assistant", "content": "hi back"}
    assert parsed[2]["role"] == "assistant"
    assert parsed[2]["tool_calls"][0]["id"] == "c1"
    assert parsed[3] == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_conversation_history_path_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "history.jsonl"
    conv = Conversation(history_depth=10, history_path=nested)
    conv.add_user("hi")
    assert nested.exists()


def test_conversation_truncation_does_not_truncate_history_file(tmp_path):
    """Truncation only affects the in-memory window; the on-disk log keeps everything."""
    history_path = tmp_path / "history.jsonl"
    conv = Conversation(history_depth=2, history_path=history_path)
    for i in range(5):
        conv.add_user(f"msg-{i}")
    # In-memory: only last 2
    assert len(conv.messages) == 2
    # On-disk: all 5
    assert len(history_path.read_text().splitlines()) == 5
