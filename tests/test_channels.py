"""Tests for ChannelStore — per-channel Conversation container with persistence."""
import asyncio
import json
import pytest
from pathlib import Path
from agent_core.channels import ChannelStore, validate_channel_id


def test_validate_channel_id_accepts_alphanumeric():
    assert validate_channel_id("abc123") is True
    assert validate_channel_id("ABC-123") is True
    assert validate_channel_id("cli-default") is True
    assert validate_channel_id("1234567890") is True  # Discord snowflake


def test_validate_channel_id_rejects_path_traversal():
    assert validate_channel_id("../etc") is False
    assert validate_channel_id("/absolute") is False
    assert validate_channel_id("a/b") is False
    assert validate_channel_id("") is False
    assert validate_channel_id("has space") is False
    assert validate_channel_id("has.dot") is False


@pytest.mark.asyncio
async def test_get_or_create_creates_directory(tmp_path):
    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    conv = await store.get_or_create("C1")
    assert (tmp_path / "_channels" / "testagent" / "C1").is_dir()
    assert conv.messages == []


@pytest.mark.asyncio
async def test_get_or_create_caches_instance(tmp_path):
    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    conv1 = await store.get_or_create("C1")
    conv2 = await store.get_or_create("C1")
    assert conv1 is conv2


@pytest.mark.asyncio
async def test_get_or_create_replays_existing_history(tmp_path):
    channel_dir = tmp_path / "_channels" / "testagent" / "C1"
    channel_dir.mkdir(parents=True)
    history_path = channel_dir / "history.jsonl"
    history_path.write_text(
        '{"role": "user", "content": "hi"}\n'
        '{"role": "assistant", "content": "hello"}\n'
    )

    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    conv = await store.get_or_create("C1")

    assert len(conv.messages) == 2
    assert conv.messages[0] == {"role": "user", "content": "hi"}
    assert conv.messages[1] == {"role": "assistant", "content": "hello"}


@pytest.mark.asyncio
async def test_get_or_create_skips_malformed_lines_with_warning(tmp_path, caplog):
    import logging
    channel_dir = tmp_path / "_channels" / "testagent" / "C1"
    channel_dir.mkdir(parents=True)
    (channel_dir / "history.jsonl").write_text(
        '{"role": "user", "content": "hi"}\n'
        'this is not json\n'
        '{"role": "assistant", "content": "hello"}\n'
    )

    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    with caplog.at_level(logging.WARNING):
        conv = await store.get_or_create("C1")

    assert len(conv.messages) == 2
    assert any("malformed" in rec.message.lower() or "skip" in rec.message.lower()
               for rec in caplog.records)


@pytest.mark.asyncio
async def test_get_or_create_renames_unreadable_history(tmp_path, monkeypatch):
    """If the history file can't even be opened, rename it and start fresh."""
    channel_dir = tmp_path / "_channels" / "testagent" / "C1"
    channel_dir.mkdir(parents=True)
    history_path = channel_dir / "history.jsonl"
    history_path.write_text('{"role": "user", "content": "hi"}')

    real_open = Path.open
    def patched_open(self, *args, **kwargs):
        if self == history_path and "r" in (args[0] if args else kwargs.get("mode", "r")):
            raise OSError("simulated read failure")
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", patched_open)

    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    conv = await store.get_or_create("C1")

    assert not history_path.exists()
    corrupt_files = list(channel_dir.glob("history.jsonl.corrupt-*"))
    assert len(corrupt_files) == 1
    assert conv.messages == []


@pytest.mark.asyncio
async def test_get_or_create_rejects_invalid_channel_id(tmp_path):
    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    with pytest.raises(ValueError, match="invalid channel_id"):
        await store.get_or_create("../etc")


@pytest.mark.asyncio
async def test_conversation_appends_to_history_file(tmp_path):
    """The Conversation returned from get_or_create is wired to persist new messages."""
    store = ChannelStore(vault_path=tmp_path, agent_name="testagent", history_depth=10)
    conv = await store.get_or_create("C1")
    conv.add_user("hello")

    history_path = tmp_path / "_channels" / "testagent" / "C1" / "history.jsonl"
    assert history_path.exists()
    line = history_path.read_text().strip()
    assert json.loads(line) == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_channel_path_includes_agent_name(tmp_path):
    store = ChannelStore(vault_path=tmp_path, agent_name="myagent", history_depth=10)
    conv = await store.get_or_create("C1")
    expected = tmp_path / "_channels" / "myagent" / "C1" / "history.jsonl"
    conv.add_user("hi")
    assert expected.exists()
