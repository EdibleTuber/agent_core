"""Tests for agent_core.adapters.discord_gateway (Phase G)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from agent_core.adapters.discord_gateway import (
    UserConnectionManager,
    format_tool_progress,
    parse_discord_message,
    rewrite_slash_prefixes,
    split_message,
    _DISCORD_MSG_LIMIT,
)


# ---------------------------------------------------------------------------
# parse_discord_message
# ---------------------------------------------------------------------------


class TestParseDiscordMessage:
    def test_command_with_args(self):
        result = parse_discord_message("!cmd args")
        assert result == ("command", "cmd", "args")

    def test_command_without_args(self):
        result = parse_discord_message("!cmd")
        assert result == ("command", "cmd", "")

    def test_command_with_multi_word_args(self):
        result = parse_discord_message("!search hello world")
        assert result == ("command", "search", "hello world")

    def test_bare_text_returns_chat(self):
        result = parse_discord_message("hello there")
        assert result == ("chat", "hello there")

    def test_empty_string_returns_none(self):
        assert parse_discord_message("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_discord_message("   ") is None

    def test_bare_exclamation_returns_none(self):
        assert parse_discord_message("!") is None

    def test_exclamation_whitespace_only_returns_none(self):
        assert parse_discord_message("!   ") is None

    def test_leading_whitespace_stripped(self):
        result = parse_discord_message("  !cmd arg  ")
        assert result == ("command", "cmd", "arg")

    def test_chat_leading_whitespace_stripped(self):
        result = parse_discord_message("  hello  ")
        assert result == ("chat", "hello")


# ---------------------------------------------------------------------------
# rewrite_slash_prefixes
# ---------------------------------------------------------------------------


class TestRewriteSlashPrefixes:
    def test_rewrites_known_command(self):
        result = rewrite_slash_prefixes("/help", {"help"})
        assert result == "!help"

    def test_does_not_rewrite_unknown_command(self):
        result = rewrite_slash_prefixes("/unknown", {"help"})
        assert result == "/unknown"

    def test_empty_names_returns_unchanged(self):
        result = rewrite_slash_prefixes("/help", set())
        assert result == "/help"

    def test_slash_in_fenced_code_unchanged(self):
        text = "before\n```\n/help\n```\nafter"
        result = rewrite_slash_prefixes(text, {"help"})
        assert "/help" in result
        # The fenced portion should not be rewritten
        assert "```\n/help\n```" in result

    def test_slash_in_inline_code_unchanged(self):
        text = "try `/help` for info"
        result = rewrite_slash_prefixes(text, {"help"})
        assert "`/help`" in result

    def test_longest_match_wins(self):
        # 'compile-batch' should match before 'compile'
        names = {"compile", "compile-batch"}
        result = rewrite_slash_prefixes("/compile-batch", names)
        assert result == "!compile-batch"

    def test_word_boundary_protection(self):
        # '/help123' should NOT match 'help' due to \b
        result = rewrite_slash_prefixes("/help123", {"help"})
        assert result == "/help123"

    def test_command_after_whitespace(self):
        result = rewrite_slash_prefixes("please /help me", {"help"})
        assert result == "please !help me"

    def test_command_after_punctuation(self):
        result = rewrite_slash_prefixes("see /help,", {"help"})
        assert "!help" in result

    def test_multiple_commands_rewritten(self):
        result = rewrite_slash_prefixes("/help and /status", {"help", "status"})
        assert "!help" in result
        assert "!status" in result

    def test_command_at_line_start(self):
        text = "/help\n/status"
        result = rewrite_slash_prefixes(text, {"help", "status"})
        assert result == "!help\n!status"


# ---------------------------------------------------------------------------
# split_message
# ---------------------------------------------------------------------------


class TestSplitMessage:
    def test_short_text_returns_single_chunk(self):
        text = "hello world"
        result = split_message(text)
        assert result == [text]

    def test_exactly_limit_returns_single_chunk(self):
        text = "x" * _DISCORD_MSG_LIMIT
        result = split_message(text)
        assert len(result) == 1
        assert result[0] == text

    def test_over_limit_splits(self):
        text = "x" * (_DISCORD_MSG_LIMIT + 1)
        result = split_message(text)
        assert len(result) > 1

    def test_splits_at_paragraph_boundary(self):
        para1 = "a" * 1800
        para2 = "b" * 300
        text = para1 + "\n\n" + para2
        result = split_message(text)
        assert len(result) == 2
        assert result[0] == para1
        assert result[1] == para2

    def test_falls_back_to_word_boundary(self):
        # No paragraph break, but spaces available
        words = ["word"] * 600  # Each "word " is 5 chars -> ~3000 chars
        text = " ".join(words)
        result = split_message(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= _DISCORD_MSG_LIMIT

    def test_preserves_total_content_paragraph(self):
        para1 = "a" * 1800
        para2 = "b" * 300
        text = para1 + "\n\n" + para2
        result = split_message(text)
        # Paragraph split strips the "\n\n" separator
        assert "".join(result) == para1 + para2

    def test_preserves_total_content_word(self):
        words = ["word"] * 600
        text = " ".join(words)
        result = split_message(text)
        reassembled = " ".join(result)
        assert reassembled == text

    def test_custom_limit(self):
        text = "hello world foo bar"
        result = split_message(text, limit=10)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 10

    def test_hard_split_no_boundary(self):
        # No spaces or paragraphs -- falls back to hard split
        text = "x" * 4001
        result = split_message(text, limit=2000)
        assert len(result) == 3
        for chunk in result[:-1]:
            assert len(chunk) == 2000
        assert len(result[-1]) == 1


# ---------------------------------------------------------------------------
# format_tool_progress
# ---------------------------------------------------------------------------


class TestFormatToolProgress:
    def test_unknown_tool_default_label(self):
        result = format_tool_progress("x", {})
        assert result == "*[x...]*"

    def test_unknown_tool_with_empty_formatters(self):
        result = format_tool_progress("x", {}, custom_formatters={})
        assert result == "*[x...]*"

    def test_known_tool_uses_custom_formatter(self):
        formatters = {
            "read_file": lambda args: f"reading {args.get('path', '?')}...",
        }
        result = format_tool_progress("read_file", {"path": "/foo/bar"}, formatters)
        assert result == "*[reading /foo/bar...]*"

    def test_unknown_tool_falls_back_with_formatters_present(self):
        formatters = {
            "read_file": lambda args: "reading...",
        }
        result = format_tool_progress("other_tool", {}, formatters)
        assert result == "*[other_tool...]*"

    def test_result_wrapped_in_italic(self):
        result = format_tool_progress("mytool", {})
        assert result.startswith("*[")
        assert result.endswith("]*")

    def test_custom_formatter_called_with_arguments(self):
        called_with = {}

        def capture(args):
            called_with.update(args)
            return "captured"

        format_tool_progress("mytool", {"key": "val"}, {"mytool": capture})
        assert called_with == {"key": "val"}


# ---------------------------------------------------------------------------
# UserConnectionManager
# ---------------------------------------------------------------------------


class TestUserConnectionManager:
    def _make_manager(self, allowed=None):
        if allowed is None:
            allowed = {"user1", "user2"}
        return UserConnectionManager(allowed, Path("/tmp/test.sock"))

    def test_is_allowed_true_for_allowed_user(self):
        mgr = self._make_manager({"alice"})
        assert mgr.is_allowed("alice") is True

    def test_is_allowed_false_for_unknown_user(self):
        mgr = self._make_manager({"alice"})
        assert mgr.is_allowed("bob") is False

    async def test_get_client_creates_connection(self):
        mgr = self._make_manager()
        mock_conn = MagicMock()
        mock_conn.connect = AsyncMock()
        mock_conn.is_connected = True

        with patch(
            "agent_core.adapters.discord_gateway.DaemonConnection",
            return_value=mock_conn,
        ):
            client = await mgr.get_client("user1")

        mock_conn.connect.assert_awaited_once()
        assert client is mock_conn

    async def test_get_client_reuses_connected_client(self):
        mgr = self._make_manager()
        mock_conn = MagicMock()
        mock_conn.connect = AsyncMock()

        # First call: is_connected returns True after connect
        type(mock_conn).is_connected = PropertyMock(return_value=True)

        with patch(
            "agent_core.adapters.discord_gateway.DaemonConnection",
            return_value=mock_conn,
        ):
            client1 = await mgr.get_client("user1")
            client2 = await mgr.get_client("user1")

        # connect() should only be called once
        assert mock_conn.connect.await_count == 1
        assert client1 is client2

    async def test_get_client_reconnects_if_disconnected(self):
        mgr = self._make_manager()

        # First mock (disconnected on second check)
        first_conn = MagicMock()
        first_conn.connect = AsyncMock()
        type(first_conn).is_connected = PropertyMock(return_value=False)

        # Second mock (fresh connection)
        second_conn = MagicMock()
        second_conn.connect = AsyncMock()
        type(second_conn).is_connected = PropertyMock(return_value=True)

        call_count = 0

        def make_conn(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_conn
            return second_conn

        with patch(
            "agent_core.adapters.discord_gateway.DaemonConnection",
            side_effect=make_conn,
        ):
            # First call: stores first_conn (after connect)
            client1 = await mgr.get_client("user1")
            assert client1 is first_conn

            # Second call: first_conn is_connected=False, so reconnects
            client2 = await mgr.get_client("user1")
            assert client2 is second_conn

    async def test_close_all_closes_and_clears(self):
        mgr = self._make_manager()
        mock_conn1 = MagicMock()
        mock_conn1.connect = AsyncMock()
        mock_conn1.close = AsyncMock()
        type(mock_conn1).is_connected = PropertyMock(return_value=True)

        mock_conn2 = MagicMock()
        mock_conn2.connect = AsyncMock()
        mock_conn2.close = AsyncMock()
        type(mock_conn2).is_connected = PropertyMock(return_value=True)

        connections = [mock_conn1, mock_conn2]
        call_count = 0

        def make_conn(path):
            nonlocal call_count
            conn = connections[call_count]
            call_count += 1
            return conn

        with patch(
            "agent_core.adapters.discord_gateway.DaemonConnection",
            side_effect=make_conn,
        ):
            await mgr.get_client("user1")
            await mgr.get_client("user2")

        await mgr.close_all()

        mock_conn1.close.assert_awaited_once()
        mock_conn2.close.assert_awaited_once()
        assert mgr._clients == {}
