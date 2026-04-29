"""In-memory conversation history with optional JSONL persistence.

Maintains a rolling in-memory window of messages, truncated to `history_depth`.
When `history_path` is set, every message is also appended to a JSONL file on
disk, enabling replay across daemon restarts (see agent_core.channels.ChannelStore).
The in-memory window is bounded; the on-disk log grows unbounded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Conversation:
    history_depth: int
    history_path: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    _messages: list[dict] = field(default_factory=list)

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def _append_to_history_file(self, message: dict) -> None:
        """Append a single message to the history JSONL file, if configured."""
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def add_user(self, text: str) -> None:
        message = {"role": "user", "content": text}
        self._messages.append(message)
        self._append_to_history_file(message)
        self._truncate()

    def add_assistant(self, text: str) -> None:
        message = {"role": "assistant", "content": text}
        self._messages.append(message)
        self._append_to_history_file(message)
        self._truncate()

    def add_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """Record an assistant message that contains tool calls (no text content)."""
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        self._messages.append(message)
        self._append_to_history_file(message)
        self._truncate()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Record a tool result message."""
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self._messages.append(message)
        self._append_to_history_file(message)
        self._truncate()

    def get_messages_for_api(self, system_prompt: str) -> list[dict]:
        """Return message list for the inference API: system + history."""
        return [{"role": "system", "content": system_prompt}] + self.messages

    def clear(self) -> None:
        self._messages.clear()

    def _truncate(self) -> None:
        if len(self._messages) > self.history_depth:
            self._messages = self._messages[-self.history_depth:]
            # Don't start with orphaned tool messages that lost their
            # matching counterpart during truncation. Drop leading
            # assistant(tool_calls) and tool result messages.
            changed = True
            while changed:
                changed = False
                if self._messages and self._messages[0].get("role") == "tool":
                    self._messages.pop(0)
                    changed = True
                elif (
                    self._messages
                    and self._messages[0].get("role") == "assistant"
                    and self._messages[0].get("tool_calls")
                ):
                    self._messages.pop(0)
                    changed = True
