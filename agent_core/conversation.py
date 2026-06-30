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
        """Reset the conversation: empty the in-memory window AND truncate the
        on-disk log, so a cleared conversation does not resurrect via replay on
        the next daemon restart. No-op on disk when history_path is unset."""
        self._messages.clear()
        if self.history_path is not None and self.history_path.exists():
            self.history_path.write_text("", encoding="utf-8")

    @staticmethod
    def _is_valid_window_start(message: dict) -> bool:
        """A window may not begin with a message that needs a preceding
        counterpart: a tool result (needs its assistant tool_calls) or an
        assistant message carrying tool_calls (needs its tool results)."""
        if message.get("role") == "tool":
            return False
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return False
        return True

    def _truncate(self) -> None:
        if len(self._messages) <= self.history_depth:
            return
        # Start the window at the most recent non-orphan boundary. Prefer the
        # earliest valid start within the last `history_depth` messages (keeps
        # the window near the depth budget). If the whole tail is one unbroken
        # run of tool exchanges — a single turn longer than history_depth — fall
        # back to the nearest valid start *before* the cutoff so the turn stays
        # intact and anchored, rather than draining the window to empty (the
        # amnesia bug). Worst case keep everything; never return [].
        cutoff = len(self._messages) - self.history_depth
        start = next(
            (i for i in range(cutoff, len(self._messages))
             if self._is_valid_window_start(self._messages[i])),
            None,
        )
        if start is None:
            start = next(
                (i for i in range(cutoff - 1, -1, -1)
                 if self._is_valid_window_start(self._messages[i])),
                0,
            )
        self._messages = self._messages[start:]
