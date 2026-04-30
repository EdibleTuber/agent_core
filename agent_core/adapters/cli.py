"""Generic terminal REPL for agent_core daemons.

Connects to the daemon's socket, reads input via prompt-toolkit, sends chat or
command messages, renders streamed responses. Agent-specific message rendering
is delegated to a Renderer protocol; the REPL falls back to default rendering
for messages the renderer doesn't claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from agent_core.client import DaemonConnection
from agent_core.protocol import (
    ChatMessage,
    CommandMessage,
    ErrorMessage,
    LearningCandidateProposalMessage,
    ResponseMessage,
    StreamChunkMessage,
    ToolProgressMessage,
)


@runtime_checkable
class Renderer(Protocol):
    """Plug-in for agent-specific rendering. The REPL falls back to default
    formatting when format_message returns None."""
    def splash(self) -> str:
        """Return banner text printed at REPL start."""
        ...

    def format_message(self, msg: object) -> str | None:
        """Return formatted text for an agent-specific message, or None to
        defer to the default rendering."""
        ...


def _default_format(msg: object) -> str:
    """Default rendering for the seven generic message types."""
    if isinstance(msg, StreamChunkMessage):
        return msg.token  # printed without newline; concatenated by caller
    if isinstance(msg, ResponseMessage):
        return msg.text
    if isinstance(msg, ErrorMessage):
        return f"Error: {msg.error}"
    if isinstance(msg, ToolProgressMessage):
        return f"  [{msg.tool}({msg.arguments})]"
    if isinstance(msg, LearningCandidateProposalMessage):
        return f"\n[Learning candidate: {msg.title}]\n{msg.body}\n"
    return f"[unrendered {type(msg).__name__}]"


async def run_repl(socket_path: Path, renderer: Renderer) -> None:
    """Connect, run the input loop, render messages until the user exits."""
    print(renderer.splash())
    conn = DaemonConnection(socket_path)
    await conn.connect()
    history_path = Path.home() / ".local" / "state" / "agent_core" / "cli_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession = PromptSession(history=FileHistory(str(history_path)))
    try:
        while True:
            try:
                line = await session.prompt_async("> ")
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split(None, 1)
                name = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                await conn.send(CommandMessage(name=name, args=args))
            else:
                await conn.send(ChatMessage(text=line))

            # Drain responses until the daemon signals end-of-turn.
            async for msg in conn.receive():
                rendered = renderer.format_message(msg)
                if rendered is None:
                    rendered = _default_format(msg)
                if isinstance(msg, StreamChunkMessage):
                    print(rendered, end="", flush=True)
                else:
                    print(rendered, flush=True)
                if isinstance(msg, (ResponseMessage, ErrorMessage)):
                    break
    finally:
        await conn.close()
