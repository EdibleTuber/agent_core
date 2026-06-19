"""Generic terminal REPL for agent_core daemons.

Connects to the daemon's socket, reads input via prompt-toolkit, sends chat or
command messages, renders streamed responses. Agent-specific message rendering
is delegated to a Renderer protocol; the REPL falls back to default rendering
for messages the renderer doesn't claim.
"""
from __future__ import annotations

import re
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
    ToolApprovalRequestMessage,
    ToolApprovalResponseMessage,
    ToolProgressMessage,
)


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_ARG_DISPLAY = 1000


def _sanitize_args(arguments: dict) -> str:
    rendered = []
    for k, v in (arguments or {}).items():
        s = _CONTROL_CHARS.sub(" ", str(v))
        if len(s) > 200:
            s = s[:200] + f"...(+{len(s) - 200} chars)"
        rendered.append(f"{k}={s}")
    out = ", ".join(rendered)
    if len(out) > _MAX_ARG_DISPLAY:
        out = out[:_MAX_ARG_DISPLAY] + "...(truncated)"
    return out


async def handle_approval_request(msg, prompt_fn, send_fn) -> None:
    """Render an approval request, collect the operator decision, send the
    response. prompt_fn(prompt_str)->str and send_fn(message)->None are
    injected so this is testable without a live socket."""
    is_critical = msg.effective_tier == "critical"
    print(f"\n--- approval required ---")
    print(f"  {msg.worker}.{msg.tool}  (declared={msg.declared_tier} effective={msg.effective_tier})")
    print(f"  args: {_sanitize_args(msg.arguments)}")
    opts = "[n/j]" if is_critical else "[y/n/j/a]"
    try:
        answer = (await prompt_fn(f"  approve? {opts}: ")).strip().lower()
    except (KeyboardInterrupt, EOFError):
        await send_fn(ToolApprovalResponseMessage(
            proposal_id=msg.proposal_id, approved=False, justification="cancelled"))
        return

    # Critical: never accept bare y/a — must justify.
    if is_critical and answer in ("y", "a"):
        answer = "j"

    if answer == "j":
        try:
            justification = (await prompt_fn("  justification: ")).strip()
        except (KeyboardInterrupt, EOFError):
            justification = ""
        approved = bool(justification) if is_critical else True
        await send_fn(ToolApprovalResponseMessage(
            proposal_id=msg.proposal_id, approved=approved,
            justification=justification or None, scope="once"))
    elif answer == "y":
        await send_fn(ToolApprovalResponseMessage(
            proposal_id=msg.proposal_id, approved=True, justification=None, scope="once"))
    elif answer == "a":
        await send_fn(ToolApprovalResponseMessage(
            proposal_id=msg.proposal_id, approved=True, justification=None, scope="session"))
    else:
        await send_fn(ToolApprovalResponseMessage(
            proposal_id=msg.proposal_id, approved=False,
            justification="declined at CLI"))


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


async def run_repl(
    socket_path: Path, renderer: Renderer, channel_id: str | None = None
) -> None:
    """Connect, run the input loop, render messages until the user exits.

    When `channel_id` is set, every outgoing chat/command message is tagged with
    it so the daemon routes this session to a dedicated channel. Left as None,
    messages carry no channel_id and the daemon applies its `cli-default`
    fallback (unchanged behavior for existing callers)."""
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
                await conn.send(CommandMessage(name=name, args=args, channel_id=channel_id))
            else:
                await conn.send(ChatMessage(text=line, channel_id=channel_id))

            # Drain responses until the daemon signals end-of-turn.
            should_exit = False
            async for msg in conn.receive():
                if isinstance(msg, ToolApprovalRequestMessage):
                    await handle_approval_request(
                        msg,
                        prompt_fn=session.prompt_async,
                        send_fn=conn.send,
                    )
                    continue
                rendered = renderer.format_message(msg)
                if rendered is None:
                    rendered = _default_format(msg)
                if isinstance(msg, StreamChunkMessage):
                    print(rendered, end="", flush=True)
                else:
                    print(rendered, flush=True)
                if getattr(msg, "end_session", False):
                    should_exit = True
                if isinstance(msg, (ResponseMessage, ErrorMessage)):
                    break
            if should_exit:
                break
    finally:
        await conn.close()
