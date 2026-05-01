"""Agent base class and HandlerContext.

The Agent is the extension surface for agent_core consumers. Subclasses set
`name` (and optionally `env_prefix`), implement the four override points, and
pass an instance to `run_daemon`. Framework managers (profile, wisdom,
learning, allowlist, approval_registry, channels, inference, retrieval,
websearch) are populated on the agent instance by `run_daemon` before
`setup()` runs, so `setup()` can use them to construct domain-specific
resources.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, ClassVar

from agent_core.config import BaseConfig
from agent_core.conversation import Conversation


@dataclass
class HandlerContext:
    """Per-turn context passed to handle_chat / handle_command.

    Carries the live Conversation, the resolved channel_id, and an
    `asyncio.StreamWriter` reference for streaming partial responses
    (StreamChunkMessage etc.) to the connected client mid-turn.
    """
    conversation: Conversation
    channel_id: str
    writer: object   # asyncio.StreamWriter; framework-internal


class Agent:
    """Base class for agent_core agents.

    Required attributes (set by subclasses):
        name: short slug for the agent (e.g. "pal", "re-lab")
        env_prefix: optional explicit env var prefix; if None, derived from name

    Framework attributes (populated by run_daemon before setup):
        config, profile, wisdom, learning, allowlist, approval_registry,
        channels, inference, retrieval, websearch
    """

    name: ClassVar[str]
    env_prefix: ClassVar[str | None] = None

    config: BaseConfig

    def setup(self) -> None:
        """Override to construct domain-specific resources. Framework managers
        are already populated when this runs."""
        pass

    def system_prompt(self, ctx: HandlerContext) -> str:
        """Return the system prompt for this turn. Override per agent."""
        raise NotImplementedError

    async def handle_chat(
        self, msg, ctx: HandlerContext,
    ) -> AsyncIterator[object]:
        """Handle a ChatMessage. Yield response messages (StreamChunk, Response,
        Error, ToolProgress, agent-specific proposal types)."""
        raise NotImplementedError
        yield  # pragma: no cover  (makes the function an async generator)

    async def handle_command(
        self, msg, ctx: HandlerContext,
    ) -> AsyncIterator[object]:
        """Handle a CommandMessage. Yield response messages."""
        raise NotImplementedError
        yield  # pragma: no cover

    async def handle_other(self, msg, ctx: HandlerContext) -> None:
        """Handle messages that aren't ChatMessage or CommandMessage.

        Default: no-op. The base Daemon dispatches non-Chat/non-Command messages
        here so agents can route domain-specific message types (approval
        responses, batch fallback choices, etc.) without subclassing the daemon.
        Synchronous (does not yield); the daemon does not write any response on
        the agent's behalf.
        """
        pass

    def decide_mode(self, conversation: Conversation) -> str:
        """Return 'on' / 'off' / 'auto' for reasoning mode. Default delegates
        to agent_core.reasoning.decide_mode."""
        from agent_core.reasoning import decide_mode
        return decide_mode(conversation)
