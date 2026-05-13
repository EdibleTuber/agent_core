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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, ClassVar

from agent_core.config import BaseConfig
from agent_core.conversation import Conversation

if TYPE_CHECKING:
    from agent_core.tools.base import Tool
    from agent_core.commands.base import Command
    from agent_core.inference import Usage


@dataclass
class HandlerContext:
    """Per-turn context passed to handle_chat / handle_command.

    Carries the live Conversation, the resolved channel_id, the
    asyncio.StreamWriter for streaming partial responses, the back-reference
    to the Agent (for tools and commands accessing ctx.agent.X), and an
    awaitable `emit` callable that NDJSON-encodes a message and writes it
    to the connection.

    `agent` and `emit` default to None so existing call sites (notably the
    Daemon constructed HandlerContext in earlier Phase E code paths and the
    test fixtures) keep working. Daemon._handle_connection populates them
    on every new connection / turn.
    """
    conversation: Conversation
    channel_id: str
    writer: object          # asyncio.StreamWriter; framework-internal
    agent: object = None    # Agent; populated by Daemon._handle_connection
    emit: object = None     # Callable[[object], Awaitable[None]]; populated by Daemon


class Agent:
    """Base class for agent_core agents.

    Required attributes (set by subclasses):
        name: short slug for the agent (e.g. "pal", "re-lab")
        env_prefix: optional explicit env var prefix; if None, derived from name

    Class-level registration (defaults are empty; opt-out via disabled_builtins):
        tools: list of Tool subclasses to register on top of BUILTIN_TOOLS
        commands: list of Command subclasses to register on top of BUILTIN_COMMANDS
        disabled_builtins: names to remove from both registries

    Framework attributes (populated by run_daemon before setup):
        config, profile, wisdom, learning, allowlist, approval_registry,
        channels, inference, retrieval, websearch, fetcher

    Registration attributes (populated by run_daemon._attach_registries before setup):
        tool_executor, command_registry, prompt_builder
    """

    name: ClassVar[str]
    env_prefix: ClassVar[str | None] = None

    # New in v0.6.0: declarative tool/command registration with opt-out.
    tools: ClassVar[list[type["Tool"]]] = []
    commands: ClassVar[list[type["Command"]]] = []
    disabled_builtins: ClassVar[frozenset[str]] = frozenset()

    config: BaseConfig

    def register_tools(self) -> list[type["Tool"]]:
        """Return tools to register dynamically at startup.

        Override this in subclasses that need to construct their tool list
        at runtime — for example, after MCP worker discovery, when the set
        of available tools depends on which external workers responded to
        list_tools.

        The returned list is unioned with cls.tools by the framework
        during _attach_registries. Returning [] (the default) is equivalent
        to relying purely on declarative cls.tools.
        """
        return []

    def __init__(self) -> None:
        # Per-channel last-turn token usage. Populated by record_usage().
        # Read by the /context command (and any other consumer).
        self.last_usage: dict[str, "Usage"] = {}

    def record_usage(self, channel_id: str, usage: "Usage | None") -> None:
        """Record token usage for a channel turn. Safe no-op if usage is None
        (e.g. inference server didn't emit a usage block).

        Lazy-initialises the dict if a subclass overrode __init__ without
        calling super().__init__(); reading code should use getattr with a
        default to mirror that.
        """
        if usage is None:
            return
        if not hasattr(self, "last_usage"):
            self.last_usage = {}
        self.last_usage[channel_id] = usage

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
