"""Implementations of the builtin commands.

Each command is a thin wrapper over a framework manager. Commands read state
from `ctx.agent.X` (framework managers) and `ctx.conversation` /
`ctx.channel_id` (per-turn context).

API deviations from the plan (discovered by reading actual modules):

  InferenceClient:  active model is .default_model, not .model
  LearningManager:  .list() not .list_candidates(); returns [{slug, title, status}]
                    .add_rating(rating, comment) not .rate(id, score)
                    .mark_promoted(slug) updates status to "promoted"
                    no built-in promote-to-wisdom; /promote does both sides
  WisdomManager:    .list() returns [{slug, title}]; .add(title, body) -> slug
                    .get(slug) -> body str; .remove(slug) -> None
  ChannelStore:     .get_or_create(channel_id) is async; no sync .conversation()
  Scratchpad:       constructed directly: Scratchpad(vault_path, agent_name,
                    channel_id, max_bytes); no ChannelStore.scratchpad() method
  Conversation:     .overrides is a plain dict; set overrides["reasoning"] for
                    /think on/off; pop to clear (auto)
  ProfileManager:   .read() -> str  (as planned)
  HandlerContext:   .agent is NOT in the dataclass; supplied by the _ctx test
                    helper and by PAL's agent dispatch -- works as duck-typed attr
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from agent_core.commands.base import Command
from agent_core.protocol.messages import ResponseMessage
from agent_core.scratchpad import Scratchpad, ScratchpadTooLarge


class Help(Command):
    name = "help"
    args = ""
    description = "Show this message"
    requires = ("command_registry",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        lines = ["Available commands:"]
        for name, args, desc in ctx.agent.command_registry.metadata():
            entry = f"  /{name}"
            if args:
                entry += f" {args}"
            entry += f"  -  {desc}"
            lines.append(entry)
        yield ResponseMessage(text="\n".join(lines))


class Clear(Command):
    name = "clear"
    args = ""
    description = "Reset the current channel's conversation"
    requires = ("channels",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        # ChannelStore.get_or_create is async; it returns the cached Conversation
        # if already loaded, or replays from disk. We call it to get the live
        # instance then clear it.
        conv = await ctx.agent.channels.get_or_create(ctx.channel_id)
        conv.clear()
        yield ResponseMessage(text="Conversation cleared.")


class Status(Command):
    name = "status"
    args = ""
    description = "Show daemon status"
    requires = ("config",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        cfg = ctx.agent.config
        agent_name = getattr(ctx.agent, "name", "agent")
        lines = [
            f"agent:   {agent_name}",
            f"model:   {getattr(cfg, 'model', '?')}",
            f"vault:   {getattr(cfg, 'vault_path', '?')}",
            f"channel: {ctx.channel_id}",
        ]
        yield ResponseMessage(text="\n".join(lines))


class Profile(Command):
    name = "profile"
    args = ""
    description = "Show the user profile"
    requires = ("profile",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        body = ctx.agent.profile.read() or "(empty)"
        yield ResponseMessage(text=body)


class Scratch(Command):
    name = "scratch"
    args = "[clear | <text>]"
    description = (
        "Read or manage the channel scratchpad. "
        "No args = read; 'clear' empties it; any other text is appended."
    )
    # Scratchpad is constructed from config; no channels.scratchpad() method.
    requires = ("config",)

    def _make_scratchpad(self, ctx) -> Scratchpad:
        cfg = ctx.agent.config
        return Scratchpad(
            vault_path=cfg.vault_path,
            agent_name=ctx.agent.name,
            channel_id=ctx.channel_id,
            max_bytes=cfg.scratchpad_max_bytes,
        )

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        sp = self._make_scratchpad(ctx)
        arg = raw_args.strip()

        if not arg:
            content = sp.read()
            yield ResponseMessage(text=content or "(scratchpad is empty)")
            return

        if arg == "clear":
            sp.write("")
            yield ResponseMessage(text="Scratchpad cleared.")
            return

        # Any other text: append
        try:
            sp.append(arg + "\n")
        except ScratchpadTooLarge as exc:
            yield ResponseMessage(
                text=(
                    f"Error: scratchpad too large after append "
                    f"({exc.proposed_bytes} bytes, cap {exc.max_bytes}). "
                    "Use '/scratch clear' to empty it."
                )
            )
            return
        yield ResponseMessage(text="Appended to scratchpad.")


class Wisdom(Command):
    name = "wisdom"
    args = "[add <title> | remove <slug>]"
    description = "List, add, or remove wisdom entries"
    requires = ("wisdom",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        arg = raw_args.strip()

        if not arg:
            entries = ctx.agent.wisdom.list()
            if not entries:
                yield ResponseMessage(text="No wisdom entries.")
                return
            lines = [f"  {e['slug']}  {e['title']}" for e in entries]
            yield ResponseMessage(text="Wisdom:\n" + "\n".join(lines))
            return

        parts = arg.split(None, 1)
        subcmd = parts[0].lower()

        if subcmd == "add":
            if len(parts) < 2:
                yield ResponseMessage(text="Usage: /wisdom add <title>")
                return
            title = parts[1].strip()
            slug = ctx.agent.wisdom.add(title=title, body="")
            yield ResponseMessage(text=f"Added wisdom: {slug}")
            return

        if subcmd == "remove":
            if len(parts) < 2:
                yield ResponseMessage(text="Usage: /wisdom remove <slug>")
                return
            slug = parts[1].strip()
            try:
                ctx.agent.wisdom.remove(slug)
            except FileNotFoundError:
                yield ResponseMessage(text=f"Wisdom not found: {slug}")
                return
            yield ResponseMessage(text=f"Removed wisdom: {slug}")
            return

        yield ResponseMessage(text="Usage: /wisdom [add <title> | remove <slug>]")


class Learnings(Command):
    name = "learnings"
    args = ""
    description = "List captured learning candidates"
    requires = ("learning",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        # LearningManager.list() returns [{slug, title, status}]
        entries = ctx.agent.learning.list()
        if not entries:
            yield ResponseMessage(text="No learnings.")
            return
        lines = [f"  [{e['status']}] {e['slug']}  {e['title']}" for e in entries]
        yield ResponseMessage(text="Learnings:\n" + "\n".join(lines))


class Promote(Command):
    name = "promote"
    args = "<slug>"
    description = "Promote a learning to wisdom"
    requires = ("learning", "wisdom")

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        slug = raw_args.strip()
        if not slug:
            yield ResponseMessage(text="Usage: /promote <slug>")
            return

        # Fetch the learning body + title
        try:
            body = ctx.agent.learning.get(slug)
            meta = ctx.agent.learning.get_meta(slug)
        except FileNotFoundError:
            yield ResponseMessage(text=f"Learning not found: {slug}")
            return

        title = meta.get("title", slug)

        # Add to wisdom
        wisdom_slug = ctx.agent.wisdom.add(title=title, body=body)

        # Mark learning as promoted
        ctx.agent.learning.mark_promoted(slug)

        yield ResponseMessage(
            text=f"Promoted '{title}' to wisdom as '{wisdom_slug}'."
        )


class Rate(Command):
    name = "rate"
    args = "<rating> [<comment>]"
    description = (
        "Append a session rating to the learning log. "
        "<rating> is any short label (e.g. good, bad, 5/5)."
    )
    # NOTE: LearningManager has no per-slug .rate() method. It has
    # .add_rating(rating, comment) which appends to a global ratings log.
    # The '/rate <id> <1-5>' semantics from the plan are NOT supported by the
    # real API. This command uses the actual .add_rating() interface instead.
    requires = ("learning",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        parts = raw_args.strip().split(None, 1)
        if not parts:
            yield ResponseMessage(text="Usage: /rate <rating> [<comment>]")
            return
        rating = parts[0]
        comment = parts[1] if len(parts) > 1 else ""
        ctx.agent.learning.add_rating(rating, comment)
        yield ResponseMessage(text=f"Rating recorded: {rating}" + (f" — {comment}" if comment else ""))


class Model(Command):
    name = "model"
    args = "[<name>]"
    description = "Show or switch the active model"
    requires = ("inference",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        target = raw_args.strip()
        if not target:
            # InferenceClient stores active model as .default_model, not .model
            current = getattr(ctx.agent.inference, "default_model", "?")
            yield ResponseMessage(text=f"model: {current}")
            return
        ctx.agent.inference.default_model = target
        yield ResponseMessage(text=f"model: {target}")


class Think(Command):
    name = "think"
    args = "[on | off | auto | show | hide]"
    description = "Control reasoning mode for this channel"
    # Reads/writes ctx.conversation.overrides["reasoning"]; no extra requires.
    requires = ()

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        mode = raw_args.strip().lower()
        conv = ctx.conversation

        if mode == "on":
            conv.overrides["reasoning"] = "on"
            yield ResponseMessage(text="Reasoning: on", command="think")

        elif mode == "off":
            conv.overrides["reasoning"] = "off"
            yield ResponseMessage(text="Reasoning: off", command="think")

        elif mode == "auto":
            conv.overrides.pop("reasoning", None)
            yield ResponseMessage(text="Reasoning: auto (off by default)", command="think")

        elif mode in ("show", "hide"):
            # show/hide is CLI display preference only; Discord rendering not yet supported
            yield ResponseMessage(
                text=(
                    f"Reasoning display: {mode} "
                    "(CLI only -- Discord reasoning display is not yet available)"
                ),
                command="think",
            )

        elif mode == "":
            current = conv.overrides.get("reasoning") or "auto"
            yield ResponseMessage(
                text=f"Reasoning: {current}",
                command="think",
            )

        else:
            yield ResponseMessage(
                text="Usage: /think [on|off|auto|show|hide]",
                command="think",
            )


class Quit(Command):
    name = "quit"
    args = ""
    description = "End the session"

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        yield ResponseMessage(text="Goodbye.")


class Context(Command):
    name = "context"
    args = ""
    description = (
        "Show the channel's context budget: last-turn token usage and current "
        "byte sizes of the system prompt, tool schemas, conversation history, "
        "and scratchpad."
    )
    requires = ("config",)

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        import json

        agent = ctx.agent

        # Component sizes from current state (what the next request would send).
        # Bytes are cheap; tokens would need a tokenizer dep we don't carry.
        try:
            sys_prompt = agent.system_prompt(ctx)
        except Exception as exc:
            sys_prompt = ""
            sys_err = f" (error building: {exc})"
        else:
            sys_err = ""
        sys_bytes = len(sys_prompt.encode("utf-8"))

        executor = getattr(agent, "tool_executor", None)
        if executor is not None:
            schemas = executor.schemas()
            schema_bytes = len(json.dumps(schemas).encode("utf-8"))
            schema_count = len(schemas)
        else:
            schema_bytes = 0
            schema_count = 0

        history = getattr(ctx.conversation, "messages", []) or []
        history_bytes = len(json.dumps(history).encode("utf-8"))
        history_turns = len(history)

        # Scratchpad: read live since the agent may not always have a cached one.
        scratchpad_bytes = 0
        try:
            from agent_core.scratchpad import Scratchpad
            sp = Scratchpad(
                vault_path=agent.config.vault_path,
                agent_name=agent.name,
                channel_id=ctx.channel_id,
                max_bytes=getattr(agent.config, "scratchpad_max_bytes", 0),
            )
            scratchpad_bytes = len(sp.read().encode("utf-8"))
        except Exception:
            pass

        request_bytes = sys_bytes + schema_bytes + history_bytes + scratchpad_bytes

        # Last-turn actual usage (ground truth, if recorded).
        last_usage_dict = getattr(agent, "last_usage", {}) or {}
        usage = last_usage_dict.get(ctx.channel_id)

        lines = [f"Channel: {ctx.channel_id}"]
        if usage is not None:
            lines.append(
                f"Last turn: {usage.prompt_tokens} prompt + "
                f"{usage.completion_tokens} completion = {usage.total_tokens} tokens"
                + (f" ({usage.model})" if usage.model else "")
            )
        else:
            lines.append("Last turn: no usage recorded yet for this channel")

        lines.append("")
        lines.append("Current component sizes (bytes):")
        lines.append(f"  System prompt:  {sys_bytes:>8}{sys_err}")
        lines.append(f"  Tool schemas:   {schema_bytes:>8}  ({schema_count} tools)")
        lines.append(f"  History:        {history_bytes:>8}  ({history_turns} turns)")
        lines.append(f"  Scratchpad:     {scratchpad_bytes:>8}")
        lines.append(f"  Approx total:   {request_bytes:>8}")

        yield ResponseMessage(text="\n".join(lines))
