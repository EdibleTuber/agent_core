"""SystemPromptBuilder: composable system-prompt section helpers.

The builder provides `render_*` methods, one per standard section. Agents
assemble their full system prompt by calling whichever they want in
whichever order from `Agent.system_prompt(ctx)`. Each render method returns
an empty string when its data is empty, so consumers can `filter(None, ...)`
freely.

The builder takes the agent itself in addition to the framework managers so
it can construct per-channel Scratchpads on demand (Scratchpad needs
vault_path + agent_name + channel_id + max_bytes from the agent's config).
"""
from __future__ import annotations

from agent_core.scratchpad import Scratchpad


class SystemPromptBuilder:
    def __init__(
        self,
        profile,                # ProfileManager
        wisdom,                 # WisdomManager
        channels,               # ChannelStore (kept for symmetry; current
                                # implementation reads scratchpads directly)
        tool_executor,          # ToolExecutor
        command_registry,       # CommandRegistry
        agent,                  # Agent (for config + name when constructing
                                # per-channel Scratchpads)
    ) -> None:
        self.profile = profile
        self.wisdom = wisdom
        self.channels = channels
        self.tool_executor = tool_executor
        self.command_registry = command_registry
        self.agent = agent

    def render_profile(self) -> str:
        body = self.profile.read()
        return f"## About the User\n\n{body}" if body else ""

    def render_wisdom(self) -> str:
        bodies = self.wisdom.bodies()
        if not bodies:
            return ""
        text = "\n".join(f"- {b}" for b in bodies)
        return f"## Active Wisdom\n\n{text}"

    def render_scratchpad(self, channel_id: str) -> str:
        cfg = self.agent.config
        sp = Scratchpad(
            vault_path=cfg.vault_path,
            agent_name=self.agent.name,
            channel_id=channel_id,
            max_bytes=cfg.scratchpad_max_bytes,
        )
        body = sp.read()
        return f"## Channel Scratchpad\n\n{body}" if body else ""

    def render_commands_catalog(self) -> str:
        meta = self.command_registry.metadata()
        if not meta:
            return ""
        lines = []
        for name, args, desc in meta:
            entry = f"- `/{name}"
            if args:
                entry += f" {args}"
            entry += f"` - {desc}"
            lines.append(entry)
        return "## Available Commands\n\n" + "\n".join(lines)

    def render_tools_catalog(self) -> str:
        schemas = self.tool_executor.schemas()
        if not schemas:
            return ""
        lines = []
        for s in schemas:
            f = s["function"]
            lines.append(f"- `{f['name']}` - {f['description']}")
        return "## Available Tools\n\n" + "\n".join(lines)
