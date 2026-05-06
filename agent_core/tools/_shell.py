"""Read-only shell-style builtin tools, scoped to the agent's vault."""
from __future__ import annotations

from agent_core.tools.base import Tool
from agent_core.tools._shell_helpers import cap_output, is_system_path, resolve_safe


class Cat(Tool):
    name = "cat"
    description = "Read the full contents of a vault file. For files larger than 32 KB the output is truncated; use head/tail/read_lines to slice."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative file path."},
        },
        "required": ["path"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        path = (args.get("path") or "").strip()
        if not path:
            return "Error: 'path' parameter is required."
        if is_system_path(path):
            return f"Error: system path is not readable: {path}"
        resolved = resolve_safe(ctx.agent.config.vault_path, path)
        if resolved is None:
            return f"Error: path escapes outside vault: {path}"
        if not resolved.exists():
            return f"File not found: {path}"
        if not resolved.is_file():
            return f"Not a file: {path}"
        try:
            content = resolved.read_text(errors="replace")
        except OSError as exc:
            return f"Error reading {path}: {exc}"
        return cap_output(content)
