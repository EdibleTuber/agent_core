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


def _read_safe(args, vault_path):
    """Resolve and validate a path arg; return (resolved, error_str_or_none)."""
    path = (args.get("path") or "").strip()
    if not path:
        return None, "Error: 'path' parameter is required."
    if is_system_path(path):
        return None, f"Error: system path is not readable: {path}"
    resolved = resolve_safe(vault_path, path)
    if resolved is None:
        return None, f"Error: path escapes outside vault: {path}"
    if not resolved.exists():
        return None, f"File not found: {path}"
    if not resolved.is_file():
        return None, f"Not a file: {path}"
    return resolved, None


class Head(Tool):
    name = "head"
    description = "Read the first N lines of a vault file (default 20)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative file path."},
            "lines": {"type": "integer", "description": "Number of lines (default 20)."},
        },
        "required": ["path"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        resolved, err = _read_safe(args, ctx.agent.config.vault_path)
        if err is not None:
            return err
        n = max(1, int(args.get("lines", 20)))
        out = []
        with resolved.open("r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append(line.rstrip("\n"))
        return cap_output("\n".join(out))


class Tail(Tool):
    name = "tail"
    description = "Read the last N lines of a vault file (default 20)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative file path."},
            "lines": {"type": "integer", "description": "Number of lines (default 20)."},
        },
        "required": ["path"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        from collections import deque
        resolved, err = _read_safe(args, ctx.agent.config.vault_path)
        if err is not None:
            return err
        n = max(1, int(args.get("lines", 20)))
        with resolved.open("r", errors="replace") as f:
            tail = deque(f, maxlen=n)
        return cap_output("\n".join(line.rstrip("\n") for line in tail))
