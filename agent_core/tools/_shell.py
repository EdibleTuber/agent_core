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


class Ls(Tool):
    name = "ls"
    description = "List files and subdirectories in a vault directory. Capped at 500 entries."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative directory path; empty for root."},
            "show_hidden": {"type": "boolean", "description": "Show _-prefixed entries (default false)."},
            "long": {"type": "boolean", "description": "Include size and mtime per entry (default false)."},
        },
        "required": [],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        from datetime import datetime, timezone

        path = (args.get("path") or "").strip()
        show_hidden = bool(args.get("show_hidden", False))
        long_fmt = bool(args.get("long", False))
        max_entries = 500

        vault = ctx.agent.config.vault_path
        if path:
            if is_system_path(path) and not show_hidden:
                return f"Error: system path is not listable without show_hidden: {path}"
            resolved = resolve_safe(vault, path)
            if resolved is None:
                return f"Error: path escapes outside vault: {path}"
        else:
            resolved = vault.resolve()
        if not resolved.exists():
            return f"Directory not found: {path or '/'}"
        if not resolved.is_dir():
            return f"Not a directory: {path or '/'}"

        try:
            entries = sorted(resolved.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return f"Error listing {path or '/'}: {exc}"

        out_lines = []
        truncated = False
        for entry in entries:
            if not show_hidden and entry.name.startswith("_"):
                continue
            if len(out_lines) >= max_entries:
                truncated = True
                break
            display = entry.name + ("/" if entry.is_dir() else "")
            if long_fmt:
                try:
                    st = entry.stat()
                    size = st.st_size
                    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    out_lines.append(f"{size:>10} {mtime} {display}")
                except OSError:
                    out_lines.append(f"         ?           ? {display}")
            else:
                out_lines.append(display)
        if truncated:
            out_lines.append(f"[output truncated: more than {max_entries} entries]")
        return cap_output("\n".join(out_lines))


class Grep(Tool):
    name = "grep"
    description = "Keyword or regex search across vault files. Returns path:lineno:line per hit, capped at 100 hits by default."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Plain string by default; regex when regex=true."},
            "path": {"type": "string", "description": "Subdir or file to search (vault-relative). Empty = vault root."},
            "regex": {"type": "boolean", "description": "Treat pattern as Python regex (default false)."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)."},
            "max_hits": {"type": "integer", "description": "Cap on number of hits (default 100)."},
        },
        "required": ["pattern"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        import re
        from pathlib import Path

        pattern_str = args.get("pattern", "")
        if not pattern_str:
            return "Error: 'pattern' parameter is required."
        path = (args.get("path") or "").strip()
        as_regex = bool(args.get("regex", False))
        ignore_case = bool(args.get("ignore_case", False))
        max_hits = max(1, min(int(args.get("max_hits", 100)), 1000))

        vault = ctx.agent.config.vault_path
        if path:
            resolved = resolve_safe(vault, path)
            if resolved is None:
                return f"Error: path escapes outside vault: {path}"
            if is_system_path(path):
                return f"Error: system path is not searchable: {path}"
        else:
            resolved = vault.resolve()
        if not resolved.exists():
            return f"Path not found: {path or '/'}"

        flags = re.IGNORECASE if ignore_case else 0
        try:
            if as_regex:
                regex = re.compile(pattern_str, flags)
            else:
                regex = re.compile(re.escape(pattern_str), flags)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"

        targets: list[Path] = []
        if resolved.is_file():
            targets = [resolved]
        else:
            for p in resolved.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(vault.resolve())
                if any(part.startswith("_") for part in rel.parts):
                    continue
                targets.append(p)

        hits = []
        for target in targets:
            if len(hits) >= max_hits:
                break
            try:
                with target.open("r", errors="replace") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            rel = target.relative_to(vault.resolve())
                            hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(hits) >= max_hits:
                                break
            except OSError:
                continue
        if not hits:
            return f"No match for: {pattern_str}"
        if len(hits) >= max_hits:
            hits.append(f"[output truncated: hit cap of {max_hits} reached]")
        return cap_output("\n".join(hits))


class Find(Tool):
    name = "find"
    description = "Filename glob search. Patterns like 'agent-*.md' or '**/quantum*'. Capped at 500 results."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Subdir to search (vault-relative). Empty = vault root."},
            "type": {"type": "string", "description": "'f' for files, 'd' for dirs, '' for both (default)."},
            "max_results": {"type": "integer", "description": "Cap on number of results (default 500)."},
        },
        "required": ["pattern"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        pattern = args.get("pattern", "")
        if not pattern:
            return "Error: 'pattern' parameter is required."
        path = (args.get("path") or "").strip()
        type_filter = (args.get("type") or "").strip()
        max_results = max(1, min(int(args.get("max_results", 500)), 5000))

        vault = ctx.agent.config.vault_path
        if path:
            resolved = resolve_safe(vault, path)
            if resolved is None:
                return f"Error: path escapes outside vault: {path}"
        else:
            resolved = vault.resolve()
        if not resolved.exists() or not resolved.is_dir():
            return f"Directory not found: {path or '/'}"

        results = []
        for p in resolved.rglob(pattern):
            rel = p.relative_to(vault.resolve())
            if any(part.startswith("_") for part in rel.parts):
                continue
            if type_filter == "f" and not p.is_file():
                continue
            if type_filter == "d" and not p.is_dir():
                continue
            results.append(str(rel))
            if len(results) >= max_results:
                break
        if not results:
            return f"No match for: {pattern}"
        if len(results) >= max_results:
            results.append(f"[output truncated: result cap of {max_results} reached]")
        return cap_output("\n".join(results))


class ReadLines(Tool):
    name = "read_lines"
    description = "Read a specific 1-indexed line range from a vault file. Pairs with grep hits."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Vault-relative file path."},
            "start": {"type": "integer", "description": "Starting line number (1-indexed, inclusive)."},
            "end": {"type": "integer", "description": "Ending line number (1-indexed, inclusive)."},
        },
        "required": ["path", "start", "end"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        resolved, err = _read_safe(args, ctx.agent.config.vault_path)
        if err is not None:
            return err
        try:
            start = int(args["start"])
            end = int(args["end"])
        except (KeyError, TypeError, ValueError):
            return "Error: 'start' and 'end' integers are required."
        if start < 1 or end < start:
            return f"Error: invalid line range: start={start}, end={end}."
        out = []
        with resolved.open("r", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if lineno < start:
                    continue
                if lineno > end:
                    break
                out.append(f"{lineno}: {line.rstrip()}")
        if not out:
            return f"No lines in range {start}..{end}."
        return cap_output("\n".join(out))
