from __future__ import annotations

import json
from typing import Any

from agent_core.tools.base import Tool


class SearchCapture(Tool):
    name = "search_capture"
    description = (
        "Search captured tool results. Captures persist on disk; find them by "
        "SEARCHING, do not rely on remembering a ref. Call with no args for the "
        "most recent captures. worker is optional (defaults to all)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "full-text query"},
            "worker": {"type": "string", "description": "optional worker filter (frida/ghidra/...)"},
            "field": {"type": "string", "description": "dotted json path to filter on; matches object-shaped captures — for array/rows captures use text= full-text search instead"},
            "contains": {"type": "string", "description": "substring the field must contain"},
            "limit": {"type": "integer", "description": "max results (default 50; recent mode caps at 20)"},
        },
    }
    requires = ("capture_store",)

    async def run(self, args: dict, ctx: Any) -> str:
        store = ctx.agent.capture_store
        text = args.get("text", "") or ""
        worker = args.get("worker", "") or ""
        field = args.get("field", "") or ""
        contains = args.get("contains", "") or ""
        limit = int(args.get("limit") or 50)
        if not any([text, worker, field, contains]):
            return json.dumps({"recent": store.recent(limit=min(limit, 20))})
        hits = store.search(text=text, worker=worker, field=field, contains=contains, limit=limit)
        lean = [{"ref": h["ref"], "worker": h["worker"], "tool": h["tool"],
                 "rows": h["rows"], "summary": h["summary"]} for h in hits]
        return json.dumps({"matches": lean, "returned": len(lean)})


class ReadCapture(Tool):
    name = "read_capture"
    description = (
        "Read one captured result by its ref (from a search_capture match or a "
        "captured-result stub). Returns a byte window; use offset to page."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
            "offset": {"type": "integer"},
            "byte_budget": {"type": "integer"},
        },
        "required": ["ref"],
    }
    requires = ("capture_store",)

    async def run(self, args: dict, ctx: Any) -> str:
        store = ctx.agent.capture_store
        ref = args.get("ref") or ""
        row = store.get(ref)
        if row is None:
            return json.dumps({"expired": True,
                               "hint": "capture expired or unknown ref; use search_capture to find current data"})
        offset = int(args.get("offset") or 0)
        budget = int(args.get("byte_budget") or 3072)
        body = row["body"] or ""
        window = body[offset:offset + budget]
        next_offset = offset + len(window)
        return json.dumps({
            "ref": ref, "worker": row["worker"], "rows": row["rows"],
            "offset": offset, "next_offset": next_offset,
            "truncated": next_offset < len(body), "text": window,
        })
