from __future__ import annotations

import json


def _clip(s: str, n: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= n:
        return s
    return b[:n].decode("utf-8", "ignore")


def build_stub(*, worker: str, ref: str, rows: int, summary: str, body_bytes: int,
               cols: list[str], max_bytes: int = 512) -> str:
    shown = cols[:3]
    if len(cols) > 3:
        shown = shown + [f"+{len(cols) - 3} more"]
    doc = {
        "summary": _clip(summary, 160),
        "captured": {
            "worker": worker,
            "ref": ref,
            "rows": rows,
            "columns": shown,
            "shape": f"{rows} row(s); body {body_bytes}B (elided)",
        },
        "hint": f'read_capture(ref="{ref}")',
    }
    blob = json.dumps(doc)
    if len(blob.encode("utf-8")) <= max_bytes:
        return blob
    # Fallback: drop columns, then clip summary harder, guaranteeing the bound.
    doc["captured"]["columns"] = [f"{len(cols)} cols"]
    doc["summary"] = _clip(summary, 60)
    blob = json.dumps(doc)
    if len(blob.encode("utf-8")) <= max_bytes:
        return blob
    return json.dumps({"captured": {"worker": worker, "ref": ref, "rows": rows},
                       "hint": f'read_capture(ref="{ref}")'})
