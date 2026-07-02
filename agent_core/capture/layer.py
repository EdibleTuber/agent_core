from __future__ import annotations

import json
from typing import Any

from agent_core.capture.store import CaptureStore, CaptureRecord
from agent_core.capture.shape import infer_rows, columns, normalize_addrs, is_substantial
from agent_core.capture.stub import build_stub


class _TextResult:
    """CallToolResult-shaped stand-in carrying the substituted stub text."""
    def __init__(self, text: str) -> None:
        self.isError = False

        class _Block:
            type = "text"
        b = _Block()
        b.text = text
        self.content = [b]


def stringify_result(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


class CaptureLayer:
    def __init__(self, store: CaptureStore, *, inline_budget: int, launch_ts: float) -> None:
        self.store = store
        self._budget = inline_budget
        self._launch_ts = launch_ts

    def maybe_substitute(self, worker: str, tool: str, result: Any, *, substitute: bool,
                         session_id: str | None = None) -> Any:
        if getattr(result, "isError", False):
            return result
        text = stringify_result(result)
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            value = text  # opaque blob -> degenerate row
        rows = infer_rows(value)
        body_bytes = len(text.encode("utf-8"))
        if not is_substantial(value, rows, body_bytes, self._budget):
            return result
        ref = self.store.write(CaptureRecord(
            worker=worker, tool=tool, session_id=session_id, launch_ts=self._launch_ts,
            summary=f"{tool}: {len(rows)} row(s)", body=text if isinstance(text, str) else json.dumps(value),
            rows=len(rows), addrs=normalize_addrs(text),
        ))
        if not substitute or body_bytes <= self._budget:
            return result  # stored, but the caller sees the real payload
        stub = build_stub(worker=worker, ref=ref, rows=len(rows),
                          summary=f"{tool}: {len(rows)} row(s)", body_bytes=body_bytes,
                          cols=columns(rows))
        return _TextResult(stub)
