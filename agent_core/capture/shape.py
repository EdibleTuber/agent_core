from __future__ import annotations

import re
from typing import Any

_HEX = re.compile(r"\b(?:0x)?([0-9a-fA-F]{4,16})\b")


def infer_rows(value: Any) -> list[dict]:
    """Map a JSON value to rows per spec §4, with guard rules."""
    if isinstance(value, dict):
        # Single-array-value object -> unwrap to that array's rows.
        vals = list(value.values())
        if len(vals) == 1 and isinstance(vals[0], list):
            return infer_rows(vals[0])
        return [value]
    if isinstance(value, list):
        out = []
        for el in value:
            out.append(el if isinstance(el, dict) else {"value": el})
        return out
    # Scalar/blob -> degenerate one-column row.
    return [{"value": value}]


def columns(rows: list[dict], cap: int = 12) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.append(k)
    ordered = sorted(seen)
    return ordered[:cap]


def normalize_addrs(body: str) -> list[str]:
    out: set[str] = set()
    for m in _HEX.finditer(body):
        tok = m.group(1).lower()
        if len(tok) >= 4:
            out.add(tok.zfill(16))
    return sorted(out)


def is_substantial(value: Any, rows: list[dict], serialized_bytes: int, inline_budget: int) -> bool:
    if isinstance(value, list) and rows:
        return True
    return serialized_bytes > inline_budget
