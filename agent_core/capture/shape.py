from __future__ import annotations

import re
from typing import Any

_HEX = re.compile(r"\b(?:0x)?([0-9a-fA-F]{4,16})\b")


def infer_rows(value: Any) -> list[dict]:
    """Map a JSON value to rows per spec §4, with guard rules."""
    if isinstance(value, dict):
        # Annotated-list object -> unwrap to that list's rows. Applies when the
        # object has exactly one list-valued value and every remaining value is
        # a scalar (e.g. a worker envelope {"summary": "2 processes",
        # "processes": [...]}): the list is the rows, the scalars are
        # annotations. A dict-valued sibling means this is a genuine multi-field
        # record, so keep it as a single row.
        list_vals = [v for v in value.values() if isinstance(v, list)]
        nonlist = [v for v in value.values() if not isinstance(v, list)]
        if len(list_vals) == 1 and all(not isinstance(v, dict) for v in nonlist):
            return infer_rows(list_vals[0])
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
    if len(rows) > 1:
        # A multi-row result (incl. an unwrapped annotated-list envelope) is
        # worth storing even when small, so it can be searched / re-viewed.
        return True
    return serialized_bytes > inline_budget
