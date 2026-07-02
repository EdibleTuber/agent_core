from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
  seq INTEGER PRIMARY KEY,
  ref TEXT NOT NULL UNIQUE,
  ts REAL NOT NULL,
  worker TEXT NOT NULL,
  tool TEXT,
  session_id TEXT,
  launch_ts REAL,
  rows INTEGER,
  summary TEXT,
  body TEXT,
  blob_ref TEXT,
  addrs TEXT
);
CREATE INDEX IF NOT EXISTS idx_captures_worker ON captures(worker);
CREATE INDEX IF NOT EXISTS idx_captures_launch ON captures(launch_ts);
CREATE INDEX IF NOT EXISTS idx_captures_ts ON captures(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS captures_fts
  USING fts5(body, addrs, content='captures', content_rowid='seq');
"""


@dataclass
class CaptureRecord:
    worker: str
    tool: str
    session_id: str | None
    launch_ts: float
    summary: str
    body: str
    rows: int
    addrs: list[str]


class CaptureStore:
    def __init__(self, conn: sqlite3.Connection, root: Path | None, blob_threshold: int = 65536):
        self._conn = conn
        self._root = root
        self._blob_threshold = blob_threshold

    @classmethod
    def open(cls, db_path: Path) -> "CaptureStore":
        root = Path(db_path).parent
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        conn = sqlite3.connect(db_path)
        Path(db_path).chmod(0o600)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        return cls(conn, root)

    @classmethod
    def open_memory(cls) -> "CaptureStore":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return cls(conn, None, blob_threshold=1 << 30)

    def write(self, record: CaptureRecord) -> str:
        ref = secrets.token_hex(8)
        addrs_text = " ".join(record.addrs)
        spill = len(record.body) > self._blob_threshold and self._root is not None
        stored_body = None if spill else record.body
        cur = self._conn.execute(
            "INSERT INTO captures (ref, ts, worker, tool, session_id, launch_ts, rows, summary, body, blob_ref, addrs)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ref, time.time(), record.worker, record.tool, record.session_id,
             record.launch_ts, record.rows, record.summary, stored_body, None, addrs_text),
        )
        seq = cur.lastrowid
        blob_ref = None
        if spill:
            blobs = self._root / "blobs"
            blobs.mkdir(exist_ok=True, mode=0o700)
            blobs.chmod(0o700)
            blob_path = blobs / f"{seq}.bin"
            try:
                blob_path.write_bytes(record.body.encode("utf-8"))
                blob_path.chmod(0o600)
            except OSError:
                blob_path.unlink(missing_ok=True)
                raise
            blob_ref = str(blob_path)
            self._conn.execute("UPDATE captures SET blob_ref=? WHERE seq=?", (blob_ref, seq))
        # FTS always gets the full body so search works on spilled rows.
        self._conn.execute(
            "INSERT INTO captures_fts (rowid, body, addrs) VALUES (?,?,?)",
            (seq, record.body, addrs_text),
        )
        self._conn.commit()
        return ref

    def get(self, ref: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM captures WHERE ref=?", (ref,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("body") is None and d.get("blob_ref"):
            d["body"] = Path(d["blob_ref"]).read_text(encoding="utf-8")
        return d

    def search(self, *, text: str = "", worker: str = "", field: str = "",
               contains: str = "", limit: int = 50) -> list[dict]:
        from agent_core.capture.query import fts_phrase, _ALLOWED_FIELDS, _COL_MAP
        clauses, params = [], []
        sql = "SELECT c.* FROM captures c"
        if text:
            sql += " JOIN captures_fts f ON f.rowid = c.seq"
            clauses.append("captures_fts MATCH ?")
            params.append(fts_phrase(text))
        if worker:
            clauses.append("c.worker = ?")
            params.append(worker)
        if field and contains:
            if field in _ALLOWED_FIELDS:
                clauses.append(f"{_COL_MAP[field]} LIKE ? ESCAPE '\\'")
            else:
                clauses.append("json_extract(c.body, ?) LIKE ? ESCAPE '\\'")
                params.append("$." + field)
            like = "%" + contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params.append(like)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY c.seq DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ref, worker, tool, rows, summary FROM captures ORDER BY seq DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def _unlink_blob(self, blob_ref: str | None) -> None:
        if blob_ref:
            Path(blob_ref).unlink(missing_ok=True)

    def delete(self, ref: str) -> bool:
        full = self.get(ref)  # restores body from blob if spilled; carries seq/addrs/blob_ref
        if full is None:
            return False
        seq = full["seq"]
        # External-content FTS5: remove this row's tokens using its ORIGINAL indexed
        # values. 'rebuild' would re-read the content table, where spilled rows have
        # body=NULL, silently dropping their tokens — breaking text search on exactly
        # the large captures spill was meant to preserve.
        self._conn.execute(
            "INSERT INTO captures_fts(captures_fts, rowid, body, addrs) VALUES('delete', ?, ?, ?)",
            (seq, full["body"] or "", full["addrs"] or ""),
        )
        self._conn.execute("DELETE FROM captures WHERE seq=?", (seq,))
        self._conn.commit()
        self._unlink_blob(full["blob_ref"])
        return True

    def total_bytes(self) -> int:
        rows = self._conn.execute(
            "SELECT COALESCE(SUM(LENGTH(body)), 0) AS b FROM captures"
        ).fetchone()["b"]
        blob_total = 0
        for r in self._conn.execute("SELECT blob_ref FROM captures WHERE blob_ref IS NOT NULL"):
            p = Path(r["blob_ref"])
            if p.exists():
                blob_total += p.stat().st_size
        return int(rows) + blob_total

    def purge(self, *, max_bytes: int | None = None, max_age_s: float | None = None,
              now: float, protected_refs: set[str] = frozenset()) -> int:
        removed = 0
        if max_age_s is not None:
            cutoff = now - max_age_s
            stale = [r["ref"] for r in self._conn.execute(
                "SELECT ref FROM captures WHERE ts < ? ORDER BY seq ASC", (cutoff,))]
            for ref in stale:
                if ref not in protected_refs and self.delete(ref):
                    removed += 1
        if max_bytes is not None:
            while self.total_bytes() > max_bytes:
                if protected_refs:
                    placeholders = ",".join("?" * len(protected_refs))
                    row = self._conn.execute(
                        f"SELECT ref FROM captures WHERE ref NOT IN ({placeholders})"
                        " ORDER BY seq ASC LIMIT 1",
                        list(protected_refs),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT ref FROM captures ORDER BY seq ASC LIMIT 1"
                    ).fetchone()
                if row is None:
                    break
                if not self.delete(row["ref"]):
                    break
                removed += 1
        return removed
