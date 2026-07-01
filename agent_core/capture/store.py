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
        ref = secrets.token_hex(4)
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
