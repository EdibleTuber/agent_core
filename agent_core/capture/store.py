from __future__ import annotations

import asyncio
import logging
import queue
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

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

# Sentinel put on the writer queue to signal shutdown. `None` (per the spec's
# writer-thread contract) -- a real queue item is always a 4-tuple.
_SHUTDOWN = None


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
    def __init__(self, conn: sqlite3.Connection, root: Path | None, blob_threshold: int = 65536,
                 *, db_path: Path | None = None,
                 _pre_write_hook: Callable[[], None] | None = None) -> None:
        self._conn = conn
        self._root = root
        self._blob_threshold = blob_threshold
        self._db_path = db_path
        self._pre_write_hook = _pre_write_hook
        # ref -> Future completed (or failed) when that ref's write has been
        # committed by the writer thread. Private: tests may inspect this
        # structurally, production callers must not.
        self._pending_writes: dict[str, asyncio.Future] = {}
        self._writer_queue: "queue.SimpleQueue | None" = None
        self._writer_thread: threading.Thread | None = None
        # A disk-backed store gets a dedicated writer thread with its own
        # sqlite3.Connection (opened on that thread -- sqlite connections are
        # thread-affine). An in-memory store cannot: a :memory: database
        # exists only on the connection that opened it, so a writer thread
        # calling sqlite3.connect(":memory:") would get a different, empty
        # database. open_memory() passes db_path=None to opt out entirely;
        # its write() runs inline on the caller's thread instead.
        if db_path is not None:
            self._start_writer_thread()

    def _start_writer_thread(self) -> None:
        self._writer_queue = queue.SimpleQueue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="capture-store-writer", daemon=True,
        )
        self._writer_thread.start()

    @classmethod
    def open(cls, db_path: Path, *,
              _pre_write_hook: Callable[[], None] | None = None) -> "CaptureStore":
        root = Path(db_path).parent
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        conn = sqlite3.connect(db_path)
        Path(db_path).chmod(0o600)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        return cls(conn, root, db_path=Path(db_path), _pre_write_hook=_pre_write_hook)

    @classmethod
    def open_memory(cls) -> "CaptureStore":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return cls(conn, None, blob_threshold=1 << 30)

    # -- writer thread -----------------------------------------------------

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            while True:
                item = self._writer_queue.get()
                if item is _SHUTDOWN:
                    break
                ref, record, future, loop = item
                if self._pre_write_hook is not None:
                    try:
                        self._pre_write_hook()
                    except Exception:
                        logger.exception("capture store pre-write hook raised")
                try:
                    self._insert_record(conn, ref, record)
                except Exception as exc:  # noqa: BLE001 - forwarded to the write's future
                    self._resolve_future(loop, future, exc)
                else:
                    self._resolve_future(loop, future, None)
                finally:
                    # After the future is set (or its resolution scheduled),
                    # never before -- a get() racing with completion must not
                    # see a stale pending-writes entry once the row (or the
                    # failure) is final.
                    self._pending_writes.pop(ref, None)
        finally:
            conn.close()

    @staticmethod
    def _resolve_future(loop: asyncio.AbstractEventLoop, future: "asyncio.Future",
                         exc: Exception | None) -> None:
        try:
            if exc is None:
                loop.call_soon_threadsafe(future.set_result, None)
            else:
                loop.call_soon_threadsafe(future.set_exception, exc)
        except RuntimeError:
            # The event loop is already closed (e.g. shutdown raced the
            # writer thread). Nothing left to notify; the row itself is
            # already committed (or the exception already logged by the
            # caller path), so this is not a silent data loss.
            logger.warning("capture store could not notify a completed write: "
                            "event loop already closed")

    def _insert_record(self, conn: sqlite3.Connection, ref: str, record: CaptureRecord) -> None:
        addrs_text = " ".join(record.addrs)
        spill = len(record.body) > self._blob_threshold and self._root is not None
        stored_body = None if spill else record.body
        cur = conn.execute(
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
            conn.execute("UPDATE captures SET blob_ref=? WHERE seq=?", (blob_ref, seq))
        # FTS always gets the full body so search works on spilled rows.
        conn.execute(
            "INSERT INTO captures_fts (rowid, body, addrs) VALUES (?,?,?)",
            (seq, record.body, addrs_text),
        )
        conn.commit()

    # -- public API ----------------------------------------------------------

    async def write(self, record: CaptureRecord) -> str:
        # Ref generation is cheap and does no I/O -- stays on the caller's
        # thread/loop, not the writer thread.
        ref = secrets.token_hex(8)
        if self._writer_thread is None:
            # open_memory(): no writer thread exists (a :memory: db is only
            # visible on the connection that opened it). Run the insert body
            # inline on the caller's thread; no pending-writes bookkeeping is
            # needed since nothing else will race this ref.
            self._insert_record(self._conn, ref, record)
            return ref
        if not self._writer_thread.is_alive():
            raise RuntimeError("capture store writer thread is not running")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_writes[ref] = future
        self._writer_queue.put((ref, record, future, loop))
        return ref

    async def get(self, ref: str) -> dict[str, Any] | None:
        future = self._pending_writes.get(ref)
        if future is not None:
            await future
        return self._get_sync(ref)

    def _get_sync(self, ref: str) -> dict[str, Any] | None:
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
        if self._writer_thread is not None:
            self._writer_queue.put(_SHUTDOWN)
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning(
                    "capture store writer thread did not shut down within 5s; "
                    "%d item(s) still queued", self._writer_queue.qsize(),
                )
        self._conn.close()

    def _unlink_blob(self, blob_ref: str | None) -> None:
        if blob_ref:
            Path(blob_ref).unlink(missing_ok=True)

    def delete(self, ref: str) -> bool:
        # Reads directly rather than through get(): delete() is sync and
        # operates on retention/purge paths over already-committed rows, so
        # awaiting a pending write here is out of scope for this task.
        full = self._get_sync(ref)  # restores body from blob if spilled; carries seq/addrs/blob_ref
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
