"""Discriminating tests for CaptureStore's writer-thread / pending-writes design.

These exercise the async write()/get() contract added to move the sqlite
write off the event loop (see docs/superpowers/specs/2026-09-19-on-loop-capture-write-design.md
section 2 in the PARE repo). Test seam: CaptureStore.open(..., _pre_write_hook=...)
-- an optional callable invoked on the writer thread at the top of each insert
loop iteration, used here to pin the writer thread with a threading.Event.wait.
"""
import asyncio
import logging
import re
import threading
import time

import pytest

from agent_core.capture.store import CaptureStore, CaptureRecord


def _rec(**kw):
    base = dict(worker="w", tool="t", session_id=None, launch_ts=0.0,
                summary="", body="hello", rows=0, addrs=[])
    base.update(kw)
    return CaptureRecord(**base)


async def test_get_awaits_a_still_pending_write(tmp_path):
    """D4: a get() of a ref whose write has not yet committed must await
    the pending-writes future -- NOT return None (which ReadCapture.run
    renders to the model as "expired capture", a lie).

    Uses the pre-write hook to pin the writer thread inside its insert loop,
    so the test's get() runs while pending_writes[ref] is still unresolved.
    """
    release = threading.Event()  # writer thread waits here until released

    def _pin_writer():
        release.wait(timeout=5.0)

    store = CaptureStore.open(tmp_path / "cap.db", _pre_write_hook=_pin_writer)
    try:
        ref = await store.write(_rec())  # returns immediately; writer is pinned

        # The writer has NOT committed. A naive get() (SELECT only) returns
        # None. The correct get() awaits pending_writes[ref] first.
        get_task = asyncio.create_task(store.get(ref))
        # Give get_task a chance to hit the pending-writes await; then
        # release the writer.
        await asyncio.sleep(0.05)
        release.set()
        row = await get_task

        assert row is not None, "get must await the pending write, not return None"
        assert row["body"] == "hello"
    finally:
        release.set()  # in case the assertion fired before we released
        store.close()


async def test_write_then_get_after_writer_finishes(tmp_path):
    """Baseline: a write's ref is visible to get() once the writer has had
    a chance to finish (no pinning -- this is the common, non-racing case)."""
    store = CaptureStore.open(tmp_path / "cap.db")
    try:
        ref = await store.write(_rec(body="plain roundtrip"))
        await asyncio.sleep(0.05)  # let the writer thread commit
        row = await store.get(ref)
        assert row is not None
        assert row["body"] == "plain roundtrip"
    finally:
        store.close()


async def test_write_returns_before_the_write_commits(tmp_path):
    """write() must not block the caller for the duration of the sqlite
    write -- it returns as soon as the item is enqueued. Discriminates a
    regression that turned write() back into a blocking call on the
    caller's thread."""
    hold = threading.Event()
    WRITER_DELAY = 0.3

    def _slow_writer():
        hold.wait(timeout=5.0)
        time.sleep(WRITER_DELAY)

    store = CaptureStore.open(tmp_path / "cap.db", _pre_write_hook=_slow_writer)
    try:
        hold.set()  # let the writer proceed straight into its (slow) insert
        start = time.monotonic()
        await store.write(_rec())
        elapsed = time.monotonic() - start
        assert elapsed < WRITER_DELAY, (
            f"write() took {elapsed:.3f}s -- expected it to return well "
            f"before the {WRITER_DELAY}s writer delay"
        )
    finally:
        store.close()


async def test_writer_raises_then_get_raises(tmp_path, caplog):
    """spec §4.3 / §2.6 row 1+4: a writer exception must surface to a
    subsequent get(ref), and (F2) the failure must be logged regardless.

    Note on the seam: _writer_loop (store.py:120-125) catches a raising
    _pre_write_hook and only logs it -- it does NOT forward that exception
    to the write's future (the `except Exception as exc:` block that does
    is the *separate* one guarding `_insert_record`, a few lines below).
    So a hook that merely raises would make the write succeed anyway and
    this test would prove nothing. Instead, this hook is pinned exactly
    like test_get_awaits_a_still_pending_write's (so get()'s `await future`
    is provably parked on the *same* future object before the writer can
    reach its `finally: pop`, which removes the race), and on release it
    monkeypatches store._insert_record to raise a known exception -- making
    the writer's insert path itself fail, deterministically.
    """
    release = threading.Event()

    def _pin_then_sabotage():
        release.wait(timeout=5.0)
        store._insert_record = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("writer sabotage")
        )

    store = CaptureStore.open(tmp_path / "cap.db", _pre_write_hook=_pin_then_sabotage)
    caplog.set_level(logging.ERROR, logger="agent_core.capture.store")
    try:
        ref = await store.write(_rec())  # writer is pinned before it can insert

        get_task = asyncio.create_task(store.get(ref))
        # Give get_task a chance to retrieve and start awaiting the pending
        # future -- once it holds the future object, the later `pop` from
        # _pending_writes cannot make it miss the resolution.
        await asyncio.sleep(0.05)
        release.set()

        with pytest.raises(RuntimeError, match="writer sabotage"):
            await get_task

        assert any(
            rec.getMessage() == f"capture store write failed for ref={ref}"
            for rec in caplog.records
        ), "F2: a failed write must be logged even though this test awaits it"
    finally:
        release.set()
        store.close()


async def test_close_drains_with_bounded_timeout(tmp_path, caplog):
    """spec §4.4: close() must not hang past its bound when a queued write
    stalls, and must warn naming the queue depth rather than silently drop
    the still-queued writes."""
    calls = {"n": 0}

    def _stall_first_write(_calls=calls):
        _calls["n"] += 1
        if _calls["n"] == 1:
            time.sleep(6.0)  # longer than close()'s 5s join timeout

    store = CaptureStore.open(tmp_path / "cap.db", _pre_write_hook=_stall_first_write)
    caplog.set_level(logging.WARNING, logger="agent_core.capture.store")
    try:
        for _ in range(3):
            await store.write(_rec())
        # Let the writer thread dequeue and stall on the first item before
        # close() races it.
        await asyncio.sleep(0.05)

        start = time.monotonic()
        store.close()
        elapsed = time.monotonic() - start

        assert elapsed < 6.0, (
            f"close() took {elapsed:.3f}s -- expected it bounded near its 5s timeout, "
            "not the 6s+ the stalled write would take if close() waited for it"
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            re.search(r"\d+ item\(s\) still queued", rec.getMessage()) for rec in warnings
        ), "close() must warn, naming the queue depth, when it times out with items still queued"
    finally:
        pass  # store.close() already ran above; the writer thread is a
        # daemon and will drain its remaining (already-enqueued) items in
        # the background after this test returns.


async def test_open_memory_never_starts_a_writer_thread(tmp_path):
    """open_memory() must not start a writer thread: a :memory: sqlite db
    exists only on the connection that opened it, so a separate writer
    thread would silently write to a different, empty database."""
    store = CaptureStore.open_memory()
    try:
        assert store._writer_thread is None
        ref = await store.write(_rec(body="inline"))
        assert store._writer_thread is None  # still true after a write
        row = await store.get(ref)
        assert row is not None
        assert row["body"] == "inline"
    finally:
        store.close()
