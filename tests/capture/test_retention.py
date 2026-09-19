import json
from pathlib import Path
from agent_core.capture.store import CaptureStore, CaptureRecord


def _big(worker="frida"):
    return CaptureRecord(worker=worker, tool="read_memory", session_id="s1", launch_ts=1.0,
                         summary="big", body=json.dumps([{"hex": "ab" * 50000}]), rows=1, addrs=[])


async def test_delete_removes_row_and_blob(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    ref = await store.write(_big())
    await store.get(ref)  # Ensure write completes before checking blob file
    blob = next((tmp_path / ".pare" / "blobs").glob("*.bin"))
    assert store.delete(ref) is True
    assert await store.get(ref) is None
    assert not blob.exists()
    store.close()


async def test_purge_by_age_respects_protected_refs(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    old = await store.write(_big())
    keep = await store.write(_big())
    # Ensure writes complete before purging
    await store.get(old)
    await store.get(keep)
    # Age everything far into the past; protect `keep`.
    removed = store.purge(max_age_s=0.0, now=1e12, protected_refs={keep})
    assert removed == 1
    assert await store.get(old) is None
    assert await store.get(keep) is not None
    store.close()


async def test_purge_by_size_skips_protected_oldest(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    protected = await store.write(_big())   # seq 1, oldest
    evictable = await store.write(_big())   # seq 2
    # Ensure writes complete before purging
    await store.get(protected)
    await store.get(evictable)
    removed = store.purge(max_bytes=0, now=0, protected_refs={protected})
    assert removed == 1
    assert await store.get(protected) is not None
    assert await store.get(evictable) is None
    store.close()


async def test_delete_unknown_ref_returns_false(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    assert store.delete("nonexistent") is False
    store.close()


async def test_text_search_survives_delete_of_other_spilled_row(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    # A spilled (>64KB) row carrying a known searchable token.
    keep = await store.write(CaptureRecord(
        worker="frida", tool="read_memory", session_id="s1", launch_ts=1.0,
        summary="big", body=json.dumps([{"tag": "UNIQUETOKEN_KEEP", "pad": "a" * 70000}]),
        rows=1, addrs=[]))
    other = await store.write(_big())  # unrelated spilled row
    # Ensure writes complete before searching
    await store.get(keep)
    await store.get(other)
    assert len(store.search(text="UNIQUETOKEN_KEEP")) == 1
    store.delete(other)
    # The rebuild bug would drop the spilled row's tokens here -> 0 hits.
    assert len(store.search(text="UNIQUETOKEN_KEEP")) == 1
    store.close()
