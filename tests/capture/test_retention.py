import json
from pathlib import Path
from agent_core.capture.store import CaptureStore, CaptureRecord


def _big(worker="frida"):
    return CaptureRecord(worker=worker, tool="read_memory", session_id="s1", launch_ts=1.0,
                         summary="big", body=json.dumps([{"hex": "ab" * 50000}]), rows=1, addrs=[])


def test_delete_removes_row_and_blob(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    ref = store.write(_big())
    blob = next((tmp_path / ".pare" / "blobs").glob("*.bin"))
    assert store.delete(ref) is True
    assert store.get(ref) is None
    assert not blob.exists()
    store.close()


def test_purge_by_age_respects_protected_refs(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    old = store.write(_big())
    keep = store.write(_big())
    # Age everything far into the past; protect `keep`.
    removed = store.purge(max_age_s=0.0, now=1e12, protected_refs={keep})
    assert removed == 1
    assert store.get(old) is None
    assert store.get(keep) is not None
    store.close()


def test_purge_by_size_skips_protected_oldest(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    protected = store.write(_big())   # seq 1, oldest
    evictable = store.write(_big())   # seq 2
    removed = store.purge(max_bytes=0, now=0, protected_refs={protected})
    assert removed == 1
    assert store.get(protected) is not None
    assert store.get(evictable) is None
    store.close()


def test_delete_unknown_ref_returns_false(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    assert store.delete("nonexistent") is False
    store.close()


def test_text_search_survives_delete_of_other_spilled_row(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    # A spilled (>64KB) row carrying a known searchable token.
    keep = store.write(CaptureRecord(
        worker="frida", tool="read_memory", session_id="s1", launch_ts=1.0,
        summary="big", body=json.dumps([{"tag": "UNIQUETOKEN_KEEP", "pad": "a" * 70000}]),
        rows=1, addrs=[]))
    other = store.write(_big())  # unrelated spilled row
    assert len(store.search(text="UNIQUETOKEN_KEEP")) == 1
    store.delete(other)
    # The rebuild bug would drop the spilled row's tokens here -> 0 hits.
    assert len(store.search(text="UNIQUETOKEN_KEEP")) == 1
    store.close()
