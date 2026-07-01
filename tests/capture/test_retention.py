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
