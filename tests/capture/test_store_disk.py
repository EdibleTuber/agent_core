import json, stat
from pathlib import Path
from agent_core.capture.store import CaptureStore, CaptureRecord


def _big_rec():
    body = json.dumps([{"hex": "ab" * 50000}])  # ~100KB > 64KB threshold
    return CaptureRecord(worker="frida", tool="read_memory", session_id="s1",
                         launch_ts=1.0, summary="big", body=body, rows=1, addrs=[])


def test_large_body_spills_to_blob_and_get_restores_it(tmp_path):
    store = CaptureStore.open(tmp_path / ".pare" / "capture.db")
    ref = store.write(_big_rec())
    row = store.get(ref)
    assert row["blob_ref"] is not None
    assert json.loads(row["body"])[0]["hex"] == "ab" * 50000
    store.close()


def test_disk_permissions_are_hardened(tmp_path):
    db = tmp_path / ".pare" / "capture.db"
    store = CaptureStore.open(db)
    store.write(_big_rec())
    assert stat.S_IMODE((tmp_path / ".pare").stat().st_mode) == 0o700
    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    blob = next((tmp_path / ".pare" / "blobs").glob("*.bin"))
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600
    store.close()
