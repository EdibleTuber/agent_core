import json
from agent_core.capture.store import CaptureStore, CaptureRecord


def _rec(**kw):
    base = dict(worker="frida", tool="enumerate_modules", session_id="s1",
                launch_ts=1000.0, summary="2 modules", body=json.dumps([{"name": "libc"}]),
                rows=1, addrs=[])
    base.update(kw)
    return CaptureRecord(**base)


def test_write_returns_ref_and_get_roundtrips():
    store = CaptureStore.open_memory()
    ref = store.write(_rec())
    assert isinstance(ref, str) and len(ref) >= 6
    row = store.get(ref)
    assert row["worker"] == "frida"
    assert json.loads(row["body"]) == [{"name": "libc"}]
    assert row["rows"] == 1
    store.close()


def test_refs_are_unique():
    store = CaptureStore.open_memory()
    refs = {store.write(_rec()) for _ in range(50)}
    assert len(refs) == 50
    store.close()
