import json
from agent_core.capture.store import CaptureStore, CaptureRecord
from agent_core.capture.query import fts_phrase


def _rec(worker="frida", body=None, addrs=None, summary="s"):
    return CaptureRecord(worker=worker, tool="t", session_id="s1", launch_ts=1.0,
                         summary=summary, body=body or json.dumps([{"name": "x"}]),
                         rows=1, addrs=addrs or [])


def test_fts_phrase_escapes_punctuation():
    assert fts_phrase('libc.so.6') == '"libc.so.6"'
    assert fts_phrase('say "hi"') == '"say ""hi"""'


def test_text_search_matches_dotted_token_without_crashing():
    store = CaptureStore.open_memory()
    store.write(_rec(body=json.dumps([{"name": "libc.so.6"}])))
    hits = store.search(text="libc.so.6")   # would raise fts5 syntax error if unescaped
    assert len(hits) == 1
    store.close()


def test_worker_filter_and_recent():
    store = CaptureStore.open_memory()
    store.write(_rec(worker="frida"))
    store.write(_rec(worker="ghidra"))
    assert len(store.search(worker="frida")) == 1
    assert len(store.recent(limit=10)) == 2
    store.close()


def test_addrs_are_searchable_after_normalization():
    store = CaptureStore.open_memory()
    store.write(_rec(body=json.dumps([{"ea": "0x401000"}]), addrs=["0000000000401000"]))
    assert len(store.search(text="0000000000401000")) == 1
    store.close()


def test_field_contains_on_allowlist_column():
    store = CaptureStore.open_memory()
    store.write(_rec(worker="frida"))  # tool defaults to "t" in _rec
    assert len(store.search(field="tool", contains="t")) == 1
    assert len(store.search(field="tool", contains="zzz")) == 0
    store.close()


def test_field_contains_via_json_extract_on_object_body():
    import json
    store = CaptureStore.open_memory()
    # An object body exposes top-level keys to json_extract('$.name').
    store.write(_rec(body=json.dumps({"name": "libc.so"})))
    assert len(store.search(field="name", contains="libc")) == 1
    assert len(store.search(field="name", contains="zzz")) == 0
    store.close()
