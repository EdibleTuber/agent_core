import json
from agent_core.capture.stub import build_stub


def test_stub_is_hard_bounded_and_shows_shape_not_content():
    stub = build_stub(worker="frida", ref="a1b2c3", rows=1, summary="read_memory: 65536 bytes @ 0x401000",
                      body_bytes=65536, cols=["address", "size", "hex", "extra1", "extra2", "extra3"])
    assert len(stub.encode("utf-8")) <= 512
    doc = json.loads(stub)
    assert doc["captured"]["ref"] == "a1b2c3"
    assert 'read_capture(ref="a1b2c3")' in doc["hint"]
    # No raw payload re-inlined: the 65536-byte body must not appear.
    assert "hex" not in json.dumps(doc.get("captured", {}).get("preview", ""))
    assert "+3 more" in " ".join(doc["captured"]["columns"]) or len(doc["captured"]["columns"]) <= 4


def test_stub_never_exceeds_budget_on_long_summary():
    stub = build_stub(worker="ghidra", ref="ffff", rows=999, summary="x" * 5000,
                      body_bytes=999999, cols=["a"] * 200)
    assert len(stub.encode("utf-8")) <= 512
    json.loads(stub)  # still valid JSON
