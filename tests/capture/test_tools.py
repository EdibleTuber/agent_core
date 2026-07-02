import json
import pytest
from agent_core.capture.store import CaptureStore, CaptureRecord
from agent_core.capture.tools import SearchCapture, ReadCapture


class _Agent:
    def __init__(self, store): self.capture_store = store


class _Ctx:
    def __init__(self, agent): self.agent = agent


def _store_with_row():
    store = CaptureStore.open_memory()
    ref = store.write(CaptureRecord(worker="frida", tool="read_memory", session_id="s1",
                                    launch_ts=1.0, summary="big", body=json.dumps([{"hex": "abcd"}]),
                                    rows=1, addrs=[]))
    return store, ref


def test_requires_capture_store():
    assert SearchCapture.requires == ("capture_store",)
    assert ReadCapture.requires == ("capture_store",)


@pytest.mark.asyncio
async def test_read_capture_returns_body():
    store, ref = _store_with_row()
    out = await ReadCapture().run({"ref": ref}, _Ctx(_Agent(store)))
    assert "abcd" in out


@pytest.mark.asyncio
async def test_read_capture_dead_ref_is_sentinel_not_exception():
    store, _ = _store_with_row()
    out = await ReadCapture().run({"ref": "deadbeef"}, _Ctx(_Agent(store)))
    doc = json.loads(out)
    assert doc["expired"] is True
    assert "search_capture" in doc["hint"]


@pytest.mark.asyncio
async def test_search_capture_recent_mode_on_empty_args():
    store, ref = _store_with_row()
    out = await SearchCapture().run({}, _Ctx(_Agent(store)))
    doc = json.loads(out)
    assert any(r["ref"] == ref for r in doc["recent"])


@pytest.mark.asyncio
async def test_read_capture_absent_ref_returns_sentinel():
    store, _ = _store_with_row()
    out = await ReadCapture().run({}, _Ctx(_Agent(store)))
    doc = json.loads(out)
    assert doc["expired"] is True
