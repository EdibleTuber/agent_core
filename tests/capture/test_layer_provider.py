import json
from agent_core.capture.store import CaptureStore
from agent_core.capture.layer import CaptureLayer, stringify_result


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Result:
    def __init__(self, text): self.isError = False; self.content = [_Block(text)]


def test_provider_selects_store_dynamically():
    a = CaptureStore.open_memory()
    b = CaptureStore.open_memory()
    current = {"store": a}
    layer = CaptureLayer(inline_budget=10, launch_ts=1.0,
                         store_provider=lambda: current["store"])
    layer.maybe_substitute("frida", "t", _Result(json.dumps([{"x": 1}])), substitute=False)
    assert len(a.recent()) == 1 and len(b.recent()) == 0
    current["store"] = b
    layer.maybe_substitute("frida", "t", _Result(json.dumps([{"y": 2}])), substitute=False)
    assert len(b.recent()) == 1


def test_none_store_passes_through_without_capture():
    layer = CaptureLayer(inline_budget=10, launch_ts=1.0, store_provider=lambda: None)
    r = _Result(json.dumps([{"x": 1}]))
    assert layer.maybe_substitute("frida", "t", r, substitute=True) is r


def test_positional_store_still_works():
    store = CaptureStore.open_memory()
    layer = CaptureLayer(store, inline_budget=10, launch_ts=1.0)
    layer.maybe_substitute("frida", "t", _Result(json.dumps([{"x": 1}])), substitute=False)
    assert len(store.recent()) == 1
