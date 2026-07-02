import json
import pytest
from agent_core.capture.store import CaptureStore
from agent_core.capture.layer import CaptureLayer, stringify_result


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Result:
    def __init__(self, text): self.isError = False; self.content = [_Block(text)]


def _layer(budget=200):
    return CaptureLayer(CaptureStore.open_memory(), inline_budget=budget, launch_ts=1.0)


def test_small_result_passes_through_unsubstituted():
    layer = _layer()
    r = _Result(json.dumps([{"a": 1}]))
    out = layer.maybe_substitute("frida", "t", r, substitute=True)
    assert out is r  # under budget -> model sees it verbatim


def test_oversized_result_is_substituted_with_bounded_stub():
    layer = _layer(budget=50)
    big = json.dumps([{"hex": "ab" * 500}])
    out = layer.maybe_substitute("frida", "read_memory", _Result(big), substitute=True)
    text = stringify_result(out)
    assert len(text.encode("utf-8")) <= 512
    doc = json.loads(text)
    ref = doc["captured"]["ref"]
    # Full body retrievable from the store the layer wrote to.
    assert json.loads(layer.store.get(ref)["body"])[0]["hex"] == "ab" * 500


def test_operator_path_stores_but_never_substitutes():
    layer = _layer(budget=10)
    big = json.dumps([{"hex": "ab" * 500}])
    r = _Result(big)
    out = layer.maybe_substitute("frida", "enumerate_processes", r, substitute=False)
    assert out is r                     # operator sees the real payload
    assert len(layer.store.recent()) == 1  # ...but it was still stored
