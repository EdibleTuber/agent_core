from agent_core.capture.shape import infer_rows, columns, normalize_addrs, is_substantial


def test_array_of_objects_is_n_rows():
    assert infer_rows([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


def test_object_is_one_row():
    assert infer_rows({"a": 1}) == [{"a": 1}]


def test_single_array_value_object_unwraps():
    assert infer_rows({"modules": [{"n": 1}, {"n": 2}]}) == [{"n": 1}, {"n": 2}]


def test_non_object_elements_are_wrapped():
    assert infer_rows(["a", "b"]) == [{"value": "a"}, {"value": "b"}]


def test_empty_array_is_no_rows():
    assert infer_rows([]) == []


def test_columns_are_deterministic_union_capped():
    rows = [{"b": 1, "a": 2}, {"c": 3}]
    assert columns(rows) == ["a", "b", "c"]


def test_normalize_addrs_strips_and_pads():
    got = set(normalize_addrs('{"ea": "0x401000", "p": "00401000"}'))
    assert "0000000000401000" in got


def test_is_substantial():
    assert is_substantial([{"a": 1}], [{"a": 1}], 10, 4096) is True          # array -> store
    assert is_substantial("3 devices", ["3 devices"], 10, 4096) is False     # small scalar
    assert is_substantial({"x": 1}, [{"x": 1}], 99999, 4096) is True         # over budget
