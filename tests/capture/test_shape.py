from agent_core.capture.shape import infer_rows, columns, normalize_addrs, is_substantial


def test_array_of_objects_is_n_rows():
    assert infer_rows([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


def test_object_is_one_row():
    assert infer_rows({"a": 1}) == [{"a": 1}]


def test_single_array_value_object_unwraps():
    assert infer_rows({"modules": [{"n": 1}, {"n": 2}]}) == [{"n": 1}, {"n": 2}]


def test_annotated_list_envelope_unwraps_ignoring_scalar_siblings():
    # A worker envelope pairs a scalar summary with the data list; the list is
    # the rows, the summary is an annotation. (frida enumerate_* shape.)
    env = {"summary": "2 processes", "processes": [{"pid": 1}, {"pid": 9}]}
    assert infer_rows(env) == [{"pid": 1}, {"pid": 9}]


def test_object_with_structured_sibling_stays_one_row():
    # A dict sibling means this is a genuine multi-field record, not an
    # annotated list -> keep it as a single row, do not unwrap.
    obj = {"config": {"k": "v"}, "items": [{"a": 1}, {"a": 2}]}
    assert infer_rows(obj) == [obj]


def test_multi_row_result_is_substantial_even_when_small():
    # A small multi-row enumerate should still be stored so /snapshot can
    # re-view it (was previously dropped as "under budget").
    rows = infer_rows({"summary": "2 procs", "processes": [{"pid": 1}, {"pid": 9}]})
    assert is_substantial({"summary": "2 procs", "processes": [{"pid": 1}, {"pid": 9}]},
                          rows, 40, 4096) is True


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
