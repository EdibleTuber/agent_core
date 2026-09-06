import pytest

from agent_core.workers.artifacts import (PRODUCES_ARTIFACT, PRODUCES_META_KEY,
                                          PRODUCES_RESULT, VALID_PRODUCES)


def test_the_daemon_states_the_wire_constant_itself():
    assert PRODUCES_META_KEY == "agent_core/produces"
    assert VALID_PRODUCES == ("result", "artifact")
    assert (PRODUCES_RESULT, PRODUCES_ARTIFACT) == ("result", "artifact")


def test_it_agrees_with_the_worker_kit():
    """The two packages are installed separately, usually on different
    machines. This test is what keeps the two statements the same, in
    whichever environment happens to have both."""
    kit = pytest.importorskip(
        "pare_worker_kit.artifacts",
        reason="pare-worker-kit is not installed here; the worker-side half "
               "of this check runs in that package's own suite")
    assert kit.PRODUCES_META_KEY == PRODUCES_META_KEY
    assert kit.VALID_PRODUCES == VALID_PRODUCES


# Task 3: Descriptor validation tests
from agent_core.workers.artifacts import DescriptorError, validate_descriptor

_GOOD = {
    "host": "pare-bench",
    "path": "/mnt/bench-store/router-b/fw-0001.bin",
    "size": 2147483648,
    "sha256": "a" * 64,
    "hashed_at": 1757160000.0,
    "media_type": "application/octet-stream",
}


def test_a_well_formed_descriptor_passes_through():
    out = validate_descriptor(dict(_GOOD), worker="hardware", tool="dump_firmware")
    assert out["path"] == _GOOD["path"]
    assert out["size"] == _GOOD["size"]


@pytest.mark.parametrize("missing", ["host", "path", "size", "sha256"])
def test_a_missing_required_field_is_rejected(missing):
    """A tool that declares `artifact` and returns something else is a contract
    violation, recorded as an error rather than silently treated as a result."""
    payload = dict(_GOOD)
    del payload[missing]
    with pytest.raises(DescriptorError, match=missing):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


def test_a_non_dict_payload_is_rejected():
    with pytest.raises(DescriptorError, match="not a JSON object"):
        validate_descriptor(["nope"], worker="hardware", tool="dump_firmware")


@pytest.mark.parametrize("bad", ["", "xyz", "a" * 63, "A" * 64, "g" * 64])
def test_a_malformed_sha256_is_rejected(bad):
    """Content-addressing on retrieval is the only control that survives an
    untrusted producer, so the hash has to be a hash."""
    payload = dict(_GOOD, sha256=bad)
    with pytest.raises(DescriptorError, match="sha256"):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


@pytest.mark.parametrize("bad", [-1, "big", 1.5, None])
def test_a_non_integer_size_is_rejected(bad):
    payload = dict(_GOOD, size=bad)
    with pytest.raises(DescriptorError, match="size"):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


def test_a_relative_path_is_rejected():
    """Containment is checked against an absolute root; a relative path cannot
    be compared against one."""
    payload = dict(_GOOD, path="fw-0001.bin")
    with pytest.raises(DescriptorError, match="absolute"):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


def test_the_error_names_the_worker_and_tool():
    """An operator reading an audit row needs to know which tool violated the
    contract, not merely that one did."""
    with pytest.raises(DescriptorError) as e:
        validate_descriptor({}, worker="hardware", tool="dump_firmware")
    assert "hardware" in str(e.value) and "dump_firmware" in str(e.value)
