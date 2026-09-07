import inspect

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
    """The hash has to be a hash -- but be exact about what a good one buys.

    It is integrity, not authenticity: it detects corruption across the
    transfer, so a corrupted or half-finished `scp` is caught rather than
    silently acted on, and it gives the artifact a stable identity for audit
    and dedup. It does NOT survive an untrusted producer, which is what this
    docstring used to claim. The worker supplies both the bytes and the
    digest, so a dump truncated AT PRODUCTION carries a perfectly correct
    sha256 of the truncated bytes and verifies successfully. pare_worker_kit's
    artifact_path says the same thing at length; agent_core is what the
    implementer of the retrieval path reads, so it must not say more.
    """
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


# Task 4: Slug validation tests
from agent_core.workers.artifacts import SLUG_RE, validate_slug


@pytest.mark.parametrize("ok", ["router-b", "a", "proj_2", "a" * 64,
                                "0target", "fw-dump_2026"])
def test_a_legal_slug_passes(ok):
    assert validate_slug(ok) == ok


@pytest.mark.parametrize("bad", [
    "proj/../../etc",     # traversal — the reason fullmatch is used
    "../etc",
    "a/b",
    "-rf",                # a leading dash is argument injection into scp/tar
    "--checkpoint-action=exec=sh",
    "_leading",           # ArcticBase requires a leading alphanumeric
    "UPPER",
    "has space",
    "a" * 65,             # ArcticBase caps at 64
    "",
    "Ünïcode",
])
def test_an_illegal_slug_is_rejected(bad):
    with pytest.raises(ValueError, match="slug"):
        validate_slug(bad)


def test_the_rule_matches_arcticbase_exactly():
    """The slug names a workbench, a capture store and a directory on the bench
    drive. ArcticBase is the strictest consumer, so its rule is the shared one:
    a project accepted here but rejected there would silently have no
    workbench."""
    from agent_core.workers.artifacts import SLUG_RE
    assert SLUG_RE.pattern == r"[a-z0-9][a-z0-9_-]{0,63}"


def test_a_non_string_is_rejected():
    with pytest.raises(ValueError, match="slug"):
        validate_slug(None)


# Fix Round 1: Host validation in descriptor
@pytest.mark.parametrize("bad", [
    "-oProxyCommand=sh",      # a leading dash is an ssh ARGUMENT, not a host
    "--rsh=sh",
    "x;rm -rf /",
    "host:/etc/shadow",       # a colon forges scp's host:path split
    "has space",
    "a" * 254,
    "", None, 123, True,
])
def test_a_host_that_is_not_a_hostname_is_rejected(bad):
    """The descriptor is retrieved with `scp <host>:<path>`, so host is as much
    a part of that command as path is."""
    payload = dict(_GOOD, host=bad)
    with pytest.raises(DescriptorError, match="host"):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


@pytest.mark.parametrize("ok", ["pare-bench", "100.68.47.23", "a",
                                "bench.local", "host_1"])
def test_a_real_host_passes(ok):
    out = validate_descriptor(dict(_GOOD, host=ok), worker="hardware",
                              tool="dump_firmware")
    assert out["host"] == ok


# Task 8: the slug rule is stated on both sides
def test_the_slug_rule_agrees_with_the_worker_kits():
    """Same arrangement as the produces constant above, for the same reason.
    The kit builds artifact paths from a slug the daemon validated; if the two
    rules drift, a slug the daemon accepts is one the worker refuses, and the
    operator sees a path error with no clue that the two disagree.

    Flags are compared as well as the pattern text: identical pattern strings
    compiled with different flags (re.IGNORECASE, re.ASCII, re.UNICODE) are
    different rules, so `.pattern` equality alone would not pin agreement.
    """
    kit = pytest.importorskip(
        "pare_worker_kit.artifacts",
        reason="pare-worker-kit is not installed here; the worker-side half "
               "of this check runs in that package's own suite")
    assert kit.SLUG_RE.pattern == SLUG_RE.pattern
    assert kit.SLUG_RE.flags == SLUG_RE.flags


# Fix round 2: `path` hardened to the same standard as `host`
@pytest.mark.parametrize("bad", [
    "/mnt/store/../../etc/shadow",   # `..` is resolved by the kernel, so no
    "/mnt/s/../..",                  # amount of quoting at the call site
    "/..",                           # stops it
    "/mnt/s/-rf",                    # lands locally as a file named `-rf`
    "/mnt/s/-oProxyCommand=sh",
    "/mnt/s/-rf/",                   # a trailing slash does not hide the name
    "/mnt/s/f\x00/../etc",           # NUL truncates the path in every C API
    "/mnt/s/f\x00.bin",
    "/mnt/s/two\nlines",             # breaks any line-oriented audit record
    "/mnt/s/esc\x1b[2Kbin",          # rewrites what the operator SEES
    "/mnt/s/bell\x07",
    "/mnt/s/del\x7f",
])
def test_a_path_that_would_misbehave_in_the_retrieval_command_is_rejected(bad):
    """`path` is the other half of `scp <host>:<path>`, and until this round
    it had only isinstance+startswith('/') while `host` had a full grammar.

    Every case here is one a CORRECT caller cannot fix: `..` belongs to the
    kernel, a leading dash belongs to argument parsing, and a control
    character is not a character the operator can see. Quoting fixes none of
    them.
    """
    payload = dict(_GOOD, path=bad)
    with pytest.raises(DescriptorError, match="path"):
        validate_descriptor(payload, worker="hardware", tool="dump_firmware")


@pytest.mark.parametrize("ok", [
    "/mnt/bench-store/router-b/fw-0001.bin",
    "/mnt/s/fw..bin",           # two dots in a NAME is not a `..` COMPONENT
    "/mnt/s/..hidden",         # accepted HERE though the kit would not build
                               # it -- see the test docstring
    "/mnt/s/v1.2.3/fw.bin",
    "/mnt/s/dump-2026-09-06.bin",
    "/mnt/s/a b.bin",           # a space is ordinary in a filename
])
def test_a_legitimate_path_still_passes(ok):
    """The traversal check is COMPONENT-WISE, never a substring search.
    Refusing every path containing the two characters `..` would refuse
    ordinary filenames and buy nothing: `fw..bin` escapes nothing.

    `..hidden` is the case where the two packages DELIBERATELY differ, and
    neither said so until now. pare-worker-kit's `_NAME_RE` requires a
    leading alphanumeric, so no kit-BUILT path can ever have that basename.
    This function is looser on purpose: it validates descriptors from ANY
    worker, including ones that never used the kit to construct the path, and
    a leading dot is an ordinary filename on the filesystem the worker owns.
    Tightening here to match the kit would reject a conformant non-kit worker
    for a rule the wire contract never stated -- and would buy nothing, since
    a leading dot escapes nothing and is not argument-injection range (that
    is the leading DASH, refused above).
    """
    out = validate_descriptor(dict(_GOOD, path=ok), worker="hardware",
                              tool="dump_firmware")
    assert out["path"] == ok


@pytest.mark.parametrize("accepted", [
    "/mnt/s/f;rm -rf /", "/mnt/s/$(id)", "/mnt/s/`id`", "/mnt/s/a|b",
    "/mnt/s/a&b", "/mnt/s/*.bin",
])
def test_shell_metacharacters_are_deliberately_accepted(accepted):
    """A DECISION, pinned so it is not reversed by reflex.

    Refusing `;`, `$`, backtick and `|` would look like shell-safety and
    could not deliver it: space, quote, backslash, `*`, `?`, `[` and `~` are
    all ordinary in filenames and all enough to break an unquoted splice, so
    the blocklist would leave the hole open while advertising that it was
    shut -- which is worse than not claiming it, because it invites the
    unquoted splice. These characters also occur in real filenames.

    The property is bought at the CALL SITE instead, but by the TRANSFER
    MODE and not by quoting: an SFTP-mode transfer (`scp` without `-O`,
    `sftp`, `rsync -s`) never lets a remote shell re-parse the argument.
    argv construction and shlex.quote protect the operator's LOCAL shell
    only -- under legacy `scp -O`, scp(1)'s CAVEATS say the REMOTE user's
    shell is executed for glob(3) matching, so `$(id)` in the path survives
    correct local quoting and runs on the named host. A blocklist here would
    not have saved that case either: it is the transfer mode that decides it.

    If a future change reverses this, it should delete this test and replace
    the reasoning -- not leave it passing by accident.
    """
    out = validate_descriptor(dict(_GOOD, path=accepted), worker="hardware",
                              tool="dump_firmware")
    assert out["path"] == accepted


def test_containment_against_the_artifact_root_is_not_checked_here():
    """The named gap, pinned as behaviour rather than left as a comment.

    `validate_descriptor(payload, *, worker, tool)` is handed a worker NAME,
    not its WorkerSpec, so it has no artifact_root to compare against and
    containment is not expressible in this signature. `/etc/shadow` is
    well-formed by every rule this function knows.

    That is not an oversight to be patched here by guessing a root. It
    belongs to the caller that resolves the WorkerSpec -- the dispatch path
    that routes on the produces declaration, which does not exist yet. This
    test exists so that whoever adds a root to this signature is told to move
    the paragraph in the docstring at the same time.
    """
    for uncontained in ("/etc/shadow", "/some-other-root/router-b/fw.bin"):
        out = validate_descriptor(dict(_GOOD, path=uncontained),
                                  worker="hardware", tool="dump_firmware")
        assert out["path"] == uncontained
    # The STRUCTURAL half of the gap, asserted rather than a search for words
    # in __doc__ -- which is None under `python -OO`, where the assertion
    # would have been vacuous. This is also the more useful trigger: the day
    # a root is threaded into this signature, this fails and says so.
    assert not {"root", "artifact_root", "spec"} & set(
        inspect.signature(validate_descriptor).parameters)
