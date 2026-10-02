import inspect

import pytest

from agent_core.workers.artifacts import (ARTIFACT_DESCRIPTOR_FIELDS,
                                          PRODUCES_ARTIFACT,
                                          PRODUCES_META_KEY,
                                          PRODUCES_RESULT,
                                          RESERVED_DRIVE_ID_ARG,
                                          RESERVED_SLUG_ARG,
                                          VALID_PRODUCES)


def test_the_daemon_states_the_wire_constant_itself():
    assert PRODUCES_META_KEY == "agent_core/produces"
    assert VALID_PRODUCES == ("result", "artifact")
    assert (PRODUCES_RESULT, PRODUCES_ARTIFACT) == ("result", "artifact")


def test_the_daemon_states_the_descriptor_field_set_itself():
    """A LOCAL pin, the daemon-side half of the arrangement the meta-key test
    above states: the field set crosses a wire and is stated independently in
    each package, so a change to it here must fail a test in THIS suite even
    when the importorskip guard below skips."""
    assert ARTIFACT_DESCRIPTOR_FIELDS == ("host", "path", "size", "sha256",
                                          "hashed_at", "media_type",
                                          "drive_id")
    assert len(set(ARTIFACT_DESCRIPTOR_FIELDS)) == len(
        ARTIFACT_DESCRIPTOR_FIELDS)


def test_the_reserved_argument_names_are_stable_wire_literals():
    """Injected by the daemon at dispatch, named after by worker handlers:
    wire vocabulary AND Python identifiers. Pinned as literals; changing one
    is a wire-breaking change."""
    assert RESERVED_SLUG_ARG == "project_slug"
    assert RESERVED_DRIVE_ID_ARG == "expected_drive_id"
    assert RESERVED_SLUG_ARG.isidentifier()
    assert RESERVED_DRIVE_ID_ARG.isidentifier()
    assert RESERVED_SLUG_ARG != RESERVED_DRIVE_ID_ARG


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
from agent_core.workers.types import WorkerSpec

_DRIVE = "12345678-90ab-4cd0-8e12-34567890abcd"
"""A made-up sentinel UUID for tests. Never the bench drive's id: that value
is read from the drive, not typed from memory."""

_GOOD = {
    "host": "pare-bench",
    "path": "/mnt/bench-store/router-b/fw-0001.bin",
    "size": 2147483648,
    "sha256": "a" * 64,
    "hashed_at": "2026-09-06T12:34:56Z",
    "media_type": "application/octet-stream",
    "drive_id": _DRIVE,
}


def _spec(root="/mnt/bench-store", drive_id=_DRIVE):
    return WorkerSpec(name="hardware", transport="stdio", command="/bin/true",
                      risk_default="high", artifact_root=root,
                      artifact_drive_id=drive_id)


def _v(payload, *, root="/mnt/bench-store", drive_id=_DRIVE, slug="router-b"):
    """The one call shape for the rest of this file. The default spec makes
    the _GOOD path contained (root /mnt/bench-store, slug router-b), so the
    shape tests keep testing shape, and the containment tests opt out by
    changing root, drive_id or slug."""
    return validate_descriptor(payload, spec=_spec(root, drive_id),
                               tool="dump_firmware", slug=slug)


def test_a_well_formed_descriptor_passes_through():
    out = _v(dict(_GOOD))
    assert out == dict(_GOOD)


@pytest.mark.parametrize("missing", ARTIFACT_DESCRIPTOR_FIELDS)
def test_a_missing_required_field_is_rejected(missing):
    """A tool that declares `artifact` and returns something else is a contract
    violation, recorded as an error rather than silently treated as a result.
    Parametrised over the constant, not a re-typed list: the field set IS the
    constant, and a field landing in it must light this test up."""
    payload = dict(_GOOD)
    del payload[missing]
    with pytest.raises(DescriptorError, match=missing):
        _v(payload)


def test_a_non_dict_payload_is_rejected():
    with pytest.raises(DescriptorError, match="not a JSON object"):
        _v(["nope"])


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
        _v(payload)


@pytest.mark.parametrize("bad", [-1, "big", 1.5, None])
def test_a_non_integer_size_is_rejected(bad):
    payload = dict(_GOOD, size=bad)
    with pytest.raises(DescriptorError, match="size"):
        _v(payload)


def test_a_relative_path_is_rejected():
    """Containment is checked against an absolute root; a relative path cannot
    be compared against one."""
    payload = dict(_GOOD, path="fw-0001.bin")
    with pytest.raises(DescriptorError, match="absolute"):
        _v(payload)


def test_the_error_names_the_worker_and_tool():
    """An operator reading an audit row needs to know which tool violated the
    contract, not merely that one did."""
    with pytest.raises(DescriptorError) as e:
        _v({})
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
        _v(payload)


@pytest.mark.parametrize("ok", ["pare-bench", "100.68.47.23", "a",
                                "bench.local", "host_1"])
def test_a_real_host_passes(ok):
    out = _v(dict(_GOOD, host=ok))
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


def test_the_descriptor_contract_agrees_with_the_worker_kit():
    """The field set and the two reserved argument names are stated on both
    sides of the wire, and this is what keeps the two statements the same,
    for the reason the slug-rule guard above gives.

    Named so the CI filter (`-k agrees_with_the_worker_kit`) collects it: the
    cross-package step fails if a guard matching that filter is skipped or if
    none is collected.

    Compared as a tuple, not a set: the order is the wire order.
    """
    kit = pytest.importorskip(
        "pare_worker_kit.artifacts",
        reason="pare-worker-kit is not installed here; the worker-side half "
               "of this check runs in that package's own suite")
    assert kit.ARTIFACT_DESCRIPTOR_FIELDS == ARTIFACT_DESCRIPTOR_FIELDS
    assert kit.RESERVED_SLUG_ARG == RESERVED_SLUG_ARG
    assert kit.RESERVED_DRIVE_ID_ARG == RESERVED_DRIVE_ID_ARG


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
        _v(payload)


@pytest.mark.parametrize("path,root,slug", [
    ("/mnt/bench-store/router-b/fw-0001.bin", "/mnt/bench-store", "router-b"),
    ("/mnt/s/proj/fw..bin", "/mnt/s", "proj"),           # two dots in a NAME
    ("/mnt/s/proj/..hidden", "/mnt/s", "proj"),          # daemon looser than the kit
    ("/mnt/s/proj/v1.2.3/fw.bin", "/mnt/s", "proj"),
    ("/mnt/s/proj/dump-2026-09-06.bin", "/mnt/s", "proj"),
    ("/mnt/s/proj/a b.bin", "/mnt/s", "proj"),           # a space is ordinary
])
def test_a_legitimate_path_still_passes(path, root, slug):
    """The traversal check is COMPONENT-WISE, never a substring search.
    Refusing every path containing the two characters `..` would refuse
    ordinary filenames and buy nothing: `fw..bin` escapes nothing.

    `..hidden` is the case where the two packages DELIBERATELY differ, and
    neither said so until then. pare-worker-kit's `_NAME_RE` requires a
    leading alphanumeric, so no kit-BUILT path can ever have that basename.
    This function is looser on purpose: it validates descriptors from ANY
    worker, including ones that never used the kit to construct the path, and
    a leading dot is an ordinary filename on the filesystem the worker owns.
    Tightening here to match the kit would reject a conformant non-kit worker
    for a rule the wire contract never stated -- and would buy nothing, since
    a leading dot escapes nothing and is not argument-injection range (that
    is the leading DASH, refused above).

    The paths now carry a project directory because containment is checked
    against {root}/{slug} (Task 4): the slug the daemon injected is what the
    path must sit under.
    """
    out = _v(dict(_GOOD, path=path), root=root, slug=slug)
    assert out["path"] == path


@pytest.mark.parametrize("accepted", [
    "/mnt/s/proj/f;rm -rf /", "/mnt/s/proj/$(id)", "/mnt/s/proj/`id`",
    "/mnt/s/proj/a|b", "/mnt/s/proj/a&b", "/mnt/s/proj/*.bin",
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
    out = _v(dict(_GOOD, path=accepted), root="/mnt/s", slug="proj")
    assert out["path"] == accepted


def test_the_signature_holds_the_spec_and_the_slug():
    """The inverse of the gap pin deleted in this round. The old test pinned
    the ABSENCE of a spec from this signature; its docstring said the day a
    spec is threaded in, the test and the docstring paragraph move together.
    This asserts the new half structurally -- not by searching __doc__, which
    is None under `python -OO` -- so the signature cannot be stripped back to
    a worker name while the docstring still claims containment."""
    params = set(inspect.signature(validate_descriptor).parameters)
    assert {"spec", "slug"} <= params
    assert "worker" not in params


def test_a_path_outside_the_root_is_refused():
    """The case the deleted gap test pinned as ACCEPTED: `/etc/shadow` is
    well-formed by every shape rule and is now refused, because containment
    is checked and the path is not under the project directory."""
    with pytest.raises(DescriptorError, match="not under"):
        _v(dict(_GOOD, path="/etc/shadow"))


def test_a_path_under_the_root_but_another_project_is_refused():
    """The reason containment is against {root}/{slug} and not the root alone
    (spec §10, Risk 1): a descriptor that lies about its project is under
    the root and would bind the injected slug to nothing if the check were
    against the root."""
    with pytest.raises(DescriptorError, match="not under"):
        _v(dict(_GOOD, path="/mnt/bench-store/other-proj/fw.bin"))


def test_a_path_directly_under_the_root_is_refused():
    """The worker builds {root}/{slug}/{name}; a file with no project
    directory component is not a descriptor this contract recognises."""
    with pytest.raises(DescriptorError, match="not under"):
        _v(dict(_GOOD, path="/mnt/bench-store/fw.bin"))


def test_a_prefix_spoof_of_the_project_directory_is_refused():
    """`/mnt/bench-store/router-b-evil/...` has the project directory as a
    STRING prefix. Containment is decided by commonpath on components, not
    by startswith -- the same reason the traversal check is component-wise."""
    with pytest.raises(DescriptorError, match="not under"):
        _v(dict(_GOOD, path="/mnt/bench-store/router-b-evil/fw.bin"))


def test_a_trailing_slash_root_is_contained_normally():
    """An operator's trailing slash in workers.yaml is not a mistake worth
    refusing: the root is normalised before the comparison, the way the
    kit's artifact_path normalises it."""
    out = _v(dict(_GOOD), root="/mnt/bench-store/")
    assert out["path"] == _GOOD["path"]


def test_a_worker_that_declares_no_root_refuses_the_descriptor():
    """Fail closed at the validator level (ruling R5). Dispatch (P3) refuses
    earlier, with the operator-facing message; this keeps the function safe
    to call standalone."""
    with pytest.raises(DescriptorError, match="artifact_root"):
        _v(dict(_GOOD), root=None)


def test_a_drive_id_mismatch_is_refused_and_names_both():
    other = "ffffeeee-dddd-4ccc-8bbb-aaaaaaaaaaaa"
    with pytest.raises(DescriptorError) as e:
        _v(dict(_GOOD, drive_id=other))
    msg = str(e.value)
    assert other in msg and _DRIVE in msg


def test_a_worker_without_a_declared_drive_id_refuses_the_descriptor():
    """A5: the drive id is required whenever the root is; a descriptor cannot
    be checked against a drive the worker never declared."""
    with pytest.raises(DescriptorError, match="artifact_drive_id"):
        _v(dict(_GOOD), drive_id=None)


@pytest.mark.parametrize("bad", [
    "12345678-90AB-4CD0-8E12-34567890ABCD",   # uppercase: a second spelling
    "1234567890ab4cd08e1234567890abcd",       # no dashes
    "12345678-90ab-4cd0-8e12-34567890abc",    # final group 11, not 12
    "12345678-90ab-4cd0-8e12-34567890abcd1",  # final group 13
    "12345678-90ab-4cd0-8e12-34567890abc\x1b",  # the value is printed into
                                                # the error an operator reads
    None, 123,
])
def test_a_drive_id_that_is_not_the_sentinel_grammar_is_refused(bad):
    with pytest.raises(DescriptorError, match="drive_id"):
        _v(dict(_GOOD, drive_id=bad))


@pytest.mark.parametrize("bad", [
    1757160000.0,                  # the old fixture's epoch float
    "2026-09-06 12:34:56Z",        # space, not T
    "2026-09-06T12:34:56+02:00",   # a non-UTC offset
    "2026-09-06T12:34:56",         # no zone
    "2026-09-06T12:34:56z",        # lowercase z: one spelling, pinned
    None,
])
def test_a_hashed_at_that_is_not_rfc3339_utc_is_refused(bad):
    with pytest.raises(DescriptorError, match="hashed_at"):
        _v(dict(_GOOD, hashed_at=bad))


@pytest.mark.parametrize("ok", ["2026-09-06T12:34:56Z",
                                "2026-09-06T12:34:56.789Z"])
def test_an_rfc3339_utc_hashed_at_passes(ok):
    assert _v(dict(_GOOD, hashed_at=ok))["hashed_at"] == ok


@pytest.mark.parametrize("bad", [
    "application",                         # no subtype
    "/octet-stream",
    "application/",
    "application/octet-stream; q=0.5",     # parameters are not part of
                                            # type/subtype
    "*/*",                                 # wildcards: a descriptor names
                                            # what the file IS
    "application/octet stream",
    "", None, 123,
])
def test_a_media_type_that_is_not_an_iana_type_subtype_is_refused(bad):
    with pytest.raises(DescriptorError, match="media_type"):
        _v(dict(_GOOD, media_type=bad))


@pytest.mark.parametrize("ok", ["application/octet-stream", "text/plain",
                                "multipart/related", "TEXT/PLAIN"])
def test_an_iana_type_subtype_passes(ok):
    assert _v(dict(_GOOD, media_type=ok))["media_type"] == ok
