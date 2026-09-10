"""Two sweeps from PARE/docs/superpowers/2026-09-07-artifact-contract-followups.md.

Both are the same shape: a rule stated on both sides of a wire, where only one
side was checking.
"""
from __future__ import annotations

import pytest

from agent_core.workers.types import WorkerSpec

# The operator declares artifact_root in workers.yaml. The WORKER enforces
# containment under it (D7) -- the daemon cannot, because the artifact is on
# another machine and resolving the path here resolves it against the wrong
# namespace. But the daemon can apply the same LEXICAL rule at config load, and
# should: otherwise the operator types a root the daemon accepts and the worker
# refuses, and finds out at hardware-run time with a dump half-written.
ROOTS_THE_WORKER_ACCEPTS = ["/mnt/store", "/mnt/store/", "///mnt/store",
                            "/./mnt/store", "/mnt/bench-store"]
ROOTS_THE_WORKER_REFUSES = ["/a/../b", "//mnt/store", "//a", "//",
                            "relative/path", ""]


def _spec(root):
    return WorkerSpec(name="hardware", transport="stdio", command="/bin/true",
                      risk_default="high", artifact_root=root)


@pytest.mark.parametrize("root", ROOTS_THE_WORKER_ACCEPTS)
def test_the_daemon_accepts_every_root_the_worker_would(root):
    assert _spec(root).artifact_root == root


@pytest.mark.parametrize("root", ROOTS_THE_WORKER_REFUSES)
def test_the_daemon_refuses_every_root_the_worker_would(root):
    """Config load is the right moment to learn this, not hardware-run time."""
    with pytest.raises(ValueError):
        _spec(root)


def test_the_two_rules_agree_exactly_when_the_kit_is_installed():
    """The real guarantee: one table, both implementations, identical verdicts.

    Asserting agreement beats restating the kit's rule here, which would just
    be a second copy free to drift.
    """
    kit = pytest.importorskip(
        "pare_worker_kit.artifacts",
        reason="pare-worker-kit is not installed here; the worker-side half of "
               "this check runs in the kit's own suite")

    for root in ROOTS_THE_WORKER_ACCEPTS + ROOTS_THE_WORKER_REFUSES:
        try:
            kit.artifact_path(root, "proj", "f.bin")
            worker_ok = True
        except kit.ArtifactPathError:
            worker_ok = False
        try:
            _spec(root)
            daemon_ok = True
        except ValueError:
            daemon_ok = False
        assert worker_ok == daemon_ok, (
            f"{root!r}: worker {'accepts' if worker_ok else 'refuses'} but "
            f"daemon {'accepts' if daemon_ok else 'refuses'}")


def test_none_still_means_may_not_produce_artifacts():
    assert _spec(None).artifact_root is None


def test_the_tier_list_agrees_with_the_worker_kits():
    """VALID_RISK_TIERS (kit) and _WIRE_VALID_TIERS (here) are independent
    literals and were unguarded in BOTH directions -- produces and the slug
    rule were guarded from the start; this one predates the practice.

    Drift is misleading in a specific way: the conformance check asserts an
    advertised tier is in THIS list, so a tier the kit gains and the daemon does
    not know makes every worker advertising it fail conformance with a message
    about an invalid tier, for a tier that is valid on one side.
    """
    from agent_core.workers.conformance import _WIRE_VALID_TIERS
    kit = pytest.importorskip(
        "pare_worker_kit",
        reason="pare-worker-kit is not installed here; the worker-side half of "
               "this check runs in the kit's own suite")
    assert set(_WIRE_VALID_TIERS) == set(kit.VALID_RISK_TIERS)
