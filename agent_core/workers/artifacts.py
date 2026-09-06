"""The daemon's half of the artifact contract.

A tool that declares `produces: artifact` returns a DESCRIPTOR of a file it
wrote on its own machine, not the file's contents. This module states the wire
constant and validates what comes back.
"""
from __future__ import annotations

import re

PRODUCES_META_KEY = "agent_core/produces"
"""Stated here and in pare-worker-kit, with a guard test on each side. See
RISK_TIER_META_KEY for the same arrangement and the same reasoning: the two
packages are installed separately on machines that never share a Python
environment."""

PRODUCES_RESULT = "result"
PRODUCES_ARTIFACT = "artifact"

VALID_PRODUCES = (PRODUCES_RESULT, PRODUCES_ARTIFACT)

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_REQUIRED = ("host", "path", "size", "sha256")


class DescriptorError(ValueError):
    """A tool declared `produces: artifact` and returned something else."""


def validate_descriptor(payload, *, worker: str, tool: str) -> dict:
    """Check the SHAPE of an artifact descriptor. Not its truthfulness.

    Everything here is self-reported by the worker. This rejects a malformed
    descriptor, a confused one, and a buggy one. It does not make a hostile
    worker honest -- that is what containment (worker side) and
    content-addressing on retrieval are for.
    """
    where = f"{worker}.{tool}"
    if not isinstance(payload, dict):
        raise DescriptorError(
            f"{where} declared produces=artifact but returned "
            f"{type(payload).__name__}, not a JSON object")
    for field in _REQUIRED:
        if field not in payload:
            raise DescriptorError(f"{where}: descriptor is missing {field!r}")

    size = payload["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise DescriptorError(
            f"{where}: descriptor size must be a non-negative int, got {size!r}")

    sha = payload["sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise DescriptorError(
            f"{where}: descriptor sha256 must be 64 lowercase hex chars, "
            f"got {sha!r}")

    path = payload["path"]
    if not isinstance(path, str) or not path.startswith("/"):
        raise DescriptorError(
            f"{where}: descriptor path must be absolute, got {path!r}")

    return dict(payload)
