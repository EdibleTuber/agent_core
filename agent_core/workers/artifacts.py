"""The daemon's half of the artifact contract.

A tool that declares `produces: artifact` returns a DESCRIPTOR of a file it
wrote on its own machine, not the file's contents. This module states the wire
constant and validates what comes back.
"""
from __future__ import annotations

PRODUCES_META_KEY = "agent_core/produces"
"""Stated here and in pare-worker-kit, with a guard test on each side. See
RISK_TIER_META_KEY for the same arrangement and the same reasoning: the two
packages are installed separately on machines that never share a Python
environment."""

PRODUCES_RESULT = "result"
PRODUCES_ARTIFACT = "artifact"

VALID_PRODUCES = (PRODUCES_RESULT, PRODUCES_ARTIFACT)
