"""RiskGate — evaluate a tool call's effective tier given declared tier
and operator-supplied override patterns.

The gate enforces "override-up only": patterns can raise a tier but
never lower one. This matters because the declared tier is the worker's
own assessment; the operator's overrides are an additional safety layer.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Tuple, get_args

from agent_core.workers.types import RiskTier, WorkerSpec


_TIER_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

RISK_TIER_META_KEY = "agent_core/risk_tier"

_VALID_TIERS = set(_TIER_RANK)


def _max_tier(a: RiskTier, b: RiskTier) -> RiskTier:
    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b


def resolve_declared_tier(
    spec: WorkerSpec | None, advertised: str | None
) -> Tuple[RiskTier, str]:
    """Resolve the declared tier for a tool call from the worker-wide floor
    and the per-tool tier advertised over the wire.

    Model (escalate-only, monotonic): the declared tier is
    ``max(risk_default_floor, advertised_wire_tier)``. The operator-pin layer
    (RiskGate, applied above this) can escalate further. Safety for dangerous
    tools that fail to advertise is provided by mandatory operator pins +
    build-time conformance (which rejects missing/invalid tiers), NOT by a
    dispatch-time fail-safe. A worker that advertises no tier simply uses its
    ``risk_default`` floor — this keeps legacy non-advertising workers
    (e.g. the apk_re_agents fleet) working unchanged, and they auto-upgrade
    to full wire-tier escalation the day they start advertising.

    Returns (tier, tier_source) where tier_source is one of:
      "wire"               advertised tier escalated above the floor
      "floor"              floor used (advertised <= floor, external_mcp, or no tier)
      "invalid_advertised" worker sent a malformed tier -> floor (flagged for audit)
      "unknown_worker"     spec is None (worker not registered) -> "high"

    Rules:
      - Unknown worker (spec is None, i.e. dispatch to an unregistered worker):
        "high". This is about an unregistered worker, not a missing wire tier.
      - external_mcp: floor only (per-tool wire tiers are not honored; out of scope).
      - valid advertised: max(floor, advertised). source="wire" if it escalated
        above the floor, else "floor".
      - missing advertised (None): the floor. source="floor".
      - invalid advertised (non-str, or a str that isn't a valid tier): the floor,
        but source="invalid_advertised" so the audit trail flags malformed wire
        data (a contract violation / possible tampering signal). Note `advertised`
        may be hostile/arbitrary (dict, list, bool, ...) — the isinstance guard
        below keeps the membership test from raising on unhashable input.
    """
    if spec is None:
        return ("high", "unknown_worker")

    floor: RiskTier = spec.risk_default
    if spec.kind == "external_mcp":
        return (floor, "floor")

    if isinstance(advertised, str) and advertised in _VALID_TIERS:
        resolved = _max_tier(floor, advertised)  # type: ignore[arg-type]
        source = "wire" if _TIER_RANK[advertised] > _TIER_RANK[floor] else "floor"  # type: ignore[index]
        return (resolved, source)

    # No usable advertised tier: fall back to the worker-wide floor. Distinguish
    # "absent" from "malformed" so an auditor can spot a tampering/bug signal.
    source = "floor" if advertised is None else "invalid_advertised"
    return (floor, source)


@dataclass(frozen=True)
class TierDecision:
    """Result of risk evaluation: effective tier and override reason (if any)."""
    effective_tier: RiskTier
    override_reason: str | None


class RiskGate:
    """Maps (worker, tool, declared_tier) → effective tier, applying
    fnmatch-style override patterns from workers.yaml."""

    def __init__(self, overrides: list[tuple[str, RiskTier]]) -> None:
        """Initialize with a list of override rules.

        Args:
            overrides: list of (pattern, tier) pairs.
                Pattern matches against `f"{worker}_{tool}"`. The first pattern
                that produces a higher tier than the declared one wins; ties go
                to the first match.

        Raises:
            ValueError: if any override tier is invalid.
        """
        valid_tiers = set(get_args(RiskTier))
        for pattern, tier in overrides:
            if tier not in valid_tiers:
                raise ValueError(
                    f"invalid override tier {tier!r} for pattern {pattern!r}"
                )
        self._overrides = overrides

    def evaluate(
        self, *, worker: str, tool: str, declared_tier: RiskTier
    ) -> TierDecision:
        """Evaluate the effective tier for a tool call.

        Args:
            worker: worker name.
            tool: tool name.
            declared_tier: the tier declared by the worker.

        Returns:
            TierDecision with effective_tier and override_reason (if an
            upgrade pattern matched).
        """
        target = f"{worker}_{tool}"
        declared_rank = _TIER_RANK[declared_tier]

        # Find the highest matching override that's strictly above declared.
        # Override-up only: an override only takes effect if its tier rank
        # is strictly greater than the declared tier's rank.
        # Multiple matches: take the highest tier; if tied, first match wins.
        best: tuple[RiskTier, str] | None = None
        for pattern, tier in self._overrides:
            if not fnmatch.fnmatchcase(target, pattern):
                continue
            if _TIER_RANK[tier] <= declared_rank:
                continue  # override-up only: don't downgrade
            # Update best if this tier is higher, or if best is None.
            if best is None or _TIER_RANK[tier] > _TIER_RANK[best[0]]:
                best = (tier, pattern)

        if best is None:
            return TierDecision(effective_tier=declared_tier, override_reason=None)

        effective, pattern = best
        return TierDecision(
            effective_tier=effective,
            override_reason=f"name pattern {pattern!r} forces {effective}",
        )
