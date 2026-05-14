"""RiskGate — evaluate a tool call's effective tier given declared tier
and operator-supplied override patterns.

The gate enforces "override-up only": patterns can raise a tier but
never lower one. This matters because the declared tier is the worker's
own assessment; the operator's overrides are an additional safety layer.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import get_args

from agent_core.workers.types import RiskTier


_TIER_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


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
