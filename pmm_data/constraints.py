"""Standalone constraint-checking logic for logically related Polymarket markets.

Two constraint types, matching how these markets are actually structured:

  - NegativeRiskGroup: a set of mutually exclusive & exhaustive outcomes
    (e.g. CPI print buckets, or the 5 buckets of a single Fed meeting) whose
    prices must sum to ~1.
  - MonotoneConstraint: one market's outcome event is a superset of another's
    (e.g. "cuts by June" is true in every scenario where "cuts by March" is
    true, plus more), so the superset's price can never sit below the
    subset's price.

This module has no knowledge of live data, websockets, or Polymarket's APIs —
it operates purely on {market_id: price} dicts, so the logic can be built and
tested against hand-made examples before it's wired to a real feed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NegativeRiskGroup:
    name: str
    market_ids: tuple[str, ...]
    tolerance: float = 0.02  # allowed slack (bid/ask noise, rounding) before flagging

    def __post_init__(self):
        if len(self.market_ids) < 2:
            raise ValueError(f"{self.name}: negative-risk group needs >= 2 markets")


@dataclass(frozen=True)
class MonotoneConstraint:
    """`superset_id`'s price must be >= `subset_id`'s price."""

    name: str
    superset_id: str
    subset_id: str
    tolerance: float = 0.0

    def __post_init__(self):
        if self.superset_id == self.subset_id:
            raise ValueError(f"{self.name}: superset and subset can't be the same market")


@dataclass(frozen=True)
class Violation:
    constraint_name: str
    constraint_type: str  # "negative_risk" | "monotonicity"
    market_ids: tuple[str, ...]
    magnitude: float  # how far past tolerance, in price units (0-1 scale)
    detail: str


class ConstraintSet:
    def __init__(self, negative_risk_groups=None, monotone_constraints=None):
        self.negative_risk_groups: list[NegativeRiskGroup] = list(negative_risk_groups or [])
        self.monotone_constraints: list[MonotoneConstraint] = list(monotone_constraints or [])

    def check(self, prices: dict[str, float]) -> list[Violation]:
        """Return every constraint currently violated by `prices`.

        Constraints referencing a market_id absent from `prices` are skipped
        (not violated) rather than erroring, since a live feed won't always
        have every market's price at every instant.
        """
        violations: list[Violation] = []
        for group in self.negative_risk_groups:
            v = self._check_negative_risk(group, prices)
            if v:
                violations.append(v)
        for constraint in self.monotone_constraints:
            v = self._check_monotone(constraint, prices)
            if v:
                violations.append(v)
        return violations

    @staticmethod
    def _check_negative_risk(group: NegativeRiskGroup, prices: dict[str, float]):
        if any(m not in prices for m in group.market_ids):
            return None
        total = sum(prices[m] for m in group.market_ids)
        deviation = total - 1.0
        if abs(deviation) <= group.tolerance:
            return None
        return Violation(
            constraint_name=group.name,
            constraint_type="negative_risk",
            market_ids=group.market_ids,
            magnitude=round(abs(deviation), 6),
            detail=f"prices sum to {total:.4f}, expected 1.0 (+/- {group.tolerance})",
        )

    @staticmethod
    def _check_monotone(constraint: MonotoneConstraint, prices: dict[str, float]):
        if constraint.superset_id not in prices or constraint.subset_id not in prices:
            return None
        superset_price = prices[constraint.superset_id]
        subset_price = prices[constraint.subset_id]
        gap = subset_price - superset_price
        if gap <= constraint.tolerance:
            return None
        return Violation(
            constraint_name=constraint.name,
            constraint_type="monotonicity",
            market_ids=(constraint.superset_id, constraint.subset_id),
            magnitude=round(gap - constraint.tolerance, 6),
            detail=(
                f"subset '{constraint.subset_id}' price {subset_price:.4f} exceeds "
                f"superset '{constraint.superset_id}' price {superset_price:.4f}"
            ),
        )
