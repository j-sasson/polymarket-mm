"""Derives a graph-implied "fair value" price for markets currently caught in
a constraint violation.

Given the violations `ConstraintSet.check()` returns, nudge the involved
markets' prices toward internal consistency:

  - negative_risk: the group's prices are renormalized to sum to exactly 1,
    scaling every member proportionally (the standard way to redistribute an
    over/under-priced set of mutually exclusive outcomes).
  - monotonicity: the superset/subset pair split their gap evenly -- we don't
    know which side is "wrong", so each moves halfway to meet the other.

Markets untouched by any active violation are left at their raw price, so
callers can treat `fair[m] - prices[m]` as "0 unless something's off".
"""
from __future__ import annotations

from pmm_data.constraints import Violation


def graph_fair_value(prices: dict[str, float], violations: list[Violation]) -> dict[str, float]:
    fair = dict(prices)

    for v in violations:
        if v.constraint_type != "negative_risk":
            continue
        total = sum(fair[m] for m in v.market_ids)
        if total > 0:
            for m in v.market_ids:
                fair[m] = fair[m] / total

    for v in violations:
        if v.constraint_type != "monotonicity":
            continue
        superset_id, subset_id = v.market_ids
        gap = fair[subset_id] - fair[superset_id]
        if gap > 0:
            fair[superset_id] += gap / 2
            fair[subset_id] -= gap / 2

    return fair
