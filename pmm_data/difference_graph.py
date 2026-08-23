"""Difference-constraint graph over monotone relationships.

pmm_data.constraints.MonotoneConstraint only checks pairs someone explicitly
wrote down (e.g. "September hike implies yearly hike"). If we also had
"October hike implies yearly hike" and, hypothetically, "yearly hike implies
some broader 2026-2027 hike market", a violation between September and that
broader market -- two markets NEVER directly compared -- would go undetected
by pairwise checks alone, even though the constraint graph implies a bound
between them once you chain the edges.

This is the classic difference-constraint-system encoding: a constraint
"subset <= superset" (subset - superset <= 0) becomes a directed edge
superset -> subset with weight 0. Floyd-Warshall then computes the TIGHTEST
bound between every pair of nodes, including ones connected only via a
multi-hop chain -- the graph-theory generalization of checking pairs by
hand. A negative cycle would mean the constraint graph is self-contradictory
regardless of any price (shouldn't happen with real superset/subset edges,
whose weights are never negative; surfaced as `negative_cycle` for safety).

Honesty note: with our current constraint set (six edges, all pointing from
a single meeting-hike bucket into the one shared "yearly hike" node -- a
star, not a chain), there are no multi-hop paths to derive yet, so this
currently finds nothing beyond what MonotoneConstraint already checks
directly. It's infrastructure for the day a chain actually exists (e.g. if
a second yearly-scope market or a cumulative-by-date market gets added),
not a claim that it's finding something new right now.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pmm_data.constraints import ConstraintSet


@dataclass(frozen=True)
class DerivedBoundViolation:
    """Shape-compatible with pmm_data.constraints.Violation (constraint_name,
    constraint_type, market_ids, magnitude) so it can drive a
    pmm_data.violation_tracker.ViolationTracker with zero changes there."""
    constraint_name: str
    constraint_type: str
    market_ids: tuple[str, ...]  # the full derivation path, from_market..to_market
    magnitude: float
    derived_bound: float
    observed_gap: float


class DifferenceGraph:
    """All-pairs shortest-path closure of a ConstraintSet's monotone
    constraints, via Floyd-Warshall."""

    def __init__(self, constraint_set: ConstraintSet):
        self.nodes: list[str] = sorted({
            m for c in constraint_set.monotone_constraints for m in (c.superset_id, c.subset_id)
        })
        self._index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        dist = [[math.inf] * n for _ in range(n)]
        nxt: list[list[int | None]] = [[None] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0

        for c in constraint_set.monotone_constraints:
            i, j = self._index[c.superset_id], self._index[c.subset_id]
            if c.tolerance < dist[i][j]:
                dist[i][j] = c.tolerance
                nxt[i][j] = j

        for k in range(n):
            for i in range(n):
                if dist[i][k] == math.inf:
                    continue
                for j in range(n):
                    if dist[k][j] == math.inf:
                        continue
                    via = dist[i][k] + dist[k][j]
                    if via < dist[i][j]:
                        dist[i][j] = via
                        nxt[i][j] = nxt[i][k]

        self.negative_cycle = any(dist[i][i] < 0 for i in range(n))
        self._dist = dist
        self._next = nxt

    def bound(self, from_market: str, to_market: str) -> float | None:
        """Tightest implied upper bound on price[to] - price[from], or None
        if no path connects them."""
        if from_market not in self._index or to_market not in self._index:
            return None
        i, j = self._index[from_market], self._index[to_market]
        d = self._dist[i][j]
        return d if d != math.inf else None

    def path(self, from_market: str, to_market: str) -> tuple[str, ...] | None:
        if from_market not in self._index or to_market not in self._index:
            return None
        i, j = self._index[from_market], self._index[to_market]
        if i != j and self._next[i][j] is None:
            return None
        route = [i]
        cur = i
        while cur != j:
            cur = self._next[cur][j]
            route.append(cur)
        return tuple(self.nodes[k] for k in route)

    def hop_count(self, from_market: str, to_market: str) -> int | None:
        p = self.path(from_market, to_market)
        return None if p is None else len(p) - 1


def check_derived_violations(
    graph: DifferenceGraph,
    prices: dict[str, float],
    min_hops: int = 2,
    tolerance: float = 0.0,
) -> list[DerivedBoundViolation]:
    """Only reports pairs reached via >= min_hops edges -- single-hop pairs
    are exactly what MonotoneConstraint.check already covers directly, so
    default-excluding them here avoids double-reporting the same finding
    under two names."""
    violations = []
    for i in graph.nodes:
        if i not in prices:
            continue
        for j in graph.nodes:
            if i == j or j not in prices:
                continue
            if (graph.hop_count(i, j) or 0) < min_hops:
                continue
            bound = graph.bound(i, j)
            if bound is None:
                continue
            observed = prices[j] - prices[i]
            excess = observed - bound - tolerance
            if excess > 0:
                path = graph.path(i, j)
                violations.append(DerivedBoundViolation(
                    constraint_name=f"derived:{'->'.join(path)}",
                    constraint_type="derived_monotone",
                    market_ids=path,
                    magnitude=round(excess, 6),
                    derived_bound=bound,
                    observed_gap=round(observed, 6),
                ))
    return violations


class DerivedBoundChecker:
    """Adapter exposing `.check(prices)`, so this can drive a
    pmm_data.violation_tracker.ViolationTracker exactly like a ConstraintSet
    does -- reuses the same tested open/persist/resolve episode logic
    rather than re-deriving it (and re-deriving its bugs)."""

    def __init__(self, constraint_set: ConstraintSet, min_hops: int = 2, tolerance: float = 0.0):
        self.graph = DifferenceGraph(constraint_set)
        self.min_hops = min_hops
        self.tolerance = tolerance

    def check(self, prices: dict[str, float]) -> list[DerivedBoundViolation]:
        return check_derived_violations(self.graph, prices, self.min_hops, self.tolerance)
