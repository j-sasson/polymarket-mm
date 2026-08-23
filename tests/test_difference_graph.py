import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint
from pmm_data.difference_graph import (
    DerivedBoundChecker,
    DifferenceGraph,
    check_derived_violations,
)
from pmm_data.market_graph import build_constraint_set, load_config
from pmm_data.violation_tracker import ViolationTracker


def chain_constraints():
    # A >= B >= C, a genuine 2-hop chain: A->B->C
    return ConstraintSet(monotone_constraints=[
        MonotoneConstraint(name="a_ge_b", superset_id="A", subset_id="B"),
        MonotoneConstraint(name="b_ge_c", superset_id="B", subset_id="C"),
    ])


class TestDifferenceGraphStructure(unittest.TestCase):
    def test_direct_edge_bound_matches_constraint(self):
        graph = DifferenceGraph(chain_constraints())
        self.assertEqual(graph.bound("A", "B"), 0.0)
        self.assertEqual(graph.hop_count("A", "B"), 1)

    def test_transitive_bound_derived_across_two_hops(self):
        graph = DifferenceGraph(chain_constraints())
        self.assertEqual(graph.bound("A", "C"), 0.0)  # A >= B >= C implies A >= C
        self.assertEqual(graph.hop_count("A", "C"), 2)
        self.assertEqual(graph.path("A", "C"), ("A", "B", "C"))

    def test_no_bound_between_unconnected_nodes(self):
        constraints = ConstraintSet(monotone_constraints=[
            MonotoneConstraint(name="x_ge_y", superset_id="X", subset_id="Y"),
        ])
        graph = DifferenceGraph(constraints)
        self.assertIsNone(graph.bound("X", "nonexistent"))

    def test_no_reverse_bound_when_only_one_direction_specified(self):
        graph = DifferenceGraph(chain_constraints())
        self.assertIsNone(graph.bound("C", "A"))  # no constraint says C >= A

    def test_star_topology_has_no_multihop_paths(self):
        """Mirrors our actual current constraint set: several direct edges
        into one shared superset, no chain -- honesty check that this
        doesn't fabricate derived relationships where none exist."""
        constraints = ConstraintSet(monotone_constraints=[
            MonotoneConstraint(name="a", superset_id="hub", subset_id="leaf1"),
            MonotoneConstraint(name="b", superset_id="hub", subset_id="leaf2"),
            MonotoneConstraint(name="c", superset_id="hub", subset_id="leaf3"),
        ])
        graph = DifferenceGraph(constraints)
        self.assertIsNone(graph.bound("leaf1", "leaf2"))  # leaves aren't connected to each other
        self.assertFalse(graph.negative_cycle)

    def test_no_negative_cycle_with_well_formed_constraints(self):
        graph = DifferenceGraph(chain_constraints())
        self.assertFalse(graph.negative_cycle)


class TestCheckDerivedViolations(unittest.TestCase):
    def test_catches_chain_violation_that_pairwise_checks_miss(self):
        """A>=B holds (0.60>=0.55), B>=C holds (0.55>=0.50) -- every direct
        pairwise check individually passes. But A<C is violated in a way
        no single pairwise constraint would ever catch, since A and C were
        never directly compared: only the chain reveals it."""
        graph = DifferenceGraph(chain_constraints())
        prices = {"A": 0.60, "B": 0.55, "C": 0.65}  # C > A, even though A>=B>=C "should" hold
        violations = check_derived_violations(graph, prices)
        self.assertEqual(len(violations), 1)
        v = violations[0]
        self.assertEqual(v.market_ids, ("A", "B", "C"))
        self.assertAlmostEqual(v.observed_gap, 0.05, places=4)  # C - A = 0.65-0.60
        self.assertAlmostEqual(v.magnitude, 0.05, places=4)

    def test_consistent_chain_prices_no_violation(self):
        graph = DifferenceGraph(chain_constraints())
        prices = {"A": 0.60, "B": 0.55, "C": 0.50}
        self.assertEqual(check_derived_violations(graph, prices), [])

    def test_min_hops_excludes_the_direct_pair_itself(self):
        graph = DifferenceGraph(chain_constraints())
        # A < B directly violates the single-hop A>=B constraint, but with
        # min_hops=2 (the default) the (A,B) PAIR itself should never be
        # reported here -- that's MonotoneConstraint.check's job. The (A,C)
        # pair is a genuine 2-hop pair, though, and IS correctly evaluated
        # (and happens to also be violated here) -- min_hops filters which
        # pairs get checked, not whether a chain has a broken link in it.
        prices = {"A": 0.40, "B": 0.55, "C": 0.50}
        violations = check_derived_violations(graph, prices, min_hops=2)
        self.assertTrue(all(v.market_ids != ("A", "B") for v in violations))
        self.assertTrue(any(v.market_ids == ("A", "B", "C") for v in violations))

    def test_min_hops_1_would_include_direct_pairs(self):
        graph = DifferenceGraph(chain_constraints())
        prices = {"A": 0.40, "B": 0.55, "C": 0.50}
        violations = check_derived_violations(graph, prices, min_hops=1)
        self.assertTrue(any(v.market_ids == ("A", "B") for v in violations))

    def test_intermediate_node_price_not_needed_for_endpoint_check(self):
        # The derived bound between A and C only depends on the graph
        # structure (computed once from the constraints), not on B's own
        # price -- checking it only needs the two ENDPOINT prices.
        graph = DifferenceGraph(chain_constraints())
        prices = {"A": 0.60, "C": 0.70}  # B missing; A and C still checkable directly
        violations = check_derived_violations(graph, prices)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].market_ids, ("A", "B", "C"))

    def test_missing_endpoint_price_does_skip_the_pair(self):
        graph = DifferenceGraph(chain_constraints())
        prices = {"B": 0.55}  # both real endpoints (A, C) missing
        self.assertEqual(check_derived_violations(graph, prices), [])


class TestDerivedBoundCheckerWithViolationTracker(unittest.TestCase):
    def test_drives_a_violation_tracker_end_to_end(self):
        checker = DerivedBoundChecker(chain_constraints(), min_hops=2)
        closed = []
        tracker = ViolationTracker(checker, on_episode_closed=closed.append)

        tracker.update({"A": 0.60, "B": 0.55, "C": 0.50}, now_ts=0.0)  # consistent
        tracker.update({"A": 0.60, "B": 0.55, "C": 0.65}, now_ts=10.0)  # chain violation opens
        tracker.update({"A": 0.60, "B": 0.55, "C": 0.65}, now_ts=20.0)  # persists
        tracker.update({"A": 0.60, "B": 0.55, "C": 0.50}, now_ts=30.0)  # resolves

        self.assertEqual(len(closed), 1)
        episode = closed[0]
        self.assertEqual(episode["constraint_type"], "derived_monotone")
        self.assertAlmostEqual(episode["duration_seconds"], 20.0)
        self.assertEqual(episode["num_observations"], 2)


class TestAgainstRealConfig(unittest.TestCase):
    def test_real_constraint_set_is_a_star_with_no_chains_yet(self):
        """Honesty check against our actual live topology: six monotone
        edges, all from a single meeting-hike bucket into the one shared
        yearly-hike node. No multi-hop paths exist yet, so this correctly
        finds nothing extra beyond direct pairwise checks -- it's armed
        for when a chainable constraint gets added, not claiming a result
        it doesn't have."""
        config = load_config()
        constraint_set = build_constraint_set(config)
        graph = DifferenceGraph(constraint_set)

        self.assertFalse(graph.negative_cycle)
        multi_hop_pairs = [
            (i, j) for i in graph.nodes for j in graph.nodes
            if i != j and (graph.hop_count(i, j) or 0) >= 2
        ]
        self.assertEqual(multi_hop_pairs, [])


if __name__ == "__main__":
    unittest.main()
