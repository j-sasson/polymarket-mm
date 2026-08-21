import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup
from pmm_data.fair_value import graph_fair_value


class TestGraphFairValue(unittest.TestCase):
    def test_negative_risk_renormalizes_to_sum_one(self):
        group = NegativeRiskGroup(name="cpi", market_ids=("a", "b", "c"), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group])
        prices = {"a": 0.3, "b": 0.3, "c": 0.6}  # sums to 1.2
        violations = constraints.check(prices)
        self.assertEqual(len(violations), 1)

        fair = graph_fair_value(prices, violations)
        self.assertAlmostEqual(sum(fair[m] for m in ("a", "b", "c")), 1.0, places=6)
        # proportional scaling preserves relative weights
        self.assertAlmostEqual(fair["a"] / fair["c"], prices["a"] / prices["c"], places=6)

    def test_monotonicity_splits_gap_evenly(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        prices = {"march": 0.60, "june": 0.40}  # march > june by 0.20
        violations = constraints.check(prices)

        fair = graph_fair_value(prices, violations)
        self.assertAlmostEqual(fair["march"], 0.50, places=6)
        self.assertAlmostEqual(fair["june"], 0.50, places=6)

    def test_untouched_markets_unchanged(self):
        group = NegativeRiskGroup(name="cpi", market_ids=("a", "b"), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group])
        prices = {"a": 0.6, "b": 0.6, "z": 0.42}  # z isn't in any constraint
        violations = constraints.check(prices)

        fair = graph_fair_value(prices, violations)
        self.assertEqual(fair["z"], 0.42)

    def test_no_violations_means_fair_equals_raw(self):
        group = NegativeRiskGroup(name="cpi", market_ids=("a", "b"), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group])
        prices = {"a": 0.4, "b": 0.6}
        violations = constraints.check(prices)
        self.assertEqual(violations, [])

        fair = graph_fair_value(prices, violations)
        self.assertEqual(fair, prices)


if __name__ == "__main__":
    unittest.main()
