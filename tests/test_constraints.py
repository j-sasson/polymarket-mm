import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup


class TestNegativeRiskGroup(unittest.TestCase):
    def setUp(self):
        # Hand-built CPI print buckets: mutually exclusive & exhaustive.
        self.group = NegativeRiskGroup(
            name="cpi_august_2026_buckets",
            market_ids=("cpi_lt_2.0", "cpi_2.0_2.5", "cpi_2.5_3.0", "cpi_gt_3.0"),
            tolerance=0.02,
        )
        self.constraints = ConstraintSet(negative_risk_groups=[self.group])

    def test_consistent_prices_sum_to_one_no_violation(self):
        prices = {
            "cpi_lt_2.0": 0.05,
            "cpi_2.0_2.5": 0.25,
            "cpi_2.5_3.0": 0.55,
            "cpi_gt_3.0": 0.15,
        }
        self.assertEqual(self.constraints.check(prices), [])

    def test_overpriced_buckets_flagged_as_violation(self):
        # Sums to 1.15 -- classic "buy every bucket" arbitrage overpricing.
        prices = {
            "cpi_lt_2.0": 0.05,
            "cpi_2.0_2.5": 0.30,
            "cpi_2.5_3.0": 0.60,
            "cpi_gt_3.0": 0.20,
        }
        violations = self.constraints.check(prices)
        self.assertEqual(len(violations), 1)
        v = violations[0]
        self.assertEqual(v.constraint_type, "negative_risk")
        self.assertEqual(v.constraint_name, "cpi_august_2026_buckets")
        self.assertAlmostEqual(v.magnitude, 0.15, places=4)

    def test_underpriced_buckets_also_flagged(self):
        # Sums to 0.80 -- "sell every bucket" arbitrage underpricing.
        prices = {
            "cpi_lt_2.0": 0.05,
            "cpi_2.0_2.5": 0.15,
            "cpi_2.5_3.0": 0.40,
            "cpi_gt_3.0": 0.20,
        }
        violations = self.constraints.check(prices)
        self.assertEqual(len(violations), 1)
        self.assertAlmostEqual(violations[0].magnitude, 0.20, places=4)

    def test_within_tolerance_not_flagged(self):
        # Sums to 1.015, inside the 0.02 tolerance band.
        prices = {
            "cpi_lt_2.0": 0.05,
            "cpi_2.0_2.5": 0.25,
            "cpi_2.5_3.0": 0.555,
            "cpi_gt_3.0": 0.16,
        }
        self.assertEqual(self.constraints.check(prices), [])

    def test_missing_market_skips_constraint(self):
        prices = {"cpi_lt_2.0": 0.05, "cpi_2.0_2.5": 0.25}  # incomplete
        self.assertEqual(self.constraints.check(prices), [])

    def test_group_requires_at_least_two_markets(self):
        with self.assertRaises(ValueError):
            NegativeRiskGroup(name="bad", market_ids=("only_one",))


class TestMonotoneConstraint(unittest.TestCase):
    def setUp(self):
        # "Cuts by June" is a superset of "cuts by March" (June includes
        # every path where a cut already happened by March), so its price
        # must be >= the March market's price.
        self.constraint = MonotoneConstraint(
            name="fed_cuts_march_implies_june",
            superset_id="fed_cuts_by_june",
            subset_id="fed_cuts_by_march",
        )
        self.constraints = ConstraintSet(monotone_constraints=[self.constraint])

    def test_consistent_prices_no_violation(self):
        prices = {"fed_cuts_by_march": 0.30, "fed_cuts_by_june": 0.55}
        self.assertEqual(self.constraints.check(prices), [])

    def test_superset_priced_below_subset_is_violation(self):
        # March priced ABOVE June -- logically impossible, clear violation.
        prices = {"fed_cuts_by_march": 0.60, "fed_cuts_by_june": 0.45}
        violations = self.constraints.check(prices)
        self.assertEqual(len(violations), 1)
        v = violations[0]
        self.assertEqual(v.constraint_type, "monotonicity")
        self.assertEqual(v.market_ids, ("fed_cuts_by_june", "fed_cuts_by_march"))
        self.assertAlmostEqual(v.magnitude, 0.15, places=4)

    def test_equal_prices_not_a_violation(self):
        prices = {"fed_cuts_by_march": 0.40, "fed_cuts_by_june": 0.40}
        self.assertEqual(self.constraints.check(prices), [])

    def test_missing_market_skips_constraint(self):
        prices = {"fed_cuts_by_march": 0.40}
        self.assertEqual(self.constraints.check(prices), [])

    def test_same_market_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MonotoneConstraint(name="bad", superset_id="x", subset_id="x")


class TestCombinedConstraintSet(unittest.TestCase):
    def test_multiple_simultaneous_violations_all_reported(self):
        group = NegativeRiskGroup(
            name="cpi_buckets", market_ids=("cpi_a", "cpi_b"), tolerance=0.02
        )
        mono = MonotoneConstraint(
            name="fed_march_implies_june",
            superset_id="fed_june",
            subset_id="fed_march",
        )
        constraints = ConstraintSet(negative_risk_groups=[group], monotone_constraints=[mono])

        prices = {
            "cpi_a": 0.60,
            "cpi_b": 0.60,  # sums to 1.20 -> violation
            "fed_march": 0.70,
            "fed_june": 0.50,  # june < march -> violation
        }
        violations = constraints.check(prices)
        self.assertEqual(len(violations), 2)
        types = {v.constraint_type for v in violations}
        self.assertEqual(types, {"negative_risk", "monotonicity"})

    def test_fully_consistent_market_no_violations(self):
        group = NegativeRiskGroup(
            name="cpi_buckets", market_ids=("cpi_a", "cpi_b"), tolerance=0.02
        )
        mono = MonotoneConstraint(
            name="fed_march_implies_june",
            superset_id="fed_june",
            subset_id="fed_march",
        )
        constraints = ConstraintSet(negative_risk_groups=[group], monotone_constraints=[mono])

        prices = {
            "cpi_a": 0.45,
            "cpi_b": 0.55,
            "fed_march": 0.30,
            "fed_june": 0.55,
        }
        self.assertEqual(constraints.check(prices), [])


if __name__ == "__main__":
    unittest.main()
