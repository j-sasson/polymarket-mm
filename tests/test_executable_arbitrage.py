import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup
from pmm_data.executable_arbitrage import BookTop, check_executable_arbitrage, taker_fee


class TestTakerFee(unittest.TestCase):
    def test_symmetric_around_midprice(self):
        self.assertAlmostEqual(taker_fee(0.30, 1.0, 0.04), taker_fee(0.70, 1.0, 0.04), places=9)

    def test_zero_at_extremes(self):
        self.assertAlmostEqual(taker_fee(0.0, 1.0, 0.04), 0.0, places=9)
        self.assertAlmostEqual(taker_fee(1.0, 1.0, 0.04), 0.0, places=9)

    def test_matches_documented_example(self):
        # docs.polymarket.com/trading/fees: 100 shares @ $0.24, feeRate=0.04 -> $0.73
        self.assertAlmostEqual(taker_fee(0.24, 100, 0.04), 0.7296, places=3)


class TestNegativeRiskArbitrageFeeAware(unittest.TestCase):
    def setUp(self):
        self.group = NegativeRiskGroup(name="grp", market_ids=("a", "b", "c"), tolerance=0.02)
        self.constraints = ConstraintSet(negative_risk_groups=[self.group])

    def test_real_october_fed_group_is_unprofitable_after_fees(self):
        """The exact books pulled live for the October Fed group: a 1.2% raw
        edge that looked like real arbitrage, but every leg must be sold by
        crossing the resting bid (taker), and Polymarket's live /fee-rate
        for these specific markets returned 10% -- which eats the entire
        edge and then some. This is the finding that killed the opportunity."""
        group5 = NegativeRiskGroup(name="oct_fed", market_ids=tuple(f"m{i}" for i in range(5)), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group5])
        prices = {"m0": 0.240, "m1": 0.011, "m2": 0.710, "m3": 0.044, "m4": 0.007}
        books = {m: BookTop(p, 150.0, p + 0.02, 150.0, taker_fee_rate=0.10) for m, p in prices.items()}

        # allow negative through for inspection -- both floors must be opened, they're independent gates
        results = check_executable_arbitrage(constraints, books, min_profit=-1.0, min_total_profit=-1000.0)
        mint_sell = [r for r in results if r.constraint_type == "mint_and_sell"]
        self.assertEqual(len(mint_sell), 1)
        arb = mint_sell[0]
        self.assertAlmostEqual(arb.gross_profit_per_set, 0.012, places=3)
        self.assertLess(arb.profit_per_set, 0)  # net negative -- fees exceed the edge
        self.assertAlmostEqual(arb.profit_per_set, -0.0328, places=3)

        # with the default min_profit=0.0 floor, this correctly reports NO opportunity
        filtered = check_executable_arbitrage(constraints, books)
        self.assertEqual(filtered, [])

    def test_low_fee_rate_can_still_leave_a_real_edge(self):
        books = {
            "a": BookTop(0.40, 500, 0.42, 500, taker_fee_rate=0.001),
            "b": BookTop(0.35, 500, 0.37, 500, taker_fee_rate=0.001),
            "c": BookTop(0.30, 500, 0.32, 500, taker_fee_rate=0.001),
        }  # bid sum = 1.05, tiny fee rate shouldn't wipe out a 5-cent edge
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertGreater(arb.profit_per_set, 0)
        self.assertLess(arb.profit_per_set, arb.gross_profit_per_set)  # fees reduced it, but didn't kill it
        self.assertAlmostEqual(arb.gross_profit_per_set, 0.05, places=4)

    def test_zero_fee_rate_matches_pre_fee_behavior(self):
        books = {
            "a": BookTop(0.40, 500, 0.42, 500, taker_fee_rate=0.0),
            "b": BookTop(0.35, 500, 0.37, 500, taker_fee_rate=0.0),
            "c": BookTop(0.30, 500, 0.32, 500, taker_fee_rate=0.0),
        }
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].profit_per_set, results[0].gross_profit_per_set, places=6)
        self.assertEqual(results[0].fee_per_set, 0.0)

    def test_max_size_capped_by_thinnest_leg(self):
        books = {
            "a": BookTop(0.40, 500, 0.42, 500),
            "b": BookTop(0.35, 30, 0.37, 500),    # thinnest bid depth: only 30
            "c": BookTop(0.30, 500, 0.32, 500),
        }
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].max_size, 30)

    def test_min_total_profit_filters_deep_but_thin_edge(self):
        books = {
            "a": BookTop(0.40, 2, 0.42, 500),
            "b": BookTop(0.35, 500, 0.37, 500),
            "c": BookTop(0.30, 500, 0.32, 500),
        }
        results = check_executable_arbitrage(self.constraints, books, min_total_profit=1.0)
        self.assertEqual(results, [])

    def test_buy_the_set_arb_when_asks_sum_under_one(self):
        books = {
            "a": BookTop(0.15, 500, 0.20, 500, taker_fee_rate=0.001),
            "b": BookTop(0.15, 500, 0.20, 500, taker_fee_rate=0.001),
            "c": BookTop(0.15, 500, 0.20, 500, taker_fee_rate=0.001),
        }  # ask sum = 0.60
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "buy_the_set")
        self.assertAlmostEqual(arb.gross_profit_per_set, 0.40, places=4)
        self.assertGreater(arb.profit_per_set, 0)

    def test_wide_spread_with_midpoint_violation_but_no_arb(self):
        books = {
            "a": BookTop(0.30, 500, 0.70, 500),
            "b": BookTop(0.25, 500, 0.55, 500),
            "c": BookTop(0.30, 500, 0.40, 500),
        }
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(results, [])

    def test_missing_market_skips_group(self):
        books = {"a": BookTop(0.40, 500, 0.42, 500), "b": BookTop(0.35, 500, 0.37, 500)}  # c missing
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(results, [])


class TestMonotoneArbitrage(unittest.TestCase):
    def test_subset_bid_above_superset_ask_is_arbitrage_with_low_fees(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {
            "march": BookTop(0.62, 200, 0.64, 200, taker_fee_rate=0.001),
            "june": BookTop(0.53, 200, 0.55, 80, taker_fee_rate=0.001),
        }
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "monotonicity")
        self.assertAlmostEqual(arb.gross_profit_per_set, 0.62 - 0.55, places=4)
        self.assertGreater(arb.profit_per_set, 0)
        self.assertEqual(arb.max_size, 80)

    def test_consistent_books_no_arbitrage(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {
            "march": BookTop(0.28, 200, 0.30, 200),
            "june": BookTop(0.53, 200, 0.55, 200),
        }
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(results, [])

    def test_overlapping_spreads_not_yet_crossed_no_arbitrage(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {
            "march": BookTop(0.38, 200, 0.40, 200),
            "june": BookTop(0.42, 200, 0.44, 200),
        }
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
