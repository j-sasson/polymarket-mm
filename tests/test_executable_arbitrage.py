import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup
from pmm_data.executable_arbitrage import BookTop, check_executable_arbitrage


class TestNegativeRiskArbitrage(unittest.TestCase):
    def setUp(self):
        self.group = NegativeRiskGroup(name="grp", market_ids=("a", "b", "c"), tolerance=0.02)
        self.constraints = ConstraintSet(negative_risk_groups=[self.group])

    def test_real_cpi_books_show_only_a_fee_losing_edge_not_real_arbitrage(self):
        """The exact books pulled live for the CPI group: midpoints summed
        to 1.119 (a 'violation' by the statistical checker), best-ask sum
        was 1.231 (no buy-side arb), and best-bid sum was 1.006 -- technically
        over 1, but only by 0.6%, which is exactly the kind of razor-thin
        edge Polymarket's fees would eat. A realistic min_profit floor
        correctly filters it; an unfiltered check still flags it (correctly
        -- it's not this function's job to know the fee schedule)."""
        group10 = NegativeRiskGroup(name="cpi", market_ids=tuple(f"m{i}" for i in range(10)), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group10])
        prices = {
            "m0": (0.019, 0.053), "m1": (0.320, 0.350), "m2": (0.410, 0.440), "m3": (0.200, 0.220),
            "m4": (0.010, 0.045), "m5": (0.030, 0.046), "m6": (0.010, 0.041), "m7": (0.002, 0.009),
            "m8": (0.002, 0.010), "m9": (0.003, 0.017),
        }
        books = {m: BookTop(bid, 100.0, ask, 100.0) for m, (bid, ask) in prices.items()}

        unfiltered = check_executable_arbitrage(constraints, books)
        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(unfiltered[0].constraint_type, "mint_and_sell")
        self.assertAlmostEqual(unfiltered[0].profit_per_set, 0.006, places=3)

        filtered = check_executable_arbitrage(constraints, books, min_profit=0.02)
        self.assertEqual(filtered, [])

    def test_mint_and_sell_arb_when_bids_sum_over_one(self):
        books = {
            "a": BookTop(0.40, 500, 0.42, 500),
            "b": BookTop(0.35, 500, 0.37, 500),
            "c": BookTop(0.30, 500, 0.32, 500),
        }  # bid sum = 1.05
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "mint_and_sell")
        self.assertAlmostEqual(arb.profit_per_set, 0.05, places=4)
        self.assertEqual(arb.max_size, 500)
        self.assertAlmostEqual(arb.total_profit, 25.0, places=4)

    def test_max_size_capped_by_thinnest_leg(self):
        books = {
            "a": BookTop(0.40, 500, 0.42, 500),
            "b": BookTop(0.35, 30, 0.37, 500),    # thinnest bid depth: only 30
            "c": BookTop(0.30, 500, 0.32, 500),
        }  # bid sum = 1.05
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].max_size, 30)
        self.assertAlmostEqual(results[0].total_profit, 0.05 * 30, places=4)

    def test_min_total_profit_filters_deep_but_thin_edge(self):
        # 0.05 edge per set (well above min_profit), but only 2 shares deep
        # -> $0.10 total, filtered out by a $1 floor even though per-set profit looks fine.
        books = {
            "a": BookTop(0.40, 2, 0.42, 500),
            "b": BookTop(0.35, 500, 0.37, 500),
            "c": BookTop(0.30, 500, 0.32, 500),
        }
        results = check_executable_arbitrage(self.constraints, books, min_total_profit=1.0)
        self.assertEqual(results, [])

    def test_buy_the_set_arb_when_asks_sum_under_one(self):
        books = {
            "a": BookTop(0.15, 500, 0.20, 500),
            "b": BookTop(0.15, 500, 0.20, 500),
            "c": BookTop(0.15, 500, 0.20, 500),
        }  # ask sum = 0.60
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "buy_the_set")
        self.assertAlmostEqual(arb.profit_per_set, 0.40, places=4)
        self.assertEqual(arb.max_size, 500)

    def test_wide_spread_with_midpoint_violation_but_no_arb(self):
        # midpoints: 0.5+0.4+0.35=1.25 (would flag under the statistical
        # checker), but bid sum=0.85 and ask sum=1.65 -- no real arb.
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
    def test_subset_bid_above_superset_ask_is_arbitrage(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {
            "march": BookTop(0.62, 200, 0.64, 200),
            "june": BookTop(0.53, 200, 0.55, 80),  # thinnest: superset ask depth = 80
        }
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "monotonicity")
        self.assertAlmostEqual(arb.profit_per_set, 0.62 - 0.55, places=4)
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
        # subset bid (0.40) still below superset ask (0.44) -- spreads
        # overlap but haven't actually crossed, so no real arb yet.
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
