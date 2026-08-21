import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup
from pmm_data.executable_arbitrage import check_executable_arbitrage


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
        books = {
            "m0": (0.019, 0.053), "m1": (0.320, 0.350), "m2": (0.410, 0.440), "m3": (0.200, 0.220),
            "m4": (0.010, 0.045), "m5": (0.030, 0.046), "m6": (0.010, 0.041), "m7": (0.002, 0.009),
            "m8": (0.002, 0.010), "m9": (0.003, 0.017),
        }
        unfiltered = check_executable_arbitrage(constraints, books)
        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(unfiltered[0].constraint_type, "mint_and_sell")
        self.assertAlmostEqual(unfiltered[0].profit_per_set, 0.006, places=3)

        filtered = check_executable_arbitrage(constraints, books, min_profit=0.02)
        self.assertEqual(filtered, [])

    def test_mint_and_sell_arb_when_bids_sum_over_one(self):
        books = {"a": (0.40, 0.42), "b": (0.35, 0.37), "c": (0.30, 0.32)}  # bid sum = 1.05
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "mint_and_sell")
        self.assertAlmostEqual(arb.profit_per_set, 0.05, places=4)

    def test_buy_the_set_arb_when_asks_sum_under_one(self):
        books = {"a": (0.20, 0.25), "b": (0.20, 0.25), "c": (0.20, 0.25)}  # ask sum = 0.75
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "buy_the_set")
        self.assertAlmostEqual(arb.profit_per_set, 0.25, places=4)

    def test_wide_spread_with_midpoint_violation_but_no_arb(self):
        # midpoints: 0.5+0.4+0.35=1.25 (would flag under the statistical
        # checker), but bid sum=0.85 and ask sum=1.65 -- no real arb.
        books = {"a": (0.30, 0.70), "b": (0.25, 0.55), "c": (0.30, 0.40)}
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(results, [])

    def test_min_profit_floor_filters_thin_edges(self):
        books = {"a": (0.400, 0.42), "b": (0.350, 0.37), "c": (0.255, 0.32)}  # bid sum = 1.005
        results = check_executable_arbitrage(self.constraints, books, min_profit=0.01)
        self.assertEqual(results, [])  # 0.5% edge doesn't clear a 1% floor

    def test_missing_market_skips_group(self):
        books = {"a": (0.40, 0.42), "b": (0.35, 0.37)}  # c missing
        results = check_executable_arbitrage(self.constraints, books)
        self.assertEqual(results, [])


class TestMonotoneArbitrage(unittest.TestCase):
    def test_subset_bid_above_superset_ask_is_arbitrage(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {"march": (0.62, 0.64), "june": (0.53, 0.55)}  # march bid 0.62 > june ask 0.55
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(len(results), 1)
        arb = results[0]
        self.assertEqual(arb.constraint_type, "monotonicity")
        self.assertAlmostEqual(arb.profit_per_set, 0.62 - 0.55, places=4)

    def test_consistent_books_no_arbitrage(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {"march": (0.28, 0.30), "june": (0.53, 0.55)}
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(results, [])

    def test_overlapping_spreads_not_yet_crossed_no_arbitrage(self):
        # subset bid (0.40) still below superset ask (0.44) -- spreads
        # overlap but haven't actually crossed, so no real arb yet.
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        books = {"march": (0.38, 0.40), "june": (0.42, 0.44)}
        results = check_executable_arbitrage(constraints, books)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
