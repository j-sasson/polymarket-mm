import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.backtest import BacktestConfig, fill_probability, next_skew, run_backtest
from pmm_data.constraints import ConstraintSet, NegativeRiskGroup


@dataclass
class Event:
    timestamp_s: float
    market_id: str
    best_bid: float
    best_ask: float


class BidOnlyRNG:
    """Deterministic stub: fills every bid draw, never fills an ask draw.
    _quote_and_fill always draws bid then ask, so alternating 0.0/1.0 by call
    count reliably produces one-sided (BUY-only) fills across a run."""

    def __init__(self):
        self.n = 0

    def random(self) -> float:
        self.n += 1
        return 0.0 if self.n % 2 == 1 else 1.0


class ConstantRNG:
    """Deterministic stand-in for random.Random: every draw returns `value`,
    so `rng.random() < prob` is controlled purely by `prob` vs. this constant."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


class TestFillProbability(unittest.TestCase):
    def test_zero_edge_equals_base_rate(self):
        self.assertAlmostEqual(fill_probability(edge=0.0, spread=0.02, base_rate=0.15), 0.15, places=6)

    def test_positive_edge_increases_probability(self):
        p = fill_probability(edge=0.02, spread=0.02, base_rate=0.15)
        self.assertGreater(p, 0.15)

    def test_negative_edge_decreases_probability(self):
        p = fill_probability(edge=-0.02, spread=0.02, base_rate=0.15)
        self.assertLess(p, 0.15)

    def test_capped(self):
        p = fill_probability(edge=10.0, spread=0.01, base_rate=0.5, cap=0.9)
        self.assertEqual(p, 0.9)


class TestNextSkew(unittest.TestCase):
    def test_no_time_elapsed_no_decay(self):
        self.assertAlmostEqual(next_skew(0.05, dt=0.0, taper_seconds=60.0), 0.05, places=6)

    def test_one_time_constant_decays_to_1_over_e(self):
        import math
        result = next_skew(0.05, dt=60.0, taper_seconds=60.0)
        self.assertAlmostEqual(result, 0.05 * math.exp(-1), places=6)

    def test_zero_taper_snaps_to_zero_immediately(self):
        self.assertEqual(next_skew(0.05, dt=1.0, taper_seconds=0.0), 0.0)

    def test_tiny_residual_snapped_to_exact_zero(self):
        result = next_skew(1e-8, dt=1.0, taper_seconds=1.0)
        self.assertEqual(result, 0.0)


class TestRunBacktestIntegration(unittest.TestCase):
    def test_skew_avoids_adverse_bid_and_captures_correct_side(self):
        """An overpriced negative-risk group (fair value below raw mid) that
        genuinely reverts down: the fair-value strategy should skew its
        quotes down, meaning it (a) never gets picked off buying on the bid
        right before the drop, unlike baseline, and (b) its ask fills mark
        as profitable once the price has actually fallen."""
        group = NegativeRiskGroup(name="grp", market_ids=("a", "b"), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group])
        config = BacktestConfig(
            half_spread=0.01, skew_aggressiveness=1.0, taper_seconds=60.0,
            fill_base_rate=0.15, mark_horizon_seconds=30.0,
        )
        # threshold: passes for baseline (prob=0.15) and the aggressive skewed
        # ask (prob near cap), fails for the passive skewed bid (prob ~0.001)
        rng = ConstantRNG(0.05)

        events = [
            Event(0.0, "a", 0.59, 0.61),   # a alone: group incomplete, no violation yet
            Event(0.1, "b", 0.59, 0.61),   # b joins: sum=1.20 -> violation, fair=.50 each
            Event(0.2, "a", 0.59, 0.61),   # a picks up the now-active violation, skews down
            Event(35.0, "a", 0.49, 0.51),  # market reverts down to the fair value
            Event(35.1, "b", 0.49, 0.51),
        ]

        results = run_backtest(events, constraints, config, rng=rng)
        baseline, fair_value = results["baseline"], results["fair_value"]

        baseline_a_bids = [f for f in baseline.fills if f.market_id == "a" and f.side == "bid"]
        self.assertTrue(baseline_a_bids, "baseline should have bought on market a's bid")
        self.assertIsNotNone(baseline_a_bids[0].pnl)
        self.assertLess(baseline_a_bids[0].pnl, 0, "baseline got picked off buying right before the drop")

        # market a's FIRST event (t=0.0) fires before it has heard about b's
        # price at all, so the violation isn't knowable yet and both
        # strategies quote identically there -- that's correct, not a bug.
        # The skew only applies once a's own event picks up the now-active
        # violation (t=0.2 onward), so that's the window we check.
        fair_value_a_bids_post_skew = [
            f for f in fair_value.fills if f.market_id == "a" and f.side == "bid" and f.fill_time >= 0.2
        ]
        self.assertEqual(fair_value_a_bids_post_skew, [], "fair-value strategy should have skewed its bid out of the way")

        fair_value_a_asks_post_skew = [
            f for f in fair_value.fills if f.market_id == "a" and f.side == "ask" and f.fill_time >= 0.2
        ]
        self.assertTrue(fair_value_a_asks_post_skew, "fair-value strategy should have sold on the skewed-down ask")
        self.assertIsNotNone(fair_value_a_asks_post_skew[0].pnl)
        self.assertGreater(fair_value_a_asks_post_skew[0].pnl, 0, "selling ahead of the drop should mark profitable")

    def test_inventory_strategy_skews_down_after_accumulating_a_long_position(self):
        """fair_value and fair_value_inventory start identical (no violation,
        no position). Force one-sided (BUY-only) fills so fair_value_inventory
        builds a long position while ignoring-inventory fair_value doesn't
        change behavior; the inventory strategy's later bid should skew
        below fair_value's, which never accounts for position."""
        constraints = ConstraintSet()  # no violations -- isolates the inventory effect alone
        config = BacktestConfig(half_spread=0.05, fill_base_rate=0.9, inventory_skew_strength=1.0,
                                 inventory_max_position=5.0, mark_horizon_seconds=1000.0)
        rng = BidOnlyRNG()

        events = [Event(float(i), "a", 0.40, 0.50) for i in range(10)]
        results = run_backtest(events, constraints, config, rng=rng)

        fv_last_bid = [f for f in results["fair_value"].fills if f.side == "bid"][-1]
        fvi_last_bid = [f for f in results["fair_value_inventory"].fills if f.side == "bid"][-1]
        self.assertLess(fvi_last_bid.fill_price, fv_last_bid.fill_price)

    def test_fills_near_end_of_data_left_unmarked_not_guessed(self):
        group = NegativeRiskGroup(name="grp", market_ids=("a", "b"), tolerance=0.02)
        constraints = ConstraintSet(negative_risk_groups=[group])
        config = BacktestConfig(mark_horizon_seconds=30.0, fill_base_rate=0.9)
        rng = ConstantRNG(0.01)  # force fills

        events = [
            Event(0.0, "a", 0.40, 0.42),
            Event(0.1, "b", 0.58, 0.60),  # sum ~1.0, no violation, just need a fillable book
            Event(1.0, "a", 0.40, 0.42),  # a fill happens here, horizon = 31.0, never reached
        ]
        results = run_backtest(events, constraints, config, rng=rng)
        baseline = results["baseline"]
        self.assertGreater(baseline.num_fills, 0)
        self.assertGreater(baseline.num_unmarked, 0)
        self.assertEqual(baseline.marked_fills, [f for f in baseline.fills if f.pnl is not None])


if __name__ == "__main__":
    unittest.main()
