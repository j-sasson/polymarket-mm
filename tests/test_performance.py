import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.performance import bucket_pnl, compute_performance, equity_curve, max_drawdown, sharpe_ratio


class TestEquityCurve(unittest.TestCase):
    def test_cumulative_sum_in_time_order(self):
        fills = [(2.0, -1.0), (0.0, 5.0), (1.0, 3.0)]  # deliberately out of order
        curve = equity_curve(fills)
        self.assertEqual(curve, [(0.0, 5.0), (1.0, 8.0), (2.0, 7.0)])

    def test_empty_fills(self):
        self.assertEqual(equity_curve([]), [])


class TestMaxDrawdown(unittest.TestCase):
    def test_no_drawdown_when_monotonically_increasing(self):
        curve = [(0, 1.0), (1, 2.0), (2, 3.0)]
        self.assertEqual(max_drawdown(curve), 0.0)

    def test_drawdown_from_peak(self):
        # peak at 10, trough at 4 -> drawdown of -6
        curve = [(0, 0.0), (1, 10.0), (2, 6.0), (3, 4.0), (4, 8.0)]
        self.assertEqual(max_drawdown(curve), -6.0)

    def test_recovers_then_drops_again_worse(self):
        curve = [(0, 0.0), (1, 10.0), (2, 5.0), (3, 12.0), (4, 1.0)]
        # second dip: peak 12 -> trough 1 = -11, worse than first dip (10->5=-5)
        self.assertEqual(max_drawdown(curve), -11.0)

    def test_empty_curve(self):
        self.assertEqual(max_drawdown([]), 0.0)


class TestBucketPnl(unittest.TestCase):
    def test_sums_within_buckets_and_fills_empty_gaps(self):
        fills = [(0.0, 1.0), (5.0, 2.0), (65.0, 3.0)]  # bucket width 60s -> buckets 0, 1(empty), ... no: 65/60=1
        buckets = bucket_pnl(fills, bucket_seconds=60.0)
        # bucket 0 covers [0,60): 1.0+2.0=3.0; bucket 1 covers [60,120): 3.0
        self.assertEqual(buckets, [(0.0, 3.0), (60.0, 3.0)])

    def test_gap_bucket_filled_with_zero(self):
        fills = [(0.0, 5.0), (185.0, -2.0)]  # bucket 0 and bucket 3 (185//60=3), buckets 1,2 empty
        buckets = bucket_pnl(fills, bucket_seconds=60.0)
        self.assertEqual(len(buckets), 4)
        self.assertEqual([pnl for _, pnl in buckets], [5.0, 0.0, 0.0, -2.0])

    def test_empty_fills(self):
        self.assertEqual(bucket_pnl([], bucket_seconds=60.0), [])


class TestSharpeRatio(unittest.TestCase):
    def test_none_with_fewer_than_two_buckets(self):
        self.assertIsNone(sharpe_ratio([1.0], periods_per_year=252))
        self.assertIsNone(sharpe_ratio([], periods_per_year=252))

    def test_none_with_zero_variance(self):
        self.assertIsNone(sharpe_ratio([1.0, 1.0, 1.0], periods_per_year=252))

    def test_positive_mean_positive_sharpe(self):
        result = sharpe_ratio([1.0, 2.0, 0.5, 1.5], periods_per_year=252)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_negative_mean_negative_sharpe(self):
        result = sharpe_ratio([-1.0, -2.0, -0.5, -1.5], periods_per_year=252)
        self.assertLess(result, 0)


class TestComputePerformance(unittest.TestCase):
    def test_full_report_on_hand_built_fills(self):
        fills = [(0.0, 1.0), (10.0, -0.5), (20.0, 2.0), (30.0, -3.0)]
        report = compute_performance(fills, bucket_seconds=10.0)

        self.assertEqual(report.num_fills, 4)
        self.assertAlmostEqual(report.total_pnl, sum(p for _, p in fills), places=6)
        self.assertEqual(report.hit_rate, 0.5)  # 2 of 4 fills positive
        self.assertEqual(report.equity_curve[-1][1], report.total_pnl)
        # peak is 2.5 (after fill 3: 1 - 0.5 + 2 = 2.5), trough after is 2.5-3=-0.5
        self.assertAlmostEqual(report.max_drawdown, -3.0, places=6)

    def test_empty_fills_returns_sane_defaults(self):
        report = compute_performance([], bucket_seconds=10.0)
        self.assertEqual(report.num_fills, 0)
        self.assertEqual(report.total_pnl, 0.0)
        self.assertEqual(report.hit_rate, 0.0)
        self.assertEqual(report.max_drawdown, 0.0)
        self.assertIsNone(report.sharpe)


if __name__ == "__main__":
    unittest.main()
