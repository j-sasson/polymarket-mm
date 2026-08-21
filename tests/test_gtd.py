import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.trading.gtd import compute_gtd_expiration


class TestComputeGtdExpiration(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)

    def test_plenty_of_runway_uses_desired_lifetime(self):
        catalyst = self.now + timedelta(days=7)
        exp = compute_gtd_expiration(self.now, catalyst, desired_lifetime_seconds=300, pre_catalyst_buffer_seconds=120)
        self.assertIsNotNone(exp)
        expected = int((self.now + timedelta(seconds=300)).timestamp())
        self.assertEqual(exp, expected)

    def test_catalyst_close_but_sufficient_caps_at_catalyst_buffer(self):
        # catalyst in 7 minutes: safe runway = 420s - 180s buffer = 240s,
        # which is less than the desired 300s lifetime, so expiration should
        # be capped at the safe runway instead of the full desired lifetime.
        catalyst = self.now + timedelta(minutes=7)
        exp = compute_gtd_expiration(self.now, catalyst, desired_lifetime_seconds=300, pre_catalyst_buffer_seconds=120)
        self.assertIsNotNone(exp)
        latest_safe = catalyst - timedelta(seconds=120 + 60)  # + GTD's own 60s early-expiry buffer
        self.assertEqual(exp, int(latest_safe.timestamp()))
        self.assertLess(exp - int(self.now.timestamp()), 300)  # confirms it was actually capped, not coincidence

    def test_catalyst_too_close_returns_none(self):
        catalyst = self.now + timedelta(minutes=2)  # far less than the 3-minute Polymarket minimum
        exp = compute_gtd_expiration(self.now, catalyst, desired_lifetime_seconds=300, pre_catalyst_buffer_seconds=120)
        self.assertIsNone(exp)

    def test_catalyst_already_passed_returns_none(self):
        catalyst = self.now - timedelta(minutes=5)
        exp = compute_gtd_expiration(self.now, catalyst, desired_lifetime_seconds=300, pre_catalyst_buffer_seconds=120)
        self.assertIsNone(exp)

    def test_never_returns_expiration_inside_polymarket_minimum_lifetime(self):
        catalyst = self.now + timedelta(minutes=4)
        exp = compute_gtd_expiration(self.now, catalyst, desired_lifetime_seconds=300, pre_catalyst_buffer_seconds=120)
        if exp is not None:
            self.assertGreaterEqual(exp - int(self.now.timestamp()), 180)


if __name__ == "__main__":
    unittest.main()
