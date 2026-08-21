import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.trading.risk_limits import KillSwitch, MarketLimits, breach_reason, check_order


class FakeLogger:
    def __init__(self):
        self.events = []
        self.cancel_alls = []

    def log_event(self, event_type, detail):
        self.events.append((event_type, detail))

    def log_cancel_all(self, reason, dry_run, order_ids, response=""):
        self.cancel_alls.append((reason, dry_run, order_ids, response))


class FakeClient:
    def __init__(self, raise_on_cancel_all=False):
        self.raise_on_cancel_all = raise_on_cancel_all
        self.cancel_all_called = False

    def cancel_all(self):
        self.cancel_all_called = True
        if self.raise_on_cancel_all:
            raise RuntimeError("network error")
        return {"canceled": ["o1", "o2"]}


class TestCheckOrder(unittest.TestCase):
    def setUp(self):
        self.limits = MarketLimits(max_position_shares=100, max_exposure_usd=50, max_order_size_shares=20)

    def test_within_all_limits_ok(self):
        ok, reason = check_order(self.limits, current_position_shares=0, price=0.40, proposed_size=10, side="BUY")
        self.assertTrue(ok)

    def test_order_size_over_max_rejected(self):
        ok, reason = check_order(self.limits, current_position_shares=0, price=0.40, proposed_size=25, side="BUY")
        self.assertFalse(ok)
        self.assertIn("max_order_size_shares", reason)

    def test_resulting_position_over_cap_rejected(self):
        ok, reason = check_order(self.limits, current_position_shares=95, price=0.10, proposed_size=10, side="BUY")
        self.assertFalse(ok)
        self.assertIn("max_position_shares", reason)

    def test_resulting_exposure_over_cap_rejected(self):
        # position stays within share cap (100) but $ exposure at this price exceeds max_exposure_usd
        limits = MarketLimits(max_position_shares=1000, max_exposure_usd=50, max_order_size_shares=1000)
        ok, reason = check_order(limits, current_position_shares=0, price=0.90, proposed_size=100, side="BUY")
        self.assertFalse(ok)
        self.assertIn("max_exposure_usd", reason)

    def test_sell_reduces_position_and_is_allowed_even_near_cap(self):
        ok, reason = check_order(self.limits, current_position_shares=95, price=0.10, proposed_size=10, side="SELL")
        self.assertTrue(ok)

    def test_non_positive_size_rejected(self):
        ok, reason = check_order(self.limits, current_position_shares=0, price=0.4, proposed_size=0, side="BUY")
        self.assertFalse(ok)


class TestBreachReason(unittest.TestCase):
    def test_no_breach_returns_none(self):
        limits = MarketLimits(max_position_shares=100, max_exposure_usd=100, max_order_size_shares=20)
        self.assertIsNone(breach_reason(limits, position_shares=50, mark_price=0.5))

    def test_position_breach_detected(self):
        limits = MarketLimits(max_position_shares=100, max_exposure_usd=1000, max_order_size_shares=20)
        reason = breach_reason(limits, position_shares=150, mark_price=0.1)
        self.assertIsNotNone(reason)
        self.assertIn("max_position_shares", reason)

    def test_exposure_breach_detected(self):
        limits = MarketLimits(max_position_shares=1000, max_exposure_usd=10, max_order_size_shares=1000)
        reason = breach_reason(limits, position_shares=50, mark_price=0.9)  # exposure = 45 > 10
        self.assertIsNotNone(reason)
        self.assertIn("max_exposure_usd", reason)


class TestKillSwitch(unittest.TestCase):
    def test_dry_run_never_touches_client(self):
        client = FakeClient()
        logger = FakeLogger()
        ks = KillSwitch(client, logger, dry_run=True)
        ks.trigger("test reason", ["o1", "o2"])

        self.assertTrue(ks.triggered)
        self.assertFalse(client.cancel_all_called)
        self.assertEqual(len(logger.cancel_alls), 1)
        self.assertTrue(logger.cancel_alls[0][1])  # dry_run flag logged True

    def test_live_calls_cancel_all(self):
        client = FakeClient()
        logger = FakeLogger()
        ks = KillSwitch(client, logger, dry_run=False)
        ks.trigger("test reason", ["o1"])

        self.assertTrue(client.cancel_all_called)
        self.assertTrue(ks.triggered)

    def test_cancel_all_failure_still_marks_triggered_and_reraises(self):
        client = FakeClient(raise_on_cancel_all=True)
        logger = FakeLogger()
        ks = KillSwitch(client, logger, dry_run=False)
        with self.assertRaises(RuntimeError):
            ks.trigger("test reason", ["o1"])
        self.assertTrue(ks.triggered)  # still marked triggered even though cancel_all itself failed
        self.assertTrue(any(e[0] == "kill_switch_cancel_all_FAILED" for e in logger.events))


if __name__ == "__main__":
    unittest.main()
